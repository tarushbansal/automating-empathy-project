# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup


# User-defined Modules
from transformers import GPT2Model
from data_classes import RewardModelBatch

# ------------------------- IMPLEMENTATION -----------------------------------


class RewardModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        model: GPT2Model,
        batch_size: int,
        initial_lr: Optional[float] = None
    ) -> None:
        super().__init__()
        self.model = model
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.linear = nn.Linear(model.config.hidden_size, 1)

    def forward(self, batch: RewardModelBatch) -> torch.FloatTensor:
        last_hidden_state = self.model(
            input_ids=batch.dialogues.masked_fill(batch.mask == 0, 0),
            attention_mask=batch.mask).last_hidden_state
        last_idx = torch.sum(batch.mask, dim=-1, keepdim=True) - 1
        last_idx = last_idx.unsqueeze(-1).expand(-1, -1, last_hidden_state.size(dim=-1))
        dialogue_hidden_state = torch.gather(last_hidden_state, 1, last_idx).squeeze(1)
        rewards = self.linear(dialogue_hidden_state).squeeze(-1)
        return rewards

    def forward_and_log_metrics(
        self, 
        batch: RewardModelBatch,
        stage: str
    ) -> float:
        pred_rewards = self.forward(batch)
        loss = F.mse_loss(pred_rewards, batch.rewards)
        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True, 
            batch_size=self.batch_size, 
            sync_dist=True
        )
        self.logger.experiment.add_scalars('loss', {stage: loss}, self.global_step)
        return loss

    def training_step(
        self, 
        batch: RewardModelBatch,
        _
    ) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def validation_step(
        self, 
        batch: RewardModelBatch,
        _
    ) -> float:

        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.initial_lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.max_epochs
        )
        return ([optimizer], [{"scheduler": scheduler, "interval": "epoch"}])

# -----------------------------------------------------------------------------