# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# User-defined Modules
from transformers import BertModel
from data_classes import RewardModelBatch

# ------------------------- IMPLEMENTATION -----------------------------------


class RewardModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        model: BertModel,
        batch_size: int,
        initial_lr: Optional[float] = None
    ) -> None:
        super().__init__()
        self.model = model
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.linear = nn.Linear(model.config.hidden_size, 1)

    def forward(self, batch: RewardModelBatch) -> torch.FloatTensor:
        cls_hidden_state = self.model(
            input_ids=batch.dialogues,
            attention_mask=batch.mask,
            output_hidden_states=True).hidden_states[-1][:, 0, :]
        rewards = self.linear(cls_hidden_state).squeeze(dim=-1)
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
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.8
        )
        return ([optimizer], [{"scheduler": scheduler, "interval": "epoch"}])

# -----------------------------------------------------------------------------