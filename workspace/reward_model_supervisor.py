# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Optional, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


# User-defined Modules
from transformers import BertModel
from data.data_classes import RewardModelBatch

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

    def forward_and_log_metrics(
        self, 
        batch: RewardModelBatch,
        stage: str
    ) -> float:

        rewards = self.model(batch.dialogues)
        loss = F.mse_loss(rewards, batch.rewards)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {stage: loss}, self.global_step)
        
        return loss

    def training_step(
        self, 
        batch: RewardModelBatch,
        _
    ) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def test_step(
        self, 
        batch: RewardModelBatch,
        _
    ) -> float:

        test_loss = self.forward_and_log_metrics(batch, "test")
        return test_loss
    
    def test_step_end(self, test_losses: List[float]) -> float:
        avg_test_loss = sum(test_losses) / len(test_losses)
        self.log(f"avg_test_loss", avg_test_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {"avg_test": avg_test_loss}, self.global_step)
        return avg_test_loss

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