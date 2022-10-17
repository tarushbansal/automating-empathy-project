# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import numpy as np
from typing import Dict, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# User-defined Modules
from model import GenerativeTransformer
from token_indexer import TokenIndexer
from utils import beam_search

# ------------------------- IMPLEMENTATION -----------------------------------

class ModelSupervisor(pl.LightningModule):
    def __init__(
        self, 
        max_seq_len: int,
        token_indexer: TokenIndexer,
        initial_lr: float,
        beam_width: int = 2
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore="token_indexer")
        self.model = GenerativeTransformer(
            vocab_size=token_indexer.vocab_size, 
            num_of_emo_labels=len(token_indexer.emo_map),
            max_seq_len=max_seq_len,
            padding_idx=token_indexer.PAD_IDX
        )
        self.initial_lr = initial_lr
        self.token_indexer = token_indexer
        self.max_seq_len = max_seq_len
        self.vocab_size = token_indexer.vocab_size
        self.beam_width = beam_width

    def forward(self, batch: Dict) -> torch.Tensor:
        out = self.model(
            input_seq=batch["dialogue"],
            dialogue_state=batch["dialogue_state"],
            emotion_label=batch["emotion"]
        )
        return out
    
    def forward_and_compute_loss(self, batch: Dict):
        logits = self.forward(batch)[:, :-1, :].permute(0, 2, 1)
        loss = F.cross_entropy(
            logits, 
            batch["dialogue"][:, 1:],
            ignore_index=self.token_indexer.PAD_IDX
        )
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> float:
        train_loss = self.forward_and_compute_loss(batch)
        self.log(
            "train_loss", 
            train_loss,
            prog_bar=True
        )
        self.logger.experiment.add_scalars(
            'loss', {'train': train_loss}, self.global_step) 
        return train_loss        

    def validation_step(self, batch: Dict, batch_idx: int) -> float:
        val_loss = self.forward_and_compute_loss(batch)
        if not self.trainer.sanity_checking:
            self.log(
                "val_loss", 
                val_loss, 
                prog_bar=True
            )
            self.logger.experiment.add_scalars(
                'loss', {'val': val_loss}, self.global_step) 
        return val_loss
    
    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        if not self.trainer.sanity_checking:
            self.log(
                "avg_val_loss", 
                avg_val_loss, 
                prog_bar=True
            )
            self.logger.experiment.add_scalars(
                'loss', {'avg_val': avg_val_loss}, self.global_step) 
        return avg_val_loss

    def test_step(self, batch: Dict, batch_idx: int) -> float:
        _, prob = beam_search(
            model=self.model,
            batch=batch,
            beam_width=self.beam_width,
            min_seq_len=batch["target"].size(dim=1) - 1,
            max_seq_len=self.max_seq_len,
            sos_token=self.token_indexer.SOS_IDX,
            eos_token=self.token_indexer.EOS_IDX
        )
        test_loss = -float(torch.log(prob).mean())
        self.log(
                "test_loss", 
                test_loss, 
                prog_bar=True
        )
        self.logger.experiment.add_scalars(
            'loss', {'test': test_loss}, self.global_step) 
        return test_loss

    def test_epoch_end(self, test_losses: List[float]) -> float:
        avg_test_loss = sum(test_losses) / len(test_losses)
        self.log(
            "avg_test_loss",
            avg_test_loss, 
            prog_bar=True
        )
        self.logger.experiment.add_scalars(
            'loss', {'avg_test': avg_test_loss}, self.global_step) 
        return avg_test_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

# -----------------------------------------------------------------------------