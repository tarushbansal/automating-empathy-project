# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from collections import OrderedDict
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

# User-defined Modules
from data_classes import RewardModelBatch
from base_classes import DialogueModelBase, TokenizerBase

# ------------------------- IMPLEMENTATION -----------------------------------


class RewardModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        config: Dict, # Must be in specificied format (See 'configs.json')
        batch_size:  Optional[int] = None,
        initial_lr: Optional[float] = None,
        dropout_prob: Optional[float] = 0.6,
    ) -> None:
        super().__init__()
        model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
        tokenizer_cls = getattr(__import__("custom_tokenizers"), config["tokenizer"]["cls"])
        self.tokenizer: TokenizerBase = tokenizer_cls(**config["tokenizer"]["kwargs"])
        self.model: DialogueModelBase = model_cls(
            vocab_size=self.tokenizer.vocab_size,
            **config["model"]["kwargs"]
        )

        # Sanity checks
        if not issubclass(model_cls, DialogueModelBase):
            raise ValueError("Model must be derived from base class 'DialogueModelBase'!")
        
        if not issubclass(tokenizer_cls, TokenizerBase):
            raise ValueError(
                "Tokenizer must be derived from base class 'TokenizerBase'!")

        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.linear = nn.Linear(self.model.hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.mean_offset = 0

        self.save_hyperparameters("config", "batch_size", "initial_lr")

    def forward(self, batch: RewardModelBatch) -> torch.FloatTensor:
        if self.model.has_encoder:
            output = self.model(
                contexts=batch.contexts,
                context_mask=batch.context_mask,
                targets=batch.targets,
                target_mask=batch.target_mask,
            )
            last_hidden_states = output.decoder_hidden_states[-1]
        else:
            output = self.model(
                dialogues=torch.cat((batch.contexts, batch.targets), dim=1),
                dialogue_mask=torch.cat((batch.context_mask, batch.target_mask), dim=1),
            )
            last_hidden_states = output.hidden_states[-1][:, batch.contexts.size(dim=1):, :]
        dialogue_hidden_state = torch.sum(
            last_hidden_states * batch.target_mask.unsqueeze(-1), dim=1
            ) / torch.sum(batch.target_mask, dim=1).unsqueeze(-1)
        rewards = self.linear(self.dropout(dialogue_hidden_state)).squeeze(-1)
        if not self.training:
            rewards -= self.mean_offset
        return rewards

    def compute_loss_and_accuracy(
        self, 
        rewards: torch.FloatTensor, 
        pairwise_ratings: List[Tuple[int]]
    ) -> float:
        sum_loss, num_correct, total = 0, 0, 0
        for rating in pairwise_ratings:
            A, B, factor = rating
            if factor == 0:
                sum_loss += (rewards[A] - rewards[B]) ** 2
            else:
                diff = (rewards[A] - rewards[B]) * (1 if factor < 0 else -1)
                sum_loss -= (abs(factor) ** 0.5) * torch.log(torch.sigmoid(diff))
                if diff > 0:
                    num_correct += 1
                total += 1
        mean_loss = sum_loss / len(pairwise_ratings)
        accuracy = num_correct / total
        return mean_loss, accuracy

    def forward_and_log_metrics(
        self, 
        batch: RewardModelBatch,
        stage: str
    ) -> float:
        pred_rewards = self.forward(batch)
        loss, accuracy = self.compute_loss_and_accuracy(pred_rewards, batch.ratings)
        N1 = batch.contexts.size(dim=0)
        N2 = len([rating for _, _, rating in batch.ratings if rating != 0])
        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True, 
            batch_size=N1, 
            sync_dist=True
        )
        self.log(
            f"{stage}_accuracy", 
            accuracy, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True, 
            batch_size=N2, 
            sync_dist=True
        )
        # Mean reward calculation for normalization
        if stage == "val":
            self.sum_rewards += torch.sum(pred_rewards)
            self.num += N1
        return loss

    def training_step(
        self, 
        batch: RewardModelBatch,
        _
    ) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def on_validation_epoch_start(self) -> None:
        self.sum_rewards, self.num = 0, 0

    def validation_step(
        self, 
        batch: RewardModelBatch,
        _
    ) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss
    
    def on_load_checkpoint(self, ckpt: OrderedDict) -> None:
        self.mean_offset = ckpt["state_dict"]["mean_offset"]
        del ckpt["state_dict"]["mean_offset"]

    def on_save_checkpoint(self, ckpt: OrderedDict):
        if hasattr(self, "sum_rewards") and hasattr(self, "num"):
            ckpt["state_dict"]["mean_offset"] = self.sum_rewards / self.num

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.initial_lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5
        )
        return ([optimizer], [{"scheduler": scheduler, "interval": "epoch"}])

# -----------------------------------------------------------------------------