# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

# User-defined Modules
from base_classes import (
    DialogueModelBase, 
    TokenizerBase, 
    DecoderModel, 
    EncoderDecoderModel
)

# ------------------------- IMPLEMENTATION -----------------------------------

class ModelSupervisor(pl.LightningModule):
    def __init__(
        self, 
        model: DialogueModelBase,
        tokenizer: TokenizerBase,
        initial_lr: float,
        beam_width: int = 1,
        max_predict_seq_len: int = 100
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "model"])

        self.model = model
        self.tokenizer = tokenizer
        self.initial_lr = initial_lr
        self.beam_width = beam_width
        self.max_predict_seq_len = max_predict_seq_len

    def forward(self, batch: Dict) -> Tuple[torch.Tensor]:
        if issubclass(type(self.model), DecoderModel):
            input_seq = torch.cat((batch["context"], batch["target"]), dim=1)
            input_dialogue_state = torch.cat(
                (batch["context_dialogue_state"], batch["target_dialogue_state"]), dim=1)
            logits = self.model(
                input_seq=input_seq,
                input_dialogue_state=input_dialogue_state,
                emotion_label=batch["emotion"]
            )
            target = input_seq[:, 1:]
        elif issubclass(type(self.model), EncoderDecoderModel):
            logits = self.model(
                source_seq=batch["context"],
                target_seq=batch["target"],
                source_dialogue_state=batch["context_dialogue_state"],
                target_dialogue_state=batch["target_dialogue_state"],
                emotion_label=batch["emotion"]
            )
            target = batch["target"][:, 1:] 
        return logits, target
    
    def forward_and_log_metrics(self, batch: Dict, stage: str):
        logits, target = self.forward(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].permute(0, 2, 1), 
            target,
            ignore_index=self.tokenizer.PAD_IDX
        )
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.logger.experiment.add_scalars(
            'loss', {stage: loss}, self.global_step) 
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss        

    def validation_step(self, batch: Dict, batch_idx: int) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss
    
    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.logger.experiment.add_scalars(
            'loss', {'avg_val': avg_val_loss}, self.global_step) 

    def predict_step(self, batch: Dict, batch_idx: int = 0) -> torch.Tensor:
        ## BEAM SEARCH: Objective is to determine the most probable output sequence from the model
        
        # Set model in evaluation mode (Important to turn off dropout layers!!)
        self.model.eval()

        N, seq_len = batch["context"].size(dim=0), 1
        ds = batch["target_dialogue_state"][:, 0:1]
        batch_beam = torch.empty(N, 1, 1, dtype=torch.long, device=batch["context"].device)
        batch_beam = batch_beam.fill_(self.tokenizer.SOS_IDX)
        batch_beam_prob = torch.ones(N, 1)
        eos_detected = torch.zeros(N, self.beam_width, dtype=torch.bool)

        # Loop until EOS token detected for all beams in batch
        while (not torch.all(eos_detected)) and (seq_len < self.max_predict_seq_len):

            sequences = torch.empty(N, self.beam_width ** 2, seq_len + 1, dtype=torch.long)
            prob_sequences = torch.zeros(N, self.beam_width ** 2)

            # Determine all possible output sequences and probabilities for current beam 
            for i in range(self.beam_width if seq_len > 1 else 1):
                batch["target"] = batch_beam[:, i, :]
                batch["target_dialogue_state"] = ds.expand(batch["target"].size())
                logits, _ = self.forward(batch)
                conditional_p, top_responses = torch.topk(
                    F.softmax(logits[:, -1, :], dim=-1), self.beam_width, dim=-1)
                conditional_p = conditional_p.masked_fill(eos_detected[:, i:i+1], 1)
                for j in range(self.beam_width if seq_len > 1 else 1):
                    prob_sequences[:, i * self.beam_width + j] = (
                        batch_beam_prob[:, i] * conditional_p[:, j])
                    sequences[:, i * self.beam_width + j, :] = torch.cat(
                            (batch_beam[:, i, :], top_responses[:, j:j+1]), dim=-1)

            # Choose {self.beam_width} number of sequences with highest probabilities
            batch_beam_prob, top_indices = torch.topk(
                prob_sequences, self.beam_width, dim=-1)
            top_indices = top_indices.unsqueeze(2).expand(N, self.beam_width, seq_len + 1)
            batch_beam = torch.gather(sequences, 1, top_indices)

            # Check which beams in batch have reached the EOS token
            for i in range(N):
                for j in range(self.beam_width):
                    if int(batch_beam[i, j, -1]) == self.tokenizer.EOS_IDX:
                        eos_detected[i, j] = True
            
            # Increment target sequence length
            seq_len += 1

        # Return most probable sequence for each beam in batch
        return batch_beam[:, 0, :]

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