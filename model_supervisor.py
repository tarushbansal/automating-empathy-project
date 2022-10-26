# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from nltk.translate.bleu_score import corpus_bleu

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
        pred_beam_width: int = 1,
        max_pred_seq_len: int = 100,
        bleu_n_grams: int = 4
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "model"])

        self.model = model
        self.tokenizer = tokenizer
        self.initial_lr = initial_lr
        self.pred_beam_width = pred_beam_width
        self.max_pred_seq_len = max_pred_seq_len
        self.bleu_n_grams = bleu_n_grams

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
        _, len = batch["target"].size()
        logits, target = self.forward(batch)
        loss = F.cross_entropy(
            logits[:, -len:-1, :].permute(0, 2, 1), 
            batch["target"][:, 1:],
            ignore_index=self.tokenizer.PAD_IDX
        )
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.logger.experiment.add_scalars(
            'loss', {stage: loss}, self.global_step) 
        return loss

    def training_step(self, batch: Dict, _) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss        

    def validation_step(self, batch: Dict, _) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss
    
    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.logger.experiment.add_scalars(
            'loss', {'avg_val': avg_val_loss}, self.global_step) 

    def on_test_epoch_start(self) -> None:
        self.targets, self.responses = [], []
        self.sum_log_prob = 0

    def test_step(self, batch: Dict, _) -> None:
        self.targets.extend(batch["target"].tolist())
        response, prob = self.beam_search(batch)
        self.responses.extend(response)
        self.sum_log_prob += sum(prob)
    
    def on_test_epoch_end(self) -> None:
        targets = [[self.tokenizer.decode_to_text(target).split(" ")]
                    for target in self.targets]
        responses = []

        # Compute all evaluation metrics
        n_unigrams, n_bigrams = 0, 0
        unique_unigrams, unique_bigrams = set(), set()

        for response in self.responses:
            decoded = self.tokenizer.decode_to_text(response).split(" ")
            unique_unigrams.update([tuple(decoded[i:i+1]) for i in range(len(decoded))])
            unique_bigrams.update([tuple(decoded[i:i+2]) for i in range(len(decoded) - 1)])
            n_unigrams += len(decoded)
            n_bigrams += len(decoded) - 1
            responses.append(decoded)

        dist1 = len(unique_unigrams) / n_unigrams
        dist2 = len(unique_bigrams) / n_bigrams
        bleu = corpus_bleu(targets, responses, weights=[1/self.bleu_n_grams]*self.bleu_n_grams)
        total_token_count = sum([len(seq) for seq in self.responses])
        ppl = 1 / math.exp(self.sum_log_prob / total_token_count)

        self.test_metrics = {
            "bleu": bleu,
            "dist-1": dist1,
            "dist-2": dist2,
            "ppl": ppl
        }

        self.log_dict(self.test_metrics)

    def beam_search(self, batch: Dict) -> Tuple[List[List[int]]]:
        ## BEAM SEARCH: Objective is to determine the most probable output sequence from the model
        
        # Set model in evaluation mode (Important to turn off dropout layers!!)
        self.model.eval()

        N, seq_len = batch["context"].size(dim=0), 1
        device=batch["context"].device
        ds = batch["target_dialogue_state"][:, 0:1]
        batch_beam = torch.empty(N, 1, 1, dtype=torch.long, device=device)
        batch_beam = batch_beam.fill_(self.tokenizer.SOS_IDX)
        batch_beam_prob = torch.zeros(N, 1, device=device)
        eos_detected = torch.zeros(N, self.pred_beam_width, dtype=torch.bool, device=device)

        # Loop until EOS token detected for all beams in batch
        while (not torch.all(eos_detected)) and (seq_len < self.max_pred_seq_len):
            sequences = torch.empty(
                N, self.pred_beam_width ** 2, seq_len + 1, dtype=torch.long, device=device)
            prob_sequences = torch.empty(
                N, self.pred_beam_width ** 2, device=device).fill_(float("-inf"))

            # Determine all possible output sequences and probabilities for current beam 
            for i in range(self.pred_beam_width if seq_len > 1 else 1):
                batch["target"] = batch_beam[:, i, :]
                batch["target_dialogue_state"] = ds.expand(batch["target"].size())
                logits, _ = self.forward(batch)
                conditional_p, top_responses = torch.topk(
                    F.log_softmax(logits[:, -1, :], dim=-1), self.pred_beam_width, dim=-1)
                conditional_p = conditional_p.masked_fill(eos_detected[:, i:i+1], 0)
                for j in range(self.pred_beam_width if seq_len > 1 else 1):
                    prob_sequences[:, i * self.pred_beam_width + j] = (
                        batch_beam_prob[:, i] + conditional_p[:, j])
                    sequences[:, i * self.pred_beam_width + j, :] = torch.cat(
                            (batch_beam[:, i, :], top_responses[:, j:j+1]), dim=-1)

            # Choose {self.pred_beam_width} number of sequences with highest probabilities
            batch_beam_prob, top_indices = torch.topk(
                prob_sequences, self.pred_beam_width, dim=-1)
            top_indices = top_indices.unsqueeze(2).expand(
                N, self.pred_beam_width, seq_len + 1)
            batch_beam = torch.gather(sequences, 1, top_indices)

            # Check which beams in batch have reached the EOS token
            for i in range(N):
                for j in range(self.pred_beam_width):
                    if int(batch_beam[i, j, -1]) == self.tokenizer.EOS_IDX:
                        eos_detected[i, j] = True
            
            # Increment target sequence length
            seq_len += 1

        # Return most probable sequence and its log probability for each beam in batch
        responses = []
        for response in batch_beam[:, 0, :].tolist():
            for i in range(len(response)):
                if response[i] == self.tokenizer.EOS_IDX:
                    i += 1
                    break
            responses.append(response[:i])

        return responses, batch_beam_prob[:, 0].tolist()

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