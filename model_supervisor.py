# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup

# User-defined Modules
from base_classes import DialogueModelBase, TokenizerBase
from metric_utils import compute_test_metrics

# ------------------------- IMPLEMENTATION -----------------------------------


class ModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        model: DialogueModelBase,
        tokenizer: TokenizerBase,
        batch_size: int,
        initial_lr: float,
        test_output_dir: str = None,
        pred_beam_width: int = 1,
        max_pred_seq_len: int = 100,
        pred_n_grams: int = 4,
    ) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "model", "test_output_dir"])

        self.model = model
        self.tokenizer = tokenizer
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.test_output_dir = test_output_dir
        self.pred_beam_width = pred_beam_width
        self.max_pred_seq_len = max_pred_seq_len
        self.pred_n_grams = pred_n_grams

    def forward(self, batch: Dict) -> Tuple[torch.Tensor]:
        input_kwargs = {}
        if self.model.requires_emotion_label:
            input_kwargs["emotion_label"] = batch["emotion"]
        if self.tokenizer.supports_external_knowledge:
            input_kwargs["external_knowledge"] = batch["external_knowledge"]
        if self.model.has_encoder:
            input_kwargs["source_seq"] = batch["context"]
            input_kwargs["target_seq"] = batch["target"]
            if self.tokenizer.supports_dialogue_states:
                input_kwargs["source_dialogue_state"] = batch["context_dialogue_state"]
                input_kwargs["target_dialogue_state"] = batch["target_dialogue_state"]
            logits = self.model(**input_kwargs)
            target_seq = batch["target"][:, 1:]
        else:
            input_seq = torch.cat((batch["context"], batch["target"]), dim=1)
            input_kwargs["input_seq"] = input_seq
            if self.tokenizer.supports_dialogue_states:
                input_kwargs["input_dialogue_state"] = torch.cat(
                    (batch["context_dialogue_state"],
                     batch["target_dialogue_state"]), dim=1)
            logits = self.model(**input_kwargs)
            target_seq = input_seq[:, 1:]
        return logits, target_seq

    def forward_and_log_metrics(self, batch: Dict, stage: str):
        logits, target_seq = self.forward(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].permute(0, 2, 1),
            target_seq,
            ignore_index=self.tokenizer.PAD_IDX
        )
        emo_loss, emo_attn_loss = 0, 0
        if hasattr(self.model, "emo_logits"):
            emo_loss = F.cross_entropy(
                self.model.emo_logits,
                batch["emotion"]
            )
            self.log(f"{stage}_emo_loss", emo_loss, prog_bar=True, batch_size=self.batch_size)
            self.logger.experiment.add_scalars('emo_loss', {stage: emo_loss}, self.global_step)
        if hasattr(self.model, "emo_attn_loss"):
            emo_attn_loss = self.model.emo_attn_loss
            self.log(f"{stage}_emo_attn_loss", emo_attn_loss,
                     prog_bar=True, batch_size=self.batch_size)
            self.logger.experiment.add_scalars(
                'emo_attn_loss', {stage: emo_attn_loss}, self.global_step)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {stage: loss}, self.global_step)
        return loss + 1 * emo_loss + 0.1 * emo_attn_loss

    def training_step(self, batch: Dict, _) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def validation_step(self, batch: Dict, _) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss

    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {'avg_val': avg_val_loss}, self.global_step)
        return avg_val_loss

    def on_test_start(self) -> None:
        (self.contexts, self.targets, self.emotions,
         self.predictions, self.enc_targets, self.enc_predictions,
         self.emo_predictions, self.concepts, self.cross_entropy) = ([] for _ in range(9))

    def test_step(self, batch: Dict, _) -> Tuple[List]:
        targets = [self.tokenizer.decode_to_text(enc) for enc in batch["target"].tolist()]
        self.targets.extend(targets)
        self.enc_targets.extend([self.tokenizer.encode_text(target, "target")[0]
                                for target in targets])

        logits, target_seq = self.forward(batch)
        self.cross_entropy.append(F.cross_entropy(
            logits[:, :-1, :].permute(0, 2, 1),
            target_seq,
            ignore_index=self.tokenizer.PAD_IDX
        ))

        enc_prediction = self.beam_search(batch)
        self.enc_predictions.extend([enc_prediction])
        self.predictions.extend([self.tokenizer.decode_to_text(enc_prediction)])

        self.contexts.extend([self.tokenizer.decode_to_text(context)
                             for context in batch["context"].tolist()])
        self.emotions.extend([self.tokenizer.rev_emo_map[emo_idx]
                             for emo_idx in batch["emotion"].tolist()])

        if self.tokenizer.supports_external_knowledge:
            if hasattr(self.model, "emo_logits"):
                self.emo_predictions.extend([self.tokenizer.rev_emo_map[emo_idx]
                                             for emo_idx in torch.max(torch.softmax(self.model.emo_logits, dim=-1), dim=-1)[1].tolist()])
            self.concepts.extend([self.tokenizer.decode_to_text(concepts)
                                  for concepts in batch["external_knowledge"]["concepts"].tolist()])

    def test_epoch_end(self, _) -> None:
        N = len(self.contexts)
        accurate_emo_labels = 0
        test_data = []
        for i in range(N):
            entry = {
                "id": i,
                "context": self.contexts[i],
                "target": self.targets[i],
                "prediction": self.predictions[i],
                "emotion": self.emotions[i],
                "pred_emotion": None if len(self.emo_predictions) == 0 else self.emo_predictions[i],
                "concepts": None if len(self.concepts) == 0 else self.concepts[i]
            }
            test_data.append(entry)
            if len(self.emo_predictions) != 0 and entry["emotion"] == entry["pred_emotion"]:
                accurate_emo_labels += 1

        with open(f"{self.test_output_dir}/test_data.json", "w") as f:
            json.dump(test_data, f)

        test_metrics = compute_test_metrics(
            self.targets,
            self.predictions,
            self.enc_targets,
            self.enc_predictions,
            self.model.word_embeddings,
            self.pred_n_grams,
        )

        test_metrics["ppl"] = math.exp(sum(self.cross_entropy) / len(self.cross_entropy))

        if len(self.emo_predictions) != 0:
            test_metrics["emo_accuracy"] = accurate_emo_labels / N

        self.log_dict(test_metrics)

        if self.test_output_dir is not None:
            with open(f"{self.test_output_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)

    def beam_search(
        self,
        batch: Dict[str, Union[Dict, torch.LongTensor]]
    ) -> Tuple[List[List[int]]]:

        # BEAM SEARCH: Objective is to determine the most probable output sequence from the model
        batch = batch.copy()
        if self.tokenizer.supports_dialogue_states:
            ds = batch["target_dialogue_state"][:, 0:1]

        # Set model in evaluation mode (Important to turn off dropout layers!!)
        self.model.eval()

        N, seq_len = batch["context"].size(dim=0), 0
        device = batch["context"].device
        if N != 1:
            raise ValueError("Can only handle a batch size of 1 for beam search generation!")

        beam = torch.empty(1, 0, dtype=torch.long, device=device)
        if self.tokenizer.SOS_IDX is not None:
            seq_len += 1
            sos_tensor = torch.empty(
                1, 1, dtype=torch.long, device=device).fill_(self.tokenizer.SOS_IDX)
            beam = torch.cat((beam, sos_tensor), dim=-1)
        beam_prob = torch.zeros(1, device=device)
        eos_detected = [False]

        # Loop until EOS token detected for all beams in batch
        while (not all(eos_detected)) and (seq_len < self.max_pred_seq_len):
            sequences = torch.empty(
                self.pred_beam_width ** 2, seq_len + 1, dtype=torch.long, device=device)
            prob_sequences = torch.empty(
                self.pred_beam_width ** 2, device=device).fill_(float("-inf"))

            # Determine all possible output sequences and probabilities for current beam
            for i in range(beam.size(dim=0)):
                batch["target"] = beam[i, :].unsqueeze(0)
                if self.tokenizer.supports_dialogue_states:
                    batch["target_dialogue_state"] = ds.expand(batch["target"].size())
                logits, _ = self.forward(batch)
                conditional_p, top_responses = torch.topk(
                    F.log_softmax(logits[0, -1, :], dim=-1),
                    self.pred_beam_width,
                    dim=-1
                )
                if eos_detected[i]:
                    prob_sequences[i * self.pred_beam_width] = beam_prob[i]
                    sequences[i * self.pred_beam_width] = torch.cat(
                        (beam[i, :], torch.LongTensor([self.tokenizer.PAD_IDX]).to(device)), dim=-1)
                    continue
                for j in range(self.pred_beam_width):
                    prob_sequences[i * self.pred_beam_width + j] = (
                        beam_prob[i] + conditional_p[j])
                    sequences[i * self.pred_beam_width + j] = torch.cat(
                        (beam[i], top_responses[j:j+1]), dim=-1)

            # Choose {self.pred_beam_width} number of sequences with highest probabilities
            beam_prob, top_indices = torch.topk(
                prob_sequences, self.pred_beam_width, dim=0)
            top_indices = top_indices.unsqueeze(1).expand(
                self.pred_beam_width, seq_len + 1)
            beam = torch.gather(sequences, 0, top_indices)

            # Increment target sequence length
            seq_len += 1

            # Break out of loop if EOS detected for all sequences in all beams
            eos_detected = [self.tokenizer.EOS_IDX in seq for seq in beam]

        # Return most probable sequence and its log probability for each beam in batch
        return beam[0].tolist()

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
