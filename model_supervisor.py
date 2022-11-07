# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

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
        initial_lr: float = 0.0001,
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
        if self.model.has_encoder:
            input_kwargs["source_seq"] = batch["context"]
            input_kwargs["target_seq"] = batch["target"]
            if self.tokenizer.supports_dialogue_states:
                input_kwargs["source_dialogue_state"] = batch["context_dialogue_state"]
                input_kwargs["target_dialogue_state"] = batch["target_dialogue_state"]
            if self.tokenizer.supports_knowledge_concepts:
                input_kwargs["concepts"] = batch["concepts"]
                input_kwargs["adjacency_mask"] = batch["adjacency_mask"]
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
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.logger.experiment.add_scalars('loss', {stage: loss}, self.global_step) 
        return loss

    def training_step(self, batch: Dict, _) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss        

    def validation_step(self, batch: Dict, _) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss
    
    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True, batch_size=self.batch_size)
        self.logger.experiment.add_scalars('loss', {'avg_val': avg_val_loss}, self.global_step)
        return avg_val_loss 

    def test_step(self, batch: Dict, _) -> Tuple[List]:
        contexts = [self.tokenizer.decode_to_text(context)
                    for context in batch["context"].tolist()]
        targets = [self.tokenizer.decode_to_text(enc) for enc in batch["target"].tolist()]
        enc_targets = [self.tokenizer.encode_text(target, "target")[0] for target in targets]
        enc_predictions, log_probs = self.beam_search(batch)
        predictions = [self.tokenizer.decode_to_text(enc) for enc in enc_predictions]
        log_probs = [log_probs[i] / len(enc_predictions[i]) for i in range(len(log_probs))]
        return contexts, targets, enc_targets, predictions, enc_predictions, log_probs

    def test_epoch_end(self, test_data: List[Tuple]) -> None:
        merged_data = [test_data]
        if self.trainer.is_global_zero:
            encoded_targets, targets, predictions, encoded_predictions, log_probs = [], [], [], [], []
            with open(f"{self.test_output_dir}/test_predictions.txt", "w") as f:
                for test_data in merged_data:
                    [batch_contexts, batch_targets, 
                    batch_enc_targets, batch_pred, 
                    batch_enc_pred, batch_log_probs] = zip(*test_data)
                    for i in range(len(batch_contexts)):
                        for j in range(len(batch_contexts[i])):
                            context, target, prediction = (
                                batch_contexts[i][j], 
                                batch_targets[i][j], 
                                batch_pred[i][j]
                            )
                            f.write(f"Context: {context}; Target: {target}; Predicted: {prediction}\n")
                            targets.append([target.split(" ")])
                            predictions.append(prediction.split(" "))

                    encoded_targets.extend([target for batch in batch_enc_targets for target in batch])
                    encoded_predictions.extend([prediction for batch in batch_enc_pred for prediction in batch])
                    log_probs.extend([log_prob for batch in batch_log_probs for log_prob in batch])

            test_metrics = compute_test_metrics(
                targets, 
                predictions,
                encoded_targets,
                encoded_predictions,
                self.pred_n_grams,
                log_probs,
                self.model.word_embeddings,
            )
            self.log_dict(test_metrics)
            
            if self.test_output_dir is not None:
                with open(f"{self.test_output_dir}/test_metrics.json", "w") as f:
                    json.dump(test_metrics, f)

    def beam_search(self, batch: Dict) -> Tuple[List[List[int]]]:
        ## BEAM SEARCH: Objective is to determine the most probable output sequence from the model
        batch = batch.copy()
        batch.pop("target", None)
        if self.tokenizer.supports_dialogue_states:
            ds = batch["target_dialogue_state"][:, 0:1]
        batch["target_dialogue_state"] = None

        # Set model in evaluation mode (Important to turn off dropout layers!!)
        self.model.eval()

        N, seq_len = batch["context"].size(dim=0), 0
        device = batch["context"].device

        batch_beam_prob = torch.zeros(N, 1, device=device)
        batch_beam = torch.empty(N, 1, 0, dtype=torch.long, device=device)
        if self.tokenizer.SOS_IDX is not None:
            seq_len += 1
            sos_tensor = torch.empty(N, 1, 1, dtype=torch.long, device=device).fill_(self.tokenizer.SOS_IDX)
            batch_beam = torch.cat((batch_beam, sos_tensor), dim=-1)
        eos_detected = torch.zeros(N, self.pred_beam_width, dtype=torch.bool, device=device)

        # Loop until EOS token detected for all beams in batch
        while (not torch.all(eos_detected)) and (seq_len < self.max_pred_seq_len):
            sequences = torch.empty(
                N, self.pred_beam_width ** 2, seq_len + 1, dtype=torch.long, device=device)
            prob_sequences = torch.empty(
                N, self.pred_beam_width ** 2, device=device).fill_(float("-inf"))

            # Determine all possible output sequences and probabilities for current beam 
            for i in range(1 if seq_len == 1 else self.pred_beam_width):
                batch["target"] = batch_beam[:, i, :]
                if self.tokenizer.supports_dialogue_states:
                    batch["target_dialogue_state"] = ds.expand(batch["target"].size())
                logits, _ = self.forward(batch)
                conditional_p, top_responses = torch.topk(
                    F.log_softmax(logits[:, -1, :], dim=-1), self.pred_beam_width, dim=-1)
                conditional_p = conditional_p.masked_fill(eos_detected[:, i:i+1], 0)
                for j in range(1 if seq_len == 1 else self.pred_beam_width):
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
        predictions = []
        for pred in batch_beam[:, 0, :].tolist():
            for i in range(len(pred)):
                if pred[i] == self.tokenizer.EOS_IDX:
                    i += 1
                    break
            predictions.append(pred[:i])
        log_prob = batch_beam_prob[:, 0].tolist()

        return predictions, log_prob

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

# -----------------------------------------------------------------------------