# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import math
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from torchmetrics.functional import pairwise_cosine_similarity
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist

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
        self.test_output_dir = test_output_dir
        self.pred_beam_width = pred_beam_width
        self.max_pred_seq_len = max_pred_seq_len
        self.pred_n_grams = pred_n_grams

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
            target = input_seq
        elif issubclass(type(self.model), EncoderDecoderModel):
            logits = self.model(
                source_seq=batch["context"],
                target_seq=batch["target"],
                source_dialogue_state=batch["context_dialogue_state"],
                target_dialogue_state=batch["target_dialogue_state"],
                emotion_label=batch["emotion"]
            )
            target = batch["target"] 
        return logits, target
    
    def forward_and_log_metrics(self, batch: Dict, stage: str):
        logits, target = self.forward(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].permute(0, 2, 1), 
            target[:, 1:],
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

    def test_step(self, batch: Dict, _) -> Tuple[List]:
        targets = [self.tokenizer.decode_to_text(target).split(" ")
                   for target in batch["target"].tolist()]
        predictions, log_probs = self.beam_search(batch)
        log_probs = [log_probs[i] / len(predictions[i]) for i in range(len(log_probs))]
        predictions = [self.tokenizer.decode_to_text(prediction).split(" ")
                       for prediction in predictions]
        return targets, predictions, log_probs

    @rank_zero_only
    def test_epoch_end(self, test_data: List[Tuple]) -> None:
        batch_targets, batch_predictions, batch_log_probs = zip(*test_data)
        targets = [[target] for batch in batch_targets for target in batch]
        predictions = [pred for batch in batch_predictions for pred in batch]
        log_probs = [prob for batch in batch_log_probs for prob in batch]
        test_metrics = self.compute_test_metrics(targets, predictions, log_probs)
        self.log_dict(test_metrics)
        
        if self.test_output_dir is not None:
            with open(f"{self.test_output_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)
    
    def compute_test_metrics(
        self, 
        targets: List[List[List[str]]], 
        predictions: List[List[str]], 
        log_probs: List[float]
    ) -> Dict[str, float]:

        n_unigrams, n_bigrams = 0, 0
        unique_unigrams, unique_bigrams = set(), set()
        avg_bow, extrema_bow, greedy_bow = [], [], []

        for i in range(len(targets)):
            # DIST - unigrams and bigrams
            unique_unigrams.update([tuple(predictions[i][j:j+1]) for j in range(len(predictions[i]))])
            unique_bigrams.update([tuple(predictions[i][j:j+2]) for j in range(len(predictions[i]) - 1)])
            n_unigrams += len(predictions[i])
            n_bigrams += len(predictions[i]) - 1
            
            # Encode and embed both targets and predictions
            encoded_targets = " ".join(targets[i][0])
            encoded_predictions = " ".join(predictions[i])
            device = self.model.word_embeddings.weight.device
            target_embeddings = self.model.word_embeddings(
                torch.LongTensor(self.tokenizer.encode_text([encoded_targets])).to(device))
            pred_embeddings = self.model.word_embeddings(
                torch.LongTensor(self.tokenizer.encode_text([encoded_predictions])).to(device))

            # Average BOW
            avg_target_embed = target_embeddings.mean(dim=1)
            avg_pred_embed = pred_embeddings.mean(dim=1)
            avg_bow.append(float(F.cosine_similarity(avg_target_embed, avg_pred_embed)))

            # Extrema BOW
            max_target, _ = torch.max(target_embeddings, dim=1)
            min_target, _ = torch.min(target_embeddings, dim=1)
            mask_target = (torch.abs(max_target) >= torch.abs(min_target)) 
            extrema_target_embed = torch.where(mask_target, max_target, min_target)
            max_pred, _ = torch.max(pred_embeddings, dim=1)
            min_pred, _ = torch.min(pred_embeddings, dim=1)
            mask_pred = (torch.abs(max_pred) >= torch.abs(min_pred)) 
            extrema_pred_embed = torch.where(mask_pred, max_pred, min_pred)
            extrema_bow.append(float(F.cosine_similarity(extrema_target_embed, extrema_pred_embed)))

            # Greedy BOW
            sim = pairwise_cosine_similarity(target_embeddings.squeeze(), pred_embeddings.squeeze())
            greedy_bow.append(float(sim.max(dim=0)[0].mean() + sim.max(dim=1)[0].mean() / 2))

        dist1 = len(unique_unigrams) / n_unigrams
        dist2 = len(unique_bigrams) / n_bigrams
        bleu = corpus_bleu(targets, predictions, weights=[1/self.pred_n_grams]*self.pred_n_grams)
        nist = corpus_nist(targets, predictions, n=self.pred_n_grams)
        ppl = 1 / math.exp(sum(log_probs) / len(log_probs))
        avg_bow = sum(avg_bow) / len(avg_bow)
        extrema_bow = sum(extrema_bow) / len(extrema_bow)
        greedy_bow = sum(greedy_bow) / len(greedy_bow)

        test_metrics = {
            "bleu": bleu,
            "nist": nist,
            "dist-1": dist1,
            "dist-2": dist2,
            "ppl": ppl,
            "avg_bow": avg_bow,
            "extrema_bow": extrema_bow,
            "greedy_bow": greedy_bow
        }

        return test_metrics

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
            optimizer, step_size=10, gamma=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

# -----------------------------------------------------------------------------