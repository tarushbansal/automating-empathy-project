# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import copy
import json
import math
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup

# User-defined Modules
from base_classes import DialogueModelBase, TokenizerBase
from data_classes import EncoderDecoderModelBatch, DecoderModelBatch, GenerationConfig
from metric_utils import compute_test_metrics
from generation import generate

# ------------------------- IMPLEMENTATION -----------------------------------


class ModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        model: DialogueModelBase,
        tokenizer: TokenizerBase,
        batch_size: int,
        initial_lr: Optional[float] = None,
        metric_n_grams: Optional[int] = None,
        test_output_dir: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:

        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.test_output_dir = test_output_dir
        self.generation_config = generation_config
        self.metric_n_grams = metric_n_grams

    def forward(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch]
    ) -> Tuple[torch.Tensor]:

        input_kwargs = {}
        if self.model.requires_emotion_label:
            input_kwargs["emotion_label"] = batch.emotions
        if self.model.requires_concept_net_data:
            input_kwargs["concept_net_data"] = batch.concept_net_data
        if self.model.has_encoder:
            input_kwargs["source_seq"] = batch.contexts
            input_kwargs["target_seq"] = batch.targets
            logits = self.model(**input_kwargs)
            target_seq = batch.targets[:, 1:]
        else:
            input_kwargs["input_seq"] = batch.dialogues
            logits = self.model(**input_kwargs)
            target_seq = batch.dialogues[:, 1:]
        
        return logits, target_seq

    def forward_and_log_metrics(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        stage: str
    ) -> float:

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

    def training_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def validation_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss

    def validation_epoch_end(self, val_losses: List[float]) -> float:
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True,
                 batch_size=self.batch_size, sync_dist=True)
        self.logger.experiment.add_scalars('loss', {'avg_val': avg_val_loss}, self.global_step)
        return avg_val_loss

    def on_test_start(self) -> None:
        self.sum_cross_entropy = 0
        self.num_tokens = 0
        (self.contexts, self.targets, self.emotions,
         self.predictions, self.enc_targets, self.enc_predictions,
         self.emo_predictions, self.concepts) = ([] for _ in range(8))

    def test_step(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch],
        _
    ) -> None:

        contexts = batch.contexts if self.model.has_encoder else batch.dialogues
        self.contexts.extend([self.tokenizer.decode_to_text(context)
                              for context in contexts.tolist()])
        targets = [self.tokenizer.decode_to_text(target)
                   for target in batch.targets.tolist()]
        self.targets.extend(targets)
        self.enc_targets.extend([self.tokenizer.encode_text(target)[0]
                                 for target in targets])

        enc_predictions = self.generate(batch)
        self.enc_predictions.extend(enc_predictions)
        self.predictions.extend([self.tokenizer.decode_to_text(enc) 
                                 for enc in enc_predictions])
    
        if hasattr(self.model, "emo_logits"):
            self.emo_predictions.extend(
                [self.tokenizer.rev_emo_map[emo_idx]
                    for emo_idx in torch.max(
                    torch.softmax(self.model.emo_logits, dim=-1), dim=-1)[1].tolist()])
        if batch.emotions is not None:
            self.emotions.extend([self.tokenizer.rev_emo_map[emo_idx]
                                for emo_idx in batch.emotions.tolist()])
        if self.model.has_encoder and batch.concept_net_data is not None:
            self.concepts.extend([self.tokenizer.decode_to_text(concepts)
                                  for concepts in batch.concept_net_data.tolist()])
        
        if not self.model.has_encoder:
            insert_idx = torch.sum(batch.dialogues != self.tokenizer.PAD_IDX, dim=1)
            unstacked_sequences = tuple(torch.cat((
                    batch.dialogues[i, :insert_idx[i]], 
                    batch.targets[i], 
                    batch.dialogues[i, insert_idx[i]:]), dim=-1) 
                for i in range(batch.dialogues.size(dim=0)))
            batch.dialogues = torch.stack(unstacked_sequences)

        logits, target_seq = self.forward(batch)
        self.num_tokens += torch.sum(target_seq != self.tokenizer.PAD_IDX)
        self.sum_cross_entropy += F.cross_entropy(
            logits[:, :-1, :].permute(0, 2, 1),
            target_seq,
            reduction="sum",
            ignore_index=self.tokenizer.PAD_IDX
        )

    def test_epoch_end(self, _) -> None:
        N = len(self.contexts)
        test_data = []
        accurate_emo_labels = 0
        log_emo_accuracy = len(self.emotions) != 0 and len(self.emo_predictions) != 0
        for i in range(N):
            entry = {
                "id": i,
                "context": self.contexts[i],
                "target": self.targets[i],
                "prediction": self.predictions[i]
            }
            if len(self.emotions) != 0:
                entry["emotion"] = self.emotions[i]
            if len(self.emo_predictions) != 0:
                entry["pred_emotion"] = self.emo_predictions[i]
            if len(self.concepts) != 0:
                entry["concepts"] = self.concepts[i]
            test_data.append(entry)
            if log_emo_accuracy and entry["emotion"] == entry["pred_emotion"]:
                accurate_emo_labels += 1

        with open(f"{self.test_output_dir}/test_data.json", "w") as f:
            json.dump(test_data, f)

        test_metrics = compute_test_metrics(
            self.targets,
            self.predictions,
            self.enc_targets,
            self.enc_predictions,
            self.model.word_embeddings,
            self.metric_n_grams,
        )

        test_metrics["ppl"] = math.exp(self.sum_cross_entropy / self.num_tokens)

        if log_emo_accuracy:
            test_metrics["emo_accuracy"] = accurate_emo_labels / N

        self.log_dict(test_metrics)

        if self.test_output_dir is not None:
            with open(f"{self.test_output_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)

    def generate(
        self, 
        batch: Union[EncoderDecoderModelBatch, DecoderModelBatch]
    ) -> torch.LongTensor:
        
        self.model.eval()
        return generate(
            forward_fn=lambda x: self.forward(x)[0], 
            batch=copy.deepcopy(batch), 
            start_token=self.tokenizer.SOS_IDX,
            stop_token=self.tokenizer.EOS_IDX,
            pad_token=self.tokenizer.PAD_IDX,
            vocab_size=self.tokenizer.vocab_size,
            model_has_encoder=self.model.has_encoder,
            generation_config=self.generation_config
        )


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