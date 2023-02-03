# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import math
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput

# User-defined Modules
from base_classes import DialogueModelBase, TokenizerBase
from utils.metric_utils import compute_test_metrics
from data_classes import ModelBatch, GenerationConfig

# ------------------------- IMPLEMENTATION -----------------------------------


class DialogueModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        model: DialogueModelBase,
        tokenizer: TokenizerBase,
        initial_lr: Optional[float] = None,
        metric_n_grams: Optional[int] = None,
        test_output_dir: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.initial_lr = initial_lr
        self.test_output_dir = test_output_dir
        self.generation_config = generation_config
        self.metric_n_grams = metric_n_grams

    def forward(
        self, 
        batch: ModelBatch,
    ) -> Tuple[Union[Seq2SeqLMOutput, CausalLMOutput, torch.Tensor]]:
        input_kwargs = {}
        if self.model.requires_emotion_label:
            input_kwargs["emotion_label"] = batch.emotions
        if self.model.requires_concept_net_data:
            input_kwargs["concept_net_data"] = batch.concept_net_data
        if self.model.has_encoder:
            input_kwargs["contexts"] = batch.contexts
            input_kwargs["context_mask"] = batch.context_mask
            input_kwargs["targets"] = batch.targets
            input_kwargs["target_mask"] = batch.target_mask
            output = self.model(**input_kwargs)
            logits = output.logits[:, :-1, :]
            labels = batch.targets[:, 1:]
        else:
            input_kwargs["dialogues"] = torch.cat((batch.contexts, batch.targets), dim=1)
            input_kwargs["dialogue_mask"] = torch.cat((batch.context_mask, batch.target_mask), dim=1)
            output = self.model(**input_kwargs)
            logits = output.logits[:, batch.contexts.size(dim=1)-1:-1]
            labels = batch.targets
        
        lm_loss = (
            F.cross_entropy(
                logits.permute(0, 2, 1),
                labels,
                ignore_index=self.tokenizer.PAD_IDX
            ) if labels.size(dim=1) > 0 else None
        )
        
        return output, lm_loss

    def forward_and_log_metrics(
        self, 
        batch: ModelBatch,
        stage: str
    ) -> float:
        N = batch.contexts.size(dim=0)
        _, lm_loss = self.forward(batch)
        emo_loss, emo_attn_loss = 0, 0
        if hasattr(self.model, "emo_logits"):
            emo_loss = F.cross_entropy(
                self.model.emo_logits,
                batch.emotions
            )
            self.log(
                f"{stage}_emo_loss", 
                emo_loss,
                on_epoch=True, 
                batch_size=N, 
                sync_dist=True
            )
        if hasattr(self.model, "emo_attn_loss"):
            emo_attn_loss = self.model.emo_attn_loss
            self.log(
                f"{stage}_emo_attn_loss", 
                emo_attn_loss,
                on_epoch=True, 
                batch_size=N, 
                sync_dist=True
            )
        
        loss = lm_loss + 1 * emo_loss + 0.1 * emo_attn_loss

        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=True, 
            on_epoch=True,
            batch_size=N, 
            sync_dist=True
        )

        if loss != lm_loss:
            self.log(
                f"{stage}_lm_loss", 
                loss,
                on_epoch=True,
                batch_size=N, 
                sync_dist=True
            )

        return loss

    def training_step(
        self, 
        batch: ModelBatch,
        _
    ) -> float:
        train_loss = self.forward_and_log_metrics(batch, "train")
        return train_loss

    def validation_step(
        self, 
        batch: ModelBatch,
        _
    ) -> float:
        val_loss = self.forward_and_log_metrics(batch, "val")
        return val_loss

    def on_test_start(self) -> None:
        (self.contexts, self.targets, self.emotions,
         self.predictions, self.enc_targets, self.enc_predictions,
         self.emo_predictions, self.concepts, self.lm_loss) = ([] for _ in range(9))

    def test_step(
        self, 
        batch: ModelBatch,
        _
    ) -> None:
        self.contexts.extend(batch.raw_contexts)
        self.enc_targets.extend([[token for token in target 
                                  if token != self.tokenizer.PAD_IDX] 
                                  for target in batch.targets.tolist()])
        self.targets.extend(self.tokenizer.decode(batch.targets))

        enc_predictions = self.generate(batch.contexts, batch.context_mask)
        self.enc_predictions.extend([[token for token in prediction 
                                      if token != self.tokenizer.PAD_IDX] 
                                      for prediction in enc_predictions.tolist()])
        self.predictions.extend(self.tokenizer.decode(enc_predictions))
    
        if hasattr(self.model, "emo_logits"):
            self.emo_predictions.extend(
                [self.tokenizer.rev_emo_map[emo_idx]
                    for emo_idx in torch.max(
                    torch.softmax(self.model.emo_logits, dim=-1), dim=-1)[1].tolist()])
        if batch.emotions is not None:
            self.emotions.extend([self.tokenizer.rev_emo_map[emo_idx]
                                  for emo_idx in batch.emotions.tolist()])
        if batch.concept_net_data is not None:
            self.concepts.extend([self.tokenizer.decode(concepts)
                                  for concepts in batch.concept_net_data.tolist()])
        
        _, lm_loss = self.forward(batch)
        self.lm_loss.append(lm_loss)

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
            self.model.word_embeddings(),
            self.metric_n_grams,
        )

        test_metrics["ppl"] = math.exp(sum(self.lm_loss) / len(self.lm_loss))

        if log_emo_accuracy:
            test_metrics["emo_accuracy"] = accurate_emo_labels / N

        self.log_dict(test_metrics)

        if self.test_output_dir is not None:
            with open(f"{self.test_output_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)

    @torch.no_grad()
    def generate(
        self, 
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor
    ) -> torch.LongTensor:
        return self.model.generate(
            contexts=contexts,
            context_mask=context_mask,
            bos_token_id=self.tokenizer.SOS_IDX,
            pad_token_id=self.tokenizer.PAD_IDX,
            eos_token_id=self.tokenizer.EOS_IDX,
            max_new_tokens=self.generation_config.max_new_tokens,
            num_beams=self.generation_config.beam_width,
            do_sample=self.generation_config.sample,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            length_penalty=self.generation_config.length_alpha
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