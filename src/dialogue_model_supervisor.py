# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import copy
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
from utils.generation_utils import generate
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
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Tuple[Union[Seq2SeqLMOutput, CausalLMOutput, torch.Tensor]]:
        input_kwargs = {"past_key_values": past_key_values, "use_cache": use_cache}
        if self.model.requires_emotion_label:
            input_kwargs["emotion_label"] = batch.emotions
        if self.model.requires_concept_net_data:
            input_kwargs["concept_net_data"] = batch.concept_net_data
        if self.model.has_encoder:
            input_kwargs["encoder_outputs"] = encoder_outputs
            input_kwargs["contexts"] = batch.contexts
            input_kwargs["targets"] = batch.targets
            output = self.model(**input_kwargs)
            target_logits = output.logits
        else:
            N = batch.contexts.size(dim=0)
            target_len = batch.targets.size(dim=1)
            insert_idx = torch.sum(batch.contexts != self.tokenizer.PAD_IDX, dim=1)
            unstacked_sequences = tuple(torch.cat((
                    batch.contexts[i, :insert_idx[i]], 
                    batch.targets[i], 
                    batch.contexts[i, insert_idx[i]:]), dim=-1) 
                for i in range(N))
            input_kwargs["dialogues"] = torch.stack(unstacked_sequences)
            output = self.model(**input_kwargs)
            target_logits = torch.stack(
                tuple(output.logits[i, insert_idx[i]:insert_idx[i]+target_len] 
                      for i in range(N)))
        lm_loss = F.cross_entropy(
            target_logits[:, :-1, :].permute(0, 2, 1),
            batch.targets[:, 1:],
            ignore_index=self.tokenizer.PAD_IDX
        )
        
        return output, target_logits, lm_loss

    def forward_and_log_metrics(
        self, 
        batch: ModelBatch,
        stage: str
    ) -> float:
        N = batch.contexts.size(dim=0)
        lm_loss = self.forward(batch)[-1]
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
        self.contexts.extend([self.tokenizer.decode_to_text(context)
                              for context in batch.contexts.tolist()])
        self.targets.extend([self.tokenizer.decode_to_text(target)
                             for target in batch.targets.tolist()])
        indices = torch.sum(batch.targets != self.tokenizer.PAD_IDX, dim=1)
        self.enc_targets.extend([batch.targets[i, :indices[i]].tolist()
                                 for i in range(batch.targets.size(dim=0))])

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
        if batch.concept_net_data is not None:
            self.concepts.extend([self.tokenizer.decode_to_text(concepts)
                                  for concepts in batch.concept_net_data.tolist()])
        
        lm_loss = self.forward(batch)[-1]
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
            self.model.word_embeddings,
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
        batch: ModelBatch
    ) -> torch.LongTensor:
        self.model.eval()
        return generate(
            forward_fn=self.forward, 
            batch=copy.deepcopy(batch), 
            model_has_encoder=self.model.has_encoder,
            start_token=self.tokenizer.SOS_IDX,
            stop_token=self.tokenizer.EOS_IDX,
            pad_token=self.tokenizer.PAD_IDX,
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