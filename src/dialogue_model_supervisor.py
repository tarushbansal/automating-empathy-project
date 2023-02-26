# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import math
from typing import Optional, Union, Tuple, Dict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput

# User-defined Modules
from base_classes import DialogueModelBase, TokenizerBase
from utils.metric_utils import compute_test_metrics
from data_classes import ModelConfig, ModelBatch, GenerationConfig

# ------------------------- IMPLEMENTATION -----------------------------------


class DialogueModelSupervisor(pl.LightningModule):
    def __init__(
        self,
        config: ModelConfig.__dict__,
        batch_size:  Optional[int] = None,
        initial_lr: Optional[float] = None,
        metric_n_grams: Optional[int] = None,
        test_output_dir: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        
        super().__init__()
        model_cls = getattr(__import__("dialogue_models"), config["model_cls"])
        tokenizer_cls = getattr(__import__("custom_tokenizers"), config["tokenizer_cls"])
        self.tokenizer: TokenizerBase = tokenizer_cls(**config["tokenizer_kwargs"])
        self.model: DialogueModelBase = model_cls(
            vocab_size=self.tokenizer.vocab_size,
            **config["model_kwargs"]
        )

        # Sanity checks
        if not issubclass(model_cls, DialogueModelBase):
            raise ValueError("Model must be derived from base class 'DialogueModelBase'!")
        
        if not issubclass(tokenizer_cls, TokenizerBase):
            raise ValueError(
                "Tokenizer must be derived from base class 'TokenizerBase'!")

        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.test_output_dir = test_output_dir
        self.generation_config = generation_config
        self.metric_n_grams = metric_n_grams
        self.default_log_config = {
            "on_step": True,
            "on_epoch": True,
            "sync_dist": True
        }
        
        self.save_hyperparameters("config", "batch_size", "initial_lr")

    def forward(
        self, 
        batch: ModelBatch,
    ) -> Tuple[Union[Seq2SeqLMOutput, CausalLMOutput, torch.Tensor]]:
        input_kwargs = {}
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
        _, loss = self.forward(batch)        

        self.log(
            f"{stage}_loss", 
            loss, 
            prog_bar=True,
            batch_size=N, 
            **self.default_log_config
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
        (self.inputs, self.raw_contexts, self.targets,
         self.outputs, self.enc_targets, self.enc_outputs, 
         self.concepts, self.lm_loss) = ([] for _ in range(8))

    def test_step(
        self, 
        batch: ModelBatch,
        _
    ) -> None:
        self.raw_contexts.extend(batch.raw_contexts)
        self.inputs.extend(self.tokenizer.decode(
            [[token for token in context 
              if token != self.tokenizer.PAD_IDX] 
              for context in batch.contexts.tolist()
            ], skip_special_tokens=False)
        )
        self.enc_targets.extend([[token for token in target 
                                  if token != self.tokenizer.PAD_IDX] 
                                  for target in batch.targets.tolist()])
        self.targets.extend(self.tokenizer.decode(batch.targets))

        enc_outputs = self.generate(batch.contexts, batch.context_mask)
        self.enc_outputs.extend([[token for token in output 
                                      if token != self.tokenizer.PAD_IDX] 
                                      for output in enc_outputs.tolist()])
        self.outputs.extend(self.tokenizer.decode(enc_outputs))
    
        if batch.concept_net_data is not None:
            self.concepts.extend([self.tokenizer.decode(concepts)
                                  for concepts in batch.concept_net_data.tolist()])
        
        _, lm_loss = self.forward(batch)
        self.lm_loss.append(lm_loss)

    def test_epoch_end(self, _) -> None:
        N = len(self.outputs)
        test_data = []
        for i in range(N):
            entry = {
                "id": i,
                "input": self.inputs[i],
                "context": self.raw_contexts[i],
                "target": self.targets[i],
                "output": self.outputs[i]
            }
            if len(self.concepts) != 0:
                entry["concepts"] = self.concepts[i]
            test_data.append(entry)

        with open(f"{self.test_output_dir}/test_data.json", "w") as f:
            json.dump(test_data, f)

        test_metrics = compute_test_metrics(
            self.targets,
            self.outputs,
            self.enc_targets,
            self.enc_outputs,
            self.model.word_embeddings(),
            self.metric_n_grams,
        )

        test_metrics["ppl"] = math.exp(sum(self.lm_loss) / len(self.lm_loss))

        self.log_dict(test_metrics)

        if self.test_output_dir is not None:
            with open(f"{self.test_output_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)

    @torch.no_grad()
    def generate(
        self, 
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        default_config: bool = False
    ) -> torch.LongTensor:
        if default_config:
            config = {}
        else:
            config = {
                "num_beams":self.generation_config.beam_width,
                "do_sample":self.generation_config.sample,
                "temperature":self.generation_config.temperature,
                "top_p":self.generation_config.top_p,
                "top_k":self.generation_config.top_k,
                "length_penalty":self.generation_config.length_alpha
            }
            config = {k:v for k,v in config.items() if v is not None}
        return self.model.generate(
            contexts=contexts,
            context_mask=context_mask,
            bos_token_id=self.tokenizer.SOS_IDX,
            pad_token_id=self.tokenizer.PAD_IDX,
            eos_token_id=self.tokenizer.EOS_IDX,
            max_new_tokens=self.generation_config.max_new_tokens,
            **config
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