# ------------------------- IMPORT MODULES -----------------------------------

# System Modules

import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput

# User-Defined Modules
from base_classes import EncoderDecoderModel, DecoderModel

# ------------------------- IMPLEMENTATION ------------------------------------


class BlenderBot(EncoderDecoderModel):
    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        self.model.resize_token_embeddings(vocab_size)
        self.hidden_size = self.model.config.hidden_size

    def word_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor
    ) -> Seq2SeqLMOutput:
        
        out = self.model(
            input_ids=contexts,
            attention_mask=context_mask,
            decoder_input_ids=targets,
            decoder_attention_mask=target_mask,
            output_hidden_states=True
        )
        return out

    def generate(
            self, 
            contexts: torch.LongTensor, 
            context_mask: torch.BoolTensor, 
            **kwargs
        ) -> torch.LongTensor:
        return self.model.generate(
            input_ids=contexts,
            attention_mask=context_mask,
            **kwargs
        )


class GODEL(EncoderDecoderModel):
    def __init__(self, vocab_size: int, version: str) -> None:
        super().__init__(vocab_size)
        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/GODEL-v1_1-{version}-seq2seq")
        self.model.resize_token_embeddings(vocab_size)
        self.model.config.dropout_rate = 0.6
        self.hidden_size = self.model.config.hidden_size

    def word_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor
    ) -> Seq2SeqLMOutput:

        # Unmask pad token used to start generation
        target_mask[:, 0] = 1
        
        out = self.model(
            input_ids=contexts,
            attention_mask=context_mask,
            decoder_input_ids=targets,
            decoder_attention_mask=target_mask,
            output_hidden_states=True
        )
        return out

    def generate(
            self, 
            contexts: torch.LongTensor, 
            context_mask: torch.BoolTensor, 
            **kwargs
        ) -> torch.LongTensor:
        return self.model.generate(
            input_ids=contexts,
            attention_mask=context_mask,
            **kwargs
        )


class DialoGPT(DecoderModel):
    def __init__(self, vocab_size: int, version: str) -> None:
        super().__init__(vocab_size)
        if version not in ["small", "medium", "large"]:
            raise ValueError("Model version must be 'small', 'medium' or 'large'!")
        self.model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{version}")
        self.model.resize_token_embeddings(vocab_size)
        self.model.config.resid_pdrop = self.model.config.attn_pdrop = self.model.config.embd_pdrop = 0.6
        self.hidden_size = self.model.config.hidden_size

    def word_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self, 
        dialogues: torch.LongTensor,
        dialogue_mask: torch.BoolTensor,
    ) -> CausalLMOutput:
        out = self.model(
            input_ids=dialogues,
            attention_mask=dialogue_mask,
            output_hidden_states=True,
        )
        return out

    def generate(
            self, 
            contexts: torch.LongTensor, 
            context_mask: torch.BoolTensor, 
            **kwargs
        ) -> torch.LongTensor:
        len = contexts.size(dim=1)
        return self.model.generate(
            input_ids=contexts,
            attention_mask=context_mask,
            **kwargs
        )[:, len:]
    
# -----------------------------------------------------------------------------
