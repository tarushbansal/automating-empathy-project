# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Tuple

import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# User-Defined Modules
from base_classes import EncoderDecoderModel, DecoderModel, TokenizerBase
from data_classes import ConceptNetBatchData

# ------------------------- IMPLEMENTATION ------------------------------------


class GODEL(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/GODEL-v1_1-{version}-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.6

    @staticmethod
    def tokenizer_cls():
        return "GODELTokenizer"

    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        source_seq: torch.LongTensor,
        target_seq: torch.LongTensor
    ) -> torch.Tensor:

        source_seq, source_mask = self.create_padding_mask(source_seq)
        target_seq, target_mask = self.create_padding_mask(target_seq)

        out = self.model(
            input_ids=source_seq,
            attention_mask=source_mask,
            decoder_input_ids=target_seq,
            decoder_attention_mask=target_mask
        )
        return out.logits


class KnowledgeBridgedGODEL(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/GODEL-v1_1-{version}-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.8
        self.graph_embeddings = nn.Embedding(2, self.model.config.hidden_size)
        self.emo_linear = nn.Linear(self.model.config.hidden_size, self.tokenizer.num_emo_labels)
        self.attn_loss = nn.MSELoss()

    @staticmethod
    def tokenizer_cls():
        return "GODELTokenizer"

    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()
    
    @property
    def requires_concept_net_data(self) -> bool:
        return True

    def knowledge_enriched_context(
        self,
        context: torch.LongTensor,
        concepts: torch.LongTensor,
    ) -> Tuple[torch.Tensor]:

        context, context_mask = self.create_padding_mask(context)
        concepts, concept_mask = self.create_padding_mask(concepts)
        pad_mask = torch.cat((context_mask, concept_mask), dim=1)

        model_embeddings = self.model.get_input_embeddings()
        context_graph_embeds = self.graph_embeddings(
            torch.zeros(context.size(), dtype=torch.long, device=context.device))
        context_embeds = model_embeddings(context) + context_graph_embeds
        concept_graph_embeds = self.graph_embeddings(
            torch.ones(concepts.size(), dtype=torch.long, device=concepts.device))
        concept_embeds = model_embeddings(concepts) + concept_graph_embeds
        embeddings = torch.cat((context_embeds, concept_embeds), dim=1)

        return embeddings, pad_mask

    def forward(
        self,
        source_seq: torch.LongTensor,
        target_seq: torch.LongTensor,
        concept_net_data: ConceptNetBatchData
    ) -> torch.Tensor:

        input_embeds, pad_mask = self.knowledge_enriched_context(
            source_seq, concept_net_data.concepts)
        # attention_mask = torch.minimum(pad_mask.unsqueeze(1), concept_net_data.adjacency_mask)
        target_seq, target_mask = self.create_padding_mask(target_seq)

        out = self.model(
            inputs_embeds=input_embeds,
            attention_mask=pad_mask,
            decoder_input_ids=target_seq,
            decoder_attention_mask=target_mask,
            output_attentions=True
        )

        emo_intensities = torch.cat(
            (concept_net_data.context_emo_intensity,
             concept_net_data.concept_emo_intensity), dim=1)
        sum_weights = torch.softmax(emo_intensities, dim=1).unsqueeze(2)
        c = torch.sum(sum_weights * out.encoder_last_hidden_state, dim=1)
        self.emo_logits = self.emo_linear(c)

        average_attn_weights = torch.stack(out.cross_attentions).mean((0, 2, 3))
        self.emo_attn_loss = self.attn_loss(emo_intensities, average_attn_weights)

        return out.logits


class DialoGPT(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["small", "medium", "large"]:
            raise ValueError("Model version must be 'small', 'medium' or 'large'!")
        self.model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{version}")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.resid_pdrop = self.model.config.attn_pdrop = self.model.config.embd_pdrop = 0.6

    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

    @staticmethod
    def tokenizer_cls():
        return "DialoGPTTokenizer"

    def forward(self, input_seq: torch.LongTensor, **_) -> torch.Tensor:
        input_seq, input_mask = self.create_padding_mask(input_seq)
        out = self.model(
            input_ids=input_seq,
            attention_mask=input_mask,
        )
        return out.logits

# -----------------------------------------------------------------------------
