# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Tuple

import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput

# User-Defined Modules
from base_classes import EncoderDecoderModel, DecoderModel, TokenizerBase
from data_classes import ConceptNetBatchData

# ------------------------- IMPLEMENTATION ------------------------------------


class BlenderBot(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.hidden_size = self.model.config.hidden_size

    def tokenizer_cls():
        return "BlenderBotTokenizer"

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
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/GODEL-v1_1-{version}-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.6
        self.hidden_size = self.model.config.hidden_size

    def tokenizer_cls():
        return "GODELTokenizer"

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


class KnowledgeBridgedGODEL(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/GODEL-v1_1-{version}-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.8
        self.graph_embeddings = nn.Embedding(2, self.hidden_size)
        self.emo_linear = nn.Linear(self.model.config.hidden_size, self.tokenizer.num_emo_labels)
        self.attn_loss = nn.MSELoss()
        self.hidden_size = self.model.config.hidden_size
        self.requires_concept_net_data = True

    def tokenizer_cls():
        return "GODELTokenizer"

    def word_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def knowledge_enriched_context(
        self,
        context: torch.LongTensor,
        context_mask: torch.BoolTensor,
        concepts: torch.LongTensor,
        concept_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor]:

        pad_mask = torch.cat((context_mask, concept_mask), dim=1)

        context_graph_embeds = self.graph_embeddings(
            torch.zeros(context.size(), dtype=torch.long, device=context.device))
        context_embeds = self.word_embeddings()(context) + context_graph_embeds
        concept_graph_embeds = self.graph_embeddings(
            torch.ones(concepts.size(), dtype=torch.long, device=concepts.device))
        concept_embeds = self.word_embeddings()(concepts) + concept_graph_embeds
        embeddings = torch.cat((context_embeds, concept_embeds), dim=1)

        return embeddings, pad_mask

    def forward(
        self,
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
        concept_net_data: ConceptNetBatchData
    ) -> Seq2SeqLMOutput:

        context_embeds, pad_mask = self.knowledge_enriched_context(
            contexts, context_mask, concept_net_data.concepts, concept_net_data.concept_mask)
        # attention_mask = torch.minimum(pad_mask.unsqueeze(1), concept_net_data.adjacency_mask)

        out = self.model(
            inputs_embeds=context_embeds,
            attention_mask=pad_mask,
            decoder_input_ids=targets,
            decoder_attention_mask=target_mask,
            output_attentions=True,
            output_hidden_states=True
        )

        emo_intensities = torch.cat(
            (concept_net_data.context_emo_intensity,
             concept_net_data.concept_emo_intensity), dim=1)
        sum_weights = torch.softmax(emo_intensities, dim=1).unsqueeze(2)
        c = torch.sum(sum_weights * out.encoder_last_hidden_state, dim=1)
        self.emo_logits = self.emo_linear(c)

        average_attn_weights = torch.stack(out.cross_attentions).mean((0, 2, 3))
        self.emo_attn_loss = self.attn_loss(emo_intensities, average_attn_weights)

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
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["small", "medium", "large"]:
            raise ValueError("Model version must be 'small', 'medium' or 'large'!")
        self.model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{version}")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.resid_pdrop = self.model.config.attn_pdrop = self.model.config.embd_pdrop = 0.6
        self.hidden_size = self.model.config.hidden_size

    def tokenizer_cls():
        return "DialoGPTTokenizer"

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
