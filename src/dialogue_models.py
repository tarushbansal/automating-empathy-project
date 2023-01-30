# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Tuple, Optional

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
        self.word_embeddings = self.model.get_input_embeddings()
        self.hidden_size = self.model.config.hidden_size

    @staticmethod
    def tokenizer_cls():
        return "BlenderBotTokenizer"

    def forward(
        self,
        contexts: torch.LongTensor,
        targets: torch.LongTensor,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Seq2SeqLMOutput:

        contexts, context_mask = self.create_padding_mask(contexts)
        target_mask = None
        if past_key_values is None:
            targets, target_mask = self.create_padding_mask(targets)

        out = self.model(
            input_ids=contexts,
            attention_mask=context_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=targets,
            decoder_attention_mask=target_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        return out


class GODEL(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/GODEL-v1_1-{version}-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.6
        self.word_embeddings = self.model.get_input_embeddings()
        self.hidden_size = self.model.config.hidden_size

    @staticmethod
    def tokenizer_cls():
        return "GODELTokenizer"

    def forward(
        self,
        contexts: torch.LongTensor,
        targets: torch.LongTensor,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Seq2SeqLMOutput:

        contexts, context_mask = self.create_padding_mask(contexts)
        targets, target_mask = self.create_padding_mask(targets)

        out = self.model(
            input_ids=contexts,
            attention_mask=context_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=targets,
            decoder_attention_mask=target_mask,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        return out


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
        self.word_embeddings = self.model.get_input_embeddings()
        self.hidden_size = self.model.config.hidden_size
        self.requires_concept_net_data = True

    @staticmethod
    def tokenizer_cls():
        return "GODELTokenizer"

    def knowledge_enriched_context(
        self,
        context: torch.LongTensor,
        concepts: torch.LongTensor,
    ) -> Tuple[torch.Tensor]:

        context, context_mask = self.create_padding_mask(context)
        concepts, concept_mask = self.create_padding_mask(concepts)
        pad_mask = torch.cat((context_mask, concept_mask), dim=1)

        context_graph_embeds = self.graph_embeddings(
            torch.zeros(context.size(), dtype=torch.long, device=context.device))
        context_embeds = self.word_embeddings(context) + context_graph_embeds
        concept_graph_embeds = self.graph_embeddings(
            torch.ones(concepts.size(), dtype=torch.long, device=concepts.device))
        concept_embeds = self.word_embeddings(concepts) + concept_graph_embeds
        embeddings = torch.cat((context_embeds, concept_embeds), dim=1)

        return embeddings, pad_mask

    def forward(
        self,
        contexts: torch.LongTensor,
        targets: torch.LongTensor,
        concept_net_data: ConceptNetBatchData,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Seq2SeqLMOutput:

        context_embeds, pad_mask = self.knowledge_enriched_context(
            contexts, concept_net_data.concepts)
        # attention_mask = torch.minimum(pad_mask.unsqueeze(1), concept_net_data.adjacency_mask)
        targets, target_mask = self.create_padding_mask(targets)

        out = self.model(
            inputs_embeds=context_embeds,
            attention_mask=pad_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=targets,
            decoder_attention_mask=target_mask,
            output_attentions=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            use_cache=use_cache
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


class DialoGPT(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase, version: str) -> None:
        super().__init__(tokenizer)
        if version not in ["small", "medium", "large"]:
            raise ValueError("Model version must be 'small', 'medium' or 'large'!")
        self.model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{version}")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.resid_pdrop = self.model.config.attn_pdrop = self.model.config.embd_pdrop = 0.6
        self.word_embeddings = self.model.get_input_embeddings()
        self.hidden_size = self.model.config.hidden_size

    @staticmethod
    def tokenizer_cls():
        return "DialoGPTTokenizer"

    def forward(
        self, 
        dialogues: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> CausalLMOutput:
        dialogues, mask = self.create_padding_mask(dialogues)
        out = self.model(
            input_ids=dialogues,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        return out
    
# -----------------------------------------------------------------------------
