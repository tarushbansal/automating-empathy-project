# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import List, Tuple

import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    BertModel
)

# User-Defined Modules
from components import Decoder, TransformerBlock
from base_classes import EncoderDecoderModel, DecoderModel, TokenizerBase

# ------------------------- IMPLEMENTATION ------------------------------------

class GODEL(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)

    @staticmethod
    def tokenizer_cls():
        return "GODELTokenizer"

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
    def __init__(
        self, 
        tokenizer: TokenizerBase, 
        dropout: float = 0.5, 
        forward_expansion: int = 4
    ) -> None:

        super().__init__(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.emotional_context = TransformerBlock(
            self.model.config.hidden_size,
            self.model.config.num_attention_heads,
            dropout=dropout,
            forward_expansion=forward_expansion
        )
        self.vertex_embeddings = nn.Embedding(2, self.model.config.hidden_size)

    @staticmethod
    def tokenizer_cls():
        return "KnowledgeBridgedGODELTokenizer"

    def construct_emotional_context(
        self,
        context: torch.LongTensor, 
        concepts: torch.LongTensor,
        adjacency_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor]:

        context, context_mask = self.create_padding_mask(context)
        concepts, concept_mask = self.create_padding_mask(concepts)
        transf_context_mask = context_mask.unsqueeze(2) * context_mask.unsqueeze(1)
        graph_attention_mask = torch.cat((transf_context_mask, adjacency_mask), dim=-1).unsqueeze(1)

        model_embeddings = self.model.get_input_embeddings()
        context_vert_embeds = self.vertex_embeddings(
            torch.zeros(context.size(), dtype=torch.long, device=context.device))
        context_embeds = model_embeddings(context) + context_vert_embeds
        concept_vert_embeds = self.vertex_embeddings(
            torch.ones(concepts.size(), dtype=torch.long, device=concepts.device))
        concept_embeds = model_embeddings(concepts) + concept_vert_embeds
        input_embeds = torch.cat((context_embeds, concept_embeds), dim=1)

        emotional_context = self.emotional_context(
            keys=input_embeds,
            values=input_embeds,
            queries=context_embeds,
            mask=graph_attention_mask
        )

        emotional_context = torch.cat((emotional_context, concept_embeds), dim=1)
        emotional_context_mask = torch.cat((context_mask, concept_mask), dim=1)

        return emotional_context, emotional_context_mask

    def forward(
        self, 
        source_seq: torch.LongTensor, 
        target_seq: torch.LongTensor,
        concepts: torch.LongTensor,
        adjacency_mask: torch.BoolTensor
    ) -> torch.Tensor:

        emo_context, emo_context_mask = self.construct_emotional_context(
            source_seq, concepts, adjacency_mask)
        target_seq, target_mask = self.create_padding_mask(target_seq)

        out = self.model(
            inputs_embeds=emo_context,
            attention_mask=emo_context_mask,
            decoder_input_ids=target_seq,
            decoder_attention_mask=target_mask
        )
        return out.logits


class GPT2(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
    
    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

    @staticmethod
    def tokenizer_cls():
        return "GPT2Tokenizer"

    def forward(self, input_seq: torch.LongTensor) -> torch.Tensor:
        input_seq, input_mask = self.create_padding_mask(input_seq)
        
        out = self.model(
            input_ids=input_seq,
            attention_mask=input_mask,
        )
        return out.logits


class PrependDialoGPT2(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.ds_embeddings = nn.Embedding(2, self.model.config.hidden_size)
        self.emo_embeddings = nn.Embedding(tokenizer.num_emo_labels, self.model.config.hidden_size)
    
    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

    @staticmethod
    def tokenizer_cls():
        return "GPT2Tokenizer"

    @property
    def requires_emotion_label(self) -> bool:
        return True

    def forward(
        self, 
        input_seq: torch.LongTensor, 
        input_dialogue_state: torch.LongTensor, 
        emotion_label: torch.LongTensor
    ) -> torch.Tensor:

        input_seq, input_mask = self.create_padding_mask(input_seq)
        input_dialogue_state, _ = self.create_padding_mask(input_dialogue_state)
        word_embeds = self.word_embeddings(input_seq)
        ds_embeds = self.ds_embeddings(input_dialogue_state)
        input_embeds = word_embeds + ds_embeds
        
        # Prepend emotion label and offset mask
        emotion = self.emo_embeddings(emotion_label).unsqueeze(1)
        input_embeds = torch.cat((emotion, input_embeds), dim=1)
        mask_offset = torch.ones(input_mask.size(dim=0), 1, 
                                 dtype=torch.long, device=input_mask.device)
        input_mask = torch.cat((mask_offset, input_mask), dim=1)
        
        out = self.model(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
        )

        return out.logits[:, 1:, :]


class BertEncodedTransformer(EncoderDecoderModel):
    def __init__(
        self,
        tokenizer: TokenizerBase,
        num_layers: int = 6,
        dropout: float = 0, 
        forward_expansion: int = 4,
        freeze_bert: bool = False
    ) -> None:

        super().__init__(tokenizer)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.word_embeddings = self.bert.embeddings
        embed_size = self.bert.config.hidden_size
        heads = self.bert.config.num_attention_heads
        self.decoder = Decoder(
            tokenizer.vocab_size,
            num_layers,
            embed_size,
            heads,
            dropout,
            forward_expansion,
            tokenizer.num_emo_labels
        )
    
    @property
    def word_embeddings(self):
        return self.model.word_embeddings

    @staticmethod
    def tokenizer_cls():
        return "BertTokenizer"
    
    def forward(
        self,
        source_seq: torch.LongTensor,
        target_seq: torch.LongTensor,
        emotion_label: torch.LongTensor
    ) -> torch.Tensor:
        
        encoder_out = self.bert(source_seq)[0][-1]

        source_seq, source_mask = self.create_padding_mask(source_seq)
        target_seq, target_mask = self.create_padding_mask(target_seq)
        
        target_mask = torch.minimum(
            target_mask,
            self.create_lookahead_mask(target_seq)
        )

        embedded_target = self.word_embeddings(target_seq)
        out = self.decoder(
            embedded_target, 
            encoder_out, 
            target_mask, 
            source_mask, 
            emotion_label
        )

        return out

# -----------------------------------------------------------------------------