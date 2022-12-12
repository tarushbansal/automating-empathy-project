# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BertModel
)

# User-Defined Modules
from components import Decoder
from base_classes import EncoderDecoderModel, DecoderModel, TokenizerBase

# ------------------------- IMPLEMENTATION ------------------------------------


class GODEL(EncoderDecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.8

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
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.dropout_rate = 0.8
        self.graph_embeddings = nn.Embedding(2, self.model.config.hidden_size)
        # self.emo_linear = nn.Linear(self.model.config.hidden_size, self.tokenizer.num_emo_labels)
        self.attn_loss = nn.MSELoss()

    @staticmethod
    def tokenizer_cls():
        return "KnowledgeBridgedGODELTokenizer"

    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

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
        external_knowledge: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        input_embeds, pad_mask = self.knowledge_enriched_context(
            source_seq, external_knowledge["concepts"])
        attention_mask = torch.minimum(pad_mask.unsqueeze(1), external_knowledge["adjacency_mask"])
        target_seq, target_mask = self.create_padding_mask(target_seq)

        out = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            encoder_attention_mask=pad_mask,
            decoder_input_ids=target_seq,
            decoder_attention_mask=target_mask,
            output_attentions=True
        )

        emo_intensities = torch.cat(
            (external_knowledge["context_emo_intensity"],
             external_knowledge["concept_emo_intensity"]), dim=1)
        # sum_weights = torch.softmax(emo_intensities, dim=1).unsqueeze(2)
        # c = torch.sum(sum_weights * out.encoder_last_hidden_state, dim=1)
        # self.emo_logits = self.emo_linear(c)

        average_attn_weights = torch.stack(out.cross_attentions).mean((0, 2, 3))
        self.emo_attn_loss = self.attn_loss(emo_intensities, average_attn_weights)

        return out.logits


class GPT2(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.resid_pdrop = self.model.config.attn_pdrop = self.model.config.embd_pdrop = 0.6

    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

    @staticmethod
    def tokenizer_cls():
        return "GPT2Tokenizer"

    def forward(self, input_seq: torch.LongTensor, **_) -> torch.Tensor:
        input_seq, input_mask = self.create_padding_mask(input_seq)

        out = self.model(
            input_ids=input_seq,
            attention_mask=input_mask,
        )
        return out.logits


class KnowledgeBridgedGPT2(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.graph_embeddings = nn.Embedding(2, self.model.config.hidden_size)

    @staticmethod
    def tokenizer_cls():
        return "KnowledgeBridgedGPT2Tokenizer"

    @property
    def word_embeddings(self):
        return self.model.get_input_embeddings()

    def knowledge_enriched_context(
        self,
        input_seq: torch.LongTensor,
        concepts: torch.LongTensor,
    ) -> Tuple[torch.Tensor]:

        dialog, dialog_mask = self.create_padding_mask(input_seq)
        concepts, concept_mask = self.create_padding_mask(concepts)
        pad_mask = torch.cat((concept_mask, dialog_mask), dim=1)

        model_embeddings = self.model.get_input_embeddings()
        dialog_graph_embeds = self.graph_embeddings(
            torch.zeros(dialog.size(), dtype=torch.long, device=dialog.device))
        dialog_embeds = model_embeddings(dialog) + dialog_graph_embeds
        concept_graph_embeds = self.graph_embeddings(
            torch.ones(concepts.size(), dtype=torch.long, device=concepts.device))
        concept_embeds = model_embeddings(concepts) + concept_graph_embeds
        embeddings = torch.cat((concept_embeds, dialog_embeds), dim=1)

        return embeddings, pad_mask

    def forward(
        self,
        input_seq: torch.LongTensor,
        external_knowledge: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        concept_size = external_knowledge["concepts"].size(dim=1)
        input_embeds, pad_mask = self.knowledge_enriched_context(
            input_seq, external_knowledge["concepts"])

        out = self.model(
            inputs_embeds=input_embeds,
            attention_mask=pad_mask,
            output_attentions=True
        )

        return out.logits[:, concept_size:, :]


class PrependDialoGPT2(DecoderModel):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.resize_token_embeddings(tokenizer.vocab_size)
        self.model.config.resid_pdrop = self.model.config.attn_pdrop = self.model.config.embd_pdrop = 0.6
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
