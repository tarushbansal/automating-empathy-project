# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    BertModel
)

# User-Defined Modules
from components import Decoder
from base_classes import EncoderDecoderModel, DecoderModel

# ------------------------- IMPLEMENTATION ------------------------------------

class HuggingFaceEncoderDecoderModel(EncoderDecoderModel):
    def __init__(
        self, 
        vocab_size: int, 
        num_emo_labels: int, 
        padding_idx: int,
        model_name: str
    ) -> None:

        super().__init__(vocab_size, num_emo_labels, padding_idx)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(vocab_size)
    
    def forward(
        self, 
        source_seq: torch.Tensor, 
        target_seq: torch.Tensor, 
        source_dialogue_state: torch.Tensor, 
        target_dialogue_state: torch.Tensor, 
        emotion_label: torch.Tensor = None
    ) -> torch.Tensor:

        out = self.model(
            input_ids=source_seq,
            attention_mask=(source_seq!=self.padding_idx),
            decoder_input_ids=target_seq,
            decoder_attention_mask=(target_seq!=self.padding_idx)
        )
        return out.logits


class HuggingFaceDecoderModel(DecoderModel):
    def __init__(
        self, 
        vocab_size: int, 
        num_emo_labels: int, 
        padding_idx: int,
        model_name: str
    ) -> None:

        super().__init__(vocab_size, num_emo_labels, padding_idx)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(vocab_size)
    
    def forward(
        self, 
        input_seq: torch.Tensor, 
        input_dialogue_state: torch.Tensor, 
        emotion_label: torch.Tensor = None
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_seq,
            attention_mask=(input_seq!=self.padding_idx),
        )
        return out.logits


class AffectiveDecodingModel(DecoderModel):
    def __init__(
        self, 
        vocab_size: int, 
        num_emo_labels: int, 
        padding_idx: int,
        model_name: str,
        emo_embed_dim: int = 512,
        dropout: float = 0
    ) -> None:

        super().__init__(vocab_size, num_emo_labels, padding_idx)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.emo_embeddings = nn.Embedding(num_emo_labels, emo_embed_dim)
        self.fc_layer = nn.Linear(emo_embed_dim, vocab_size)
    
    def forward(
        self, 
        input_seq: torch.Tensor, 
        input_dialogue_state: torch.Tensor, 
        emotion_label: torch.Tensor = None
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_seq,
            attention_mask=(input_seq!=self.padding_idx),
        )
        out = out.logits
        emo_embedding = self.emo_embeddings(emotion_label)
        emo_embedding = emo_embedding.unsqueeze(1).expand(-1, out.size(dim=1), -1)
        out += + self.fc_layer(emo_embedding)
        
        return out

class BertEncodedTransformer(EncoderDecoderModel):
    def __init__(
        self,
        vocab_size: int,
        num_emo_labels: int,
        padding_idx: int,
        num_layers: int = 6,
        dropout: float = 0, 
        forward_expansion: int = 4,
        freeze_bert: bool = False
    ) -> None:

        super().__init__(vocab_size, num_emo_labels, padding_idx)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.embeddings = self.bert.embeddings
        embed_size = self.bert.config.hidden_size
        heads = self.bert.config.num_attention_heads
        self.decoder = Decoder(
            vocab_size,
            num_layers,
            embed_size,
            heads,
            dropout,
            forward_expansion,
            num_emo_labels
        )
    
    def forward(
        self,
        source_seq: torch.Tensor,
        target_seq: torch.Tensor,
        source_dialogue_state: torch.Tensor,
        target_dialogue_state: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> torch.Tensor:
        
        encoder_out = self.bert(source_seq)[0][-1]

        source_mask = self.create_padding_mask(source_seq)
        target_mask = torch.minimum(
            self.create_lookahead_mask(target_seq), 
            self.create_padding_mask(target_seq)
        )

        embedded_target = self.embeddings(target_seq)
        out = self.decoder(
            embedded_target, 
            encoder_out, 
            target_mask, 
            source_mask, 
            emotion_label
        )

        return out

# -----------------------------------------------------------------------------