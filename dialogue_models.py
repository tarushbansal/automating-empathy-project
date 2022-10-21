# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertModel

# User-Defined Modules
from transformer_modules import TransformerBlock, Decoder
from base_classes import EncoderDecoderModel, DecoderModel

# ------------------------- IMPLEMENTATION ------------------------------------

class GenerativeTransformer(DecoderModel):
    def __init__(
        self,
        vocab_size: int,
        num_emo_labels: int,
        padding_idx: int,
        max_seq_len: int = 1000,
        num_layers: int = 6,
        embed_size: int = 300,
        heads: int = 10,
        dropout: float = 0, 
        forward_expansion: int = 4,
        pretrained_embed: torch.Tensor = None
    ) -> None:

        super().__init__(vocab_size, num_emo_labels, padding_idx)

        self.padding_idx = padding_idx
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )

        if pretrained_embed is not None:
            if pretrained_embed.size() != (vocab_size, embed_size):
                raise ValueError(
                    f"""Specified model hyperparameters 
                        (vocab_size, embed_size)={(vocab_size,embed_size)}
                        not compatible with pretrained embedding matrix 
                        of size {tuple(pretrained_embed.size())}"""
                )
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embed)
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(max_seq_len, embed_size)
        self.ds_embeddings = nn.Embedding(3, embed_size, padding_idx=padding_idx)
        self.emotion_embedding = nn.Embedding(num_emo_labels, embed_size)

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(
        self,
        input_seq: torch.Tensor,
        input_dialogue_state: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> None:

        input_mask = torch.minimum(
            self.create_lookahead_mask(input_seq), 
            self.create_padding_mask(input_seq)
        )

        N, seq_len = input_seq.shape
        positions = torch.arange(0, seq_len, device=input_seq.device).expand(N, seq_len)

        word_embeddings = self.word_embeddings(input_seq)
        pos_embeddings = self.pos_embeddings(positions)
        ds_embeddings = self.ds_embeddings(input_dialogue_state)

        out = self.dropout(word_embeddings + pos_embeddings + ds_embeddings)

        for layer in self.layers:
            out = layer(out, out, out, input_mask)

        emotion_embeddings = self.emotion_embedding(
            emotion_label).unsqueeze(1).expand(out.size())
        out = self.fc_out(self.dropout(out + emotion_embeddings))

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