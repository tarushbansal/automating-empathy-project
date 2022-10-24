# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- IMPLEMENTATION ------------------------------------

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int) -> None:
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        if self.head_dim * self.heads != self.embed_size:
            raise ValueError(
                f"embed_size {embed_size} not divisible by number of heads {heads}")

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, bias=False)

    def forward(
        self,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        queries: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:

        N = queries.shape[0]
        keys_len, values_len, queries_len = keys.shape[1], values.shape[1], queries.shape[1]

        values = self.values(values).reshape(N, values_len, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(N, keys_len, self.heads, self.head_dim)
        queries = self.queries(queries).reshape(N, queries_len, self.heads, self.head_dim)

        scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask to attention scores if specified
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        # Normalise with respect to all keys
        attention = F.softmax(scores / (self.embed_size ** 0.5), dim=-1)

        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values])
        out = self.fc_out(out.reshape(N, queries_len, self.embed_size))

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size: int, 
        heads: int, 
        dropout: float, 
        forward_expansion: int
    ) -> None:

        super().__init__()

        self.attention = MultiHeadedAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

    def forward(
        self,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        queries: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:

        attention = self.attention(keys, values, queries, mask)

        contextualised = self.dropout(self.norm1(attention + queries))
        forward = self.ff(contextualised)
        out = self.dropout(self.norm2(forward + contextualised))

        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int, 
        heads: int, 
        dropout: float, 
        forward_expansion: int
    ) -> None:

        super().__init__()

        self.attention = MultiHeadedAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )

    def forward(
        self,
        keys: torch.Tensor, 
        values: torch.Tensor, 
        queries: torch.Tensor, 
        target_mask: torch.Tensor,
        source_mask: torch.Tensor
    ) -> torch.Tensor:

        attention = self.attention(queries, queries, queries, target_mask)
        queries = self.dropout(self.norm(attention + queries))
        out = self.transformer_block(keys, values, queries, source_mask)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_size: int,
        heads: int,
        dropout: float, 
        forward_expansion: int,
        num_emo_labels: int
    ) -> None:

        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )

        self.emo_embeddings = nn.Embedding(num_emo_labels, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(
        self,
        embedded_target: torch.Tensor,
        encoder_out: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> torch.Tensor:
        
        out = embedded_target
        
        for layer in self.layers:
            out = layer(encoder_out, encoder_out, out, target_mask, source_mask)
        
        emo_embeddings = self.emo_embeddings(
            emotion_label).unsqueeze(1).expand(out.size())
        
        out = self.fc_out(out + emo_embeddings)

        return out

#---------------------------------------------------------------------------------------