# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- IMPLEMENTATION -----------------------------------


class MultiHeadAttention(nn.Module):
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

        self.attention = MultiHeadAttention(embed_size, heads)

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

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        num_layers: int,
        embed_size: int,
        heads: int,
        dropout: float, 
        forward_expansion: int,
        max_seq_len: int,
        num_of_emo_labels: int,
        pretrained_embed: torch.Tensor = None
    ) -> None:

        super().__init__()

        if pretrained_embed is not None:
            if pretrained_embed.size() != (vocab_size, embed_size):
                raise ValueError(
                    f"Specified model hyperparameters (vocab_size, embed_size)={(vocab_size,embed_size)} "
                    + f"not compatible with pretrained embedding matrix of size {tuple(pretrained_embed.size())}"
                )
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embed)
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(max_seq_len, embed_size)
        self.ds_embeddings = nn.Embedding(3, embed_size, padding_idx=padding_idx)
        self.emotion_embedding = nn.Embedding(num_of_emo_labels, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(
        self,
        input_seq: torch.Tensor,
        dialogue_state: torch.Tensor,
        input_mask: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> torch.Tensor:

        N, seq_len = input_seq.shape
        positions = torch.arange(0, seq_len, device=input_seq.device).expand(N, seq_len)

        word_embeddings = self.word_embeddings(input_seq)
        pos_embeddings = self.pos_embeddings(positions)
        ds_embeddings = self.ds_embeddings(dialogue_state)

        out = self.dropout(word_embeddings + pos_embeddings + ds_embeddings)
        
        for layer in self.layers:
            out = layer(out, out, out, input_mask)
        
        emotion_embedding = self.emotion_embedding(
            emotion_label).unsqueeze(1).expand(-1, seq_len, -1)
        
        out = self.fc_out(out + emotion_embedding)

        return out

class GenerativeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_of_emo_labels: int,
        max_seq_len: int,
        padding_idx: int,
        num_layers: int = 6,
        embed_size: int = 300,
        heads: int = 10,
        dropout: float = 0.5, 
        forward_expansion: int = 4,
        pretrained_embed: torch.Tensor = None
    ) -> None:

        super().__init__()

        self.padding_idx = padding_idx
        self.decoder = Decoder(
            vocab_size,
            padding_idx,
            num_layers, 
            embed_size, 
            heads,
            dropout, 
            forward_expansion, 
            max_seq_len,
            num_of_emo_labels,
            pretrained_embed
        )

    def create_padding_mask(self, batch_seq):
        N = batch_seq.size(dim=0)
        padding_mask = (batch_seq != self.padding_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask
    
    def create_lookahead_mask(self, batch_seq):
        N, seq_len = batch_seq.shape
        lookahead_mask = torch.tril(torch.ones(
            N, 1, seq_len, seq_len, device=batch_seq.device))
        return lookahead_mask
    
    def forward(
        self,
        input_seq: torch.Tensor,
        dialogue_state: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> None:

        input_mask = torch.minimum(
            self.create_lookahead_mask(input_seq), 
            self.create_padding_mask(input_seq)
        )

        out = self.decoder(
            input_seq, 
            dialogue_state,
            input_mask,
            emotion_label
        )

        return out