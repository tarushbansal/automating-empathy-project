# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn

from typing import Tuple

# User-Defined Modules
from tokenizers.base import TokenizerBase

# ------------------------- IMPLEMENTATION -----------------------------------


class DialogueModelBase(nn.Module):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer
    
    @staticmethod
    def tokenizer_cls():
        return None

    @property
    def word_embeddings(self):
        return None
        
    @property
    def requires_emotion_label(self) -> bool:
        return False
    
    @property
    def requires_concept_net_data(self) -> bool:
        return False

    def create_padding_mask(self, batch_seq) -> Tuple[torch.Tensor]:
        padding_mask = (batch_seq != self.tokenizer.PAD_IDX)
        batch_seq = batch_seq.masked_fill(padding_mask == 0, 0)
        return batch_seq, padding_mask
    
    def create_lookahead_mask(self, batch_seq) -> torch.Tensor:
        N, seq_len = batch_seq.shape
        lookahead_mask = torch.tril(torch.ones(
            N, 1, seq_len, seq_len, device=batch_seq.device))
        return lookahead_mask


class EncoderDecoderModel(DialogueModelBase):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
    
    @property
    def has_encoder(self) -> bool:
        # DO NOT OVERWRITE THIS PROPERTY
        return True

    def forward(
        self,
        source_seq: torch.Tensor,
        target_seq: torch.Tensor,
        source_dialogue_state: torch.Tensor,
        target_dialogue_state: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> torch.Tensor:
        pass


class DecoderModel(DialogueModelBase):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
    
    @property
    def has_encoder(self) -> bool:
        # DO NOT OVERWRITE THIS PROPERTY
        return False

    def forward(
        self,
        input_seq: torch.Tensor,
        input_dialogue_state: torch.Tensor,
        emotion_label: torch.Tensor
    ) -> torch.Tensor:
        pass

#---------------------------------------------------------------------------