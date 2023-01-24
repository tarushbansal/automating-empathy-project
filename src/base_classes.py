# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn

from typing import Tuple, Optional, Union, List

# User-Defined Modules
from data_classes import ConceptNetRawData

# ------------------------- IMPLEMENTATION -----------------------------------

class TokenizerBase:
    def __init__(self) -> None:
        # Value of PAD_IDX does not matter due to the paddding mask created in the model
        # However, it should not conflict with any token id in the vocab!!
        self.PAD_IDX = -100

        # No start of sentence tokens will be used for target responses by default
        self.SOS_IDX = None
        
        # Emotion label map
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 
            'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9,
            'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 
            'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31
        }
        self.rev_emo_map = {v : k for k, v in self.emo_map.items()}
        self.num_emo_labels = len(self.emo_map)

    def encode_text(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None
    ) -> Tuple[Union[List[int], Optional[ConceptNetRawData]]]:
        raise NotImplementedError

    def decode_to_text(self, sequence: List[int]) -> str:
        try:
            i = sequence.index(self.PAD_IDX)
        except ValueError:
            i = len(sequence)
        decoded_text = self.tokenizer.decode(
            sequence[:i],
            skip_special_tokens=True).strip()
        return decoded_text

class DialogueModelBase(nn.Module):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer
    
    @staticmethod
    def tokenizer_cls():
        raise NotImplementedError

    @property
    def has_encoder(self) -> bool:
        raise NotImplementedError
    
    @property
    def hidden_size(self) -> int:
        raise NotImplementedError
    
    @property
    def word_embeddings(self):
        raise NotImplementedError
        
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
        raise NotImplementedError


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
        raise NotImplementedError

#---------------------------------------------------------------------------