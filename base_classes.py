# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn

from typing import List, Tuple

# ------------------------- IMPLEMENTATION -----------------------------------

class TokenizerBase:
    def __init__(self) -> None:
        # Value of PAD_IDX does not matter due to the paddding mask created in the model
        # However, it should not conflict with any token id in the vocab!!
        self.PAD_IDX = -100

        # Dialog state indices
        self.DS_SPEAKER_IDX = 1
        self.DS_LISTENER_IDX = 2
        
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
        self.num_emo_labels = len(self.emo_map)

    def encode_text(self, sequences: List[List[str]]) -> List[List[int]]:
        pass

    def decode_to_text(self, sequence: List[int]) -> str:
        pass


class DialogueModelBase(nn.Module):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer
    
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