# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
from typing import List, Tuple, Union, Optional

# User-Defined Modules
from data.data_classes import ConceptNetRawData

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
        pass

    def decode_to_text(self, sequence: List[int]) -> str:
        try:
            i = sequence.index(self.PAD_IDX)
        except ValueError:
            i = len(sequence)
        decoded_text = self.tokenizer.decode(
            sequence[:i],
            skip_special_tokens=True).strip()
        return decoded_text

#---------------------------------------------------------------------------