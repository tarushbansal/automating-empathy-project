# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from typing import Tuple, Optional, Union, List

from transformers import AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutput

# User-Defined Modules
from data_classes import ConceptNetRawData, CustomSeq2SeqLMOutput, CustomCausalLMOutput

# ------------------------- IMPLEMENTATION -----------------------------------

class TokenizerBase(ABC):
    def __init__(self) -> None:
        # Start sequence for generation is none by default
        self._SOS_IDX = None

        # Properties to be set in child class
        self._EOS_IDX = None
        self._PAD_IDX = None
        self._vocab_size = None
        
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

    @property
    def SOS_IDX(self) -> int:
        return self._SOS_IDX
    
    @SOS_IDX.setter
    def SOS_IDX(self, value: int):
        if type(value) != int:
            raise TypeError("Property must be of type 'int'!")
        self._SOS_IDX = value

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise NotImplementedError
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value: int):
        if type(value) != int:
            raise TypeError("Property must be of type 'int'!")
        self._vocab_size = value

    @property
    def EOS_IDX(self) -> int:
        if self._EOS_IDX is None:
            raise NotImplementedError
        return self._EOS_IDX
    
    @EOS_IDX.setter
    def EOS_IDX(self, value: int):
        if type(value) != int:
            raise TypeError("Property must be of type 'int'!")
        self._EOS_IDX = value

    @property
    def PAD_IDX(self) -> int:
        if self._PAD_IDX is None:
            raise NotImplementedError
        return self._PAD_IDX
    
    @PAD_IDX.setter
    def PAD_IDX(self, value: int):
        if type(value) != int:
            raise TypeError("Property must be of type 'int'!")
        self._PAD_IDX = value

    @abstractmethod
    def encode_text(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None
    ) -> Tuple[Union[List[int], Optional[ConceptNetRawData]]]:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self, 
        sequences: Union[List[int], List[List[int]], torch.Tensor]
    ) -> List[str]:
        raise NotImplementedError


class HuggingFaceTokenizerBase(TokenizerBase):
    def __init__(self, tokenizer: AutoTokenizer, add_bos_token: bool = True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        special_tokens = {"pad_token": "<PAD>"}
        if add_bos_token:
            special_tokens["bos_token"] = "<SOS>"
        self.tokenizer.add_special_tokens(special_tokens)
        self.PAD_IDX = self.tokenizer.pad_token_id
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
    
    def decode(
        self, 
        sequences: Union[List[int], List[List[int]], torch.Tensor]
    ) -> List[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=True)


class DialogueModelBase(ABC, nn.Module):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self._has_encoder = None
        self._hidden_size = None
        self._requires_emotion_label = False
        self._requires_concept_net_data = False
    
    @staticmethod
    @abstractmethod
    def tokenizer_cls():
        raise NotImplementedError

    @property
    def has_encoder(self) -> bool:
        if self._has_encoder is None:
            raise NotImplementedError
        return self._has_encoder
    
    @has_encoder.setter
    def has_encoder(self, value: bool) -> None:
        if type(value) != bool:
            raise TypeError("Property must be of type 'bool'!")
        self._has_encoder = bool

    @property
    def hidden_size(self) -> bool:
        if self._hidden_size is None:
            raise NotImplementedError
        return self._hidden_size
    
    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        if type(value) != int:
            raise TypeError("Property must be of type 'int'!")
        self._hidden_size = value
    
    @abstractmethod
    def word_embeddings(self) -> nn.Embedding:
        raise NotImplementedError
        
    @property
    def requires_emotion_label(self) -> bool:
        return self._requires_emotion_label
    
    @requires_emotion_label.setter
    def requires_emotion_label(self, value: bool) -> None:
        if type(value) != bool:
            raise TypeError("Property must be of type 'bool'!")
        self._requires_emotion_label = bool
    
    @property
    def requires_concept_net_data(self) -> bool:
        return self._requires_concept_net_data
    
    @requires_concept_net_data.setter
    def requires_concept_net_data(self, value: bool) -> None:
        if type(value) != bool:
            raise TypeError("Property must be of type 'bool'!")
        self._requires_concept_net_data = bool

    @abstractmethod
    def generate(
        self,
        contexts: torch.LongTensor, 
        context_mask: torch.BoolTensor, 
        **kwargs
    ) -> torch.LongTensor:
        raise NotImplementedError


class EncoderDecoderModel(DialogueModelBase):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
    
    @property
    def has_encoder(self) -> bool:
        return True

    @abstractmethod
    def forward(
        self,
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
        **kwargs
    ) -> Union[Seq2SeqLMOutput, CustomSeq2SeqLMOutput]:
        raise NotImplementedError


class DecoderModel(DialogueModelBase):
    def __init__(self, tokenizer: TokenizerBase) -> None:
        super().__init__(tokenizer)
    
    @property
    def has_encoder(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self, 
        dialogues: torch.LongTensor,
        dialogue_mask: torch.BoolTensor,
        **kwargs
    ) -> Union[CausalLMOutput, CustomCausalLMOutput]:
        raise NotImplementedError

#---------------------------------------------------------------------------