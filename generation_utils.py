# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
from typing import Union

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)

# ------------------------- IMPLEMENTATION -----------------------------------

class Generate:
    def __init__(
        self, 
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
        tokenizer: AutoTokenizer, 
        beam_width: int, 
        max_len: int
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_len = max_len

class GODEL(Generate):
    def __init__(
        self, 
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
        tokenizer: AutoTokenizer, 
        beam_width: int, 
        max_len: int
    ) -> None:

        super().__init__(model, tokenizer, beam_width, max_len)
    
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

class GPT2(Generate):
    def __init__(
        self, 
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
        tokenizer: AutoTokenizer, 
        beam_width: int, 
        max_len: int
    ) -> None:
    
        super().__init__(model, tokenizer, beam_width, max_len)
    
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(dim=1)
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id, 
            max_length=self.max_len, 
            n_beams=self.beam_width
        )
        outputs = outputs[:, seq_len:]
        return outputs

generate_map = {
    "gpt2": GPT2,
    "microsoft/GODEL-v1_1-base-seq2seq": GODEL
}

# ----------------------------------------------------------------------------