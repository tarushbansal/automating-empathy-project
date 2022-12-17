# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
from typing import List, Union

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

# ------------------------- IMPLEMENTATION -----------------------------------


class GenerationBase:
    def __init__(
        self,
        model: Union[AutoModelForSeq2SeqLM, AutoModelForCausalLM],
        tokenizer: AutoTokenizer,
        device: torch.device,
        beam_width: int,
        max_len: int
    ) -> None:

        self.model = model
        self.model = self.model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.beam_width = beam_width
        self.max_len = max_len


class GODEL(GenerationBase):
    def __init__(
        self,
        device: torch.device,
        beam_width: int,
        max_len: int
    ) -> None:

        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        super().__init__(model, tokenizer, device, beam_width, max_len)

        self.instruction = "Instruction: given a dialog context, you need to response empathically."

    def generate(self, batch: List[List[str]]) -> List[List[int]]:
        batch = [f"{self.instruction} [CONTEXT] {' EOS '.join(history)}" for history in batch]
        input_ids = self.tokenizer(batch, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_len,
            num_beams=self.beam_width
        )
        return outputs.tolist()

# ----------------------------------------------------------------------------
