# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
from typing import Union, List

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
        device: torch.device,
        beam_width: int,
        max_len: int
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_len = max_len
        self.device = device
        self.model.eval()


class GODEL(Generate):
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        tokenizer: AutoTokenizer,
        device: torch.device,
        beam_width: int,
        max_len: int
    ) -> None:

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
            top_p=0.9,
            do_sample=True
        )
        return outputs.tolist()


class GPT2(Generate):
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        tokenizer: AutoTokenizer,
        device: torch.device,
        beam_width: int,
        max_len: int
    ) -> None:

        super().__init__(model, tokenizer, device, beam_width, max_len)
        self.instruction = "Respond empathetically to this conversation: "

    def generate(self, batch: List[List[str]]) -> List[List[int]]:
        batch = [self.instruction + " ".join(history) for history in batch]
        input_ids = self.tokenizer(batch, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.size(dim=1)
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=(seq_len + self.max_len),
            top_p=0.9,
            do_sample=True
        )
        outputs = outputs[:, seq_len:]
        return outputs.tolist()


generate_map = {
    "gpt2": GPT2,
    "microsoft/GODEL-v1_1-base-seq2seq": GODEL
}

# ----------------------------------------------------------------------------
