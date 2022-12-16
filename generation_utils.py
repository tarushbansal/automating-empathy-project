# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
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


class BlenderBot(Generate):
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        tokenizer: AutoTokenizer,
        device: torch.device,
        beam_width: int,
        max_len: int
    ) -> None:

        super().__init__(model, tokenizer, device, beam_width, max_len)
        model.config.max_position_embeddings = 512

    def generate(self, batch: List[List[str]]) -> List[List[int]]:
        batch = [f"{f' {self.tokenizer.sep_token} '.join(history)} {self.tokenizer.sep_token}"
                 for history in batch]
        input_ids = self.tokenizer(batch, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_len,
            num_beams=self.beam_width
        )
        return outputs.tolist()

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
            num_beams=self.beam_width
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
        # Make modifications to tokenizer to allow padding
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer.pad_token = tokenizer.eos_token
        print("Model tokenizer does not contain any padding tokens! Setting pad token to eos token.")
        self.instruction = "Respond empathetically to this conversation: "

    def generate(self, batch: List[List[str]]) -> List[List[int]]:
        _inputs = [] 
        for history in batch:
            dialog = self.instruction + " "
            for i in range(len(history)):
                if i % 2 == 0:
                    dialog += f"Speaker: {history[i]} ;"
                else:
                    dialog += f"Listener: {history[i]} ;"
            dialog += "Listener: "
            _inputs.append(dialog)
        input_ids = self.tokenizer(_inputs, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.size(dim=1)
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=(seq_len + self.max_len),
            num_beams=self.beam_width
        )
        outputs = outputs[:, seq_len:]
        return outputs.tolist()


generate_map = {
    "gpt2": GPT2,
    "microsoft/GODEL-v1_1-base-seq2seq": GODEL,
    "facebook/blenderbot-400M-distill": BlenderBot
}

# ----------------------------------------------------------------------------
