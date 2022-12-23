# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import torch
import torch.nn.functional as F

from typing import List, Union, Tuple, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

# ------------------------- IMPLEMENTATION -----------------------------------


class PretrainedGenerationBase:
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
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.beam_width = beam_width
        self.max_len = max_len


class GODEL(PretrainedGenerationBase):
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

    def generate(self, batch: Dict[str, Tuple]) -> Tuple[List[List[int]], float]:
        _input = [f"{self.instruction} [CONTEXT] {' EOS '.join(dialogue)}"
                  for dialogue in batch["contexts"]]
        _input = self.tokenizer(_input, return_tensors="pt", padding=True).to(self.device)
        decoder_input = self.tokenizer(batch["targets"], return_tensors="pt", padding=True).to(self.device)
        output = self.model(
            input_ids=_input.input_ids,
            attention_mask=_input.attention_mask,
            decoder_input_ids=decoder_input.input_ids,
            decoder_attention_mask=decoder_input.attention_mask
        )
        cross_entropy = F.cross_entropy(
            output.logits[:, :-1, :].permute(0, 2, 1),
            decoder_input.input_ids[:, 1:],
            ignore_index=self.tokenizer.pad_token_id
        )

        predictions = self.model.generate(
            _input.input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_len,
            num_beams=self.beam_width
        )

        return predictions.tolist(), cross_entropy

# ----------------------------------------------------------------------------
