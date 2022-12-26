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

# User-Defined Modules
from data_classes import GenerationConfig

# ------------------------- IMPLEMENTATION -----------------------------------


class PretrainedGenerationBase:
    def __init__(
        self,
        model: Union[AutoModelForSeq2SeqLM, AutoModelForCausalLM],
        tokenizer: AutoTokenizer,
        device: torch.device,
        generation_config: GenerationConfig
    ) -> None:

        self.model = model
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = generation_config


class GODEL(PretrainedGenerationBase):
    def __init__(
        self,
        device: torch.device,
        generation_config: GenerationConfig
    ) -> None:

        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
        super().__init__(model, tokenizer, device, generation_config)

        self.instruction = "Instruction: given a dialog context, you need to response empathically."

    def generate(self, batch: Dict[str, Tuple]) -> Tuple[Union[List[List[int]], float, int]]:
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
        sum_cross_entropy = F.cross_entropy(
            output.logits[:, :-1, :].permute(0, 2, 1),
            decoder_input.input_ids[:, 1:],
            reduction="sum",
            ignore_index=self.tokenizer.pad_token_id
        )
        num_tokens = torch.sum(decoder_input.input_ids[:, 1:] != self.tokenizer.pad_token_id)

        predictions = self.model.generate(
            _input.input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.generation_config.max_new_tokens,
            num_beams=self.generation_config.beam_width,
            do_sample=self.generation_config.sample,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k
        )

        return predictions.tolist(), sum_cross_entropy, num_tokens

# ----------------------------------------------------------------------------
