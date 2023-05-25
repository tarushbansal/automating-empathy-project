# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
from typing import List, Union, Optional

from transformers import AutoTokenizer

# User-Defined Modules
from base_classes import HuggingFaceTokenizerBase

# ------------------------- IMPLEMENTATION ----------------------------------------


class BlenderBotTokenizer(HuggingFaceTokenizerBase):
    def __init__(self) -> None:
        super().__init__(AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill"))
        self.tokenizer.truncation_side = "left"

    def encode_text(
        self,
        text: Union[str, List[str]],
        *_
    ) -> List[int]:

        if type(text) == list:
            dialogue = "</s> <s>".join(text)
            token_ids = self.tokenizer(
                dialogue,
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )["input_ids"]

        else:
            token_ids = self.tokenizer(f"<s> {text}")["input_ids"]
            token_ids = token_ids[:self.tokenizer.model_max_length]

        return token_ids


class GODELTokenizer(HuggingFaceTokenizerBase):
    def __init__(
        self,
        version: str
    ) -> None:

        if version not in ["base", "large"]:
            raise ValueError("Model version must be either 'base' or 'large'!")
        super().__init__(AutoTokenizer.from_pretrained(
            f"microsoft/GODEL-v1_1-{version}-seq2seq"
        ))
        self.tokenizer.truncation_side = "left"
        self.default_instruction = "Given the dialog context, you need to respond to respond empathetically."

    def encode_text(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
        knowledge: Optional[str] = None
    ) -> List[int]:

        if type(text) == list:
            if instruction is None:
                instruction = self.default_instruction
            instruction = f"Instruction: {instruction}"
            knowledge = f"[KNOWLEDGE] {knowledge}" if knowledge is not None else ""
            dialogue = f' EOS '.join(text)
            token_ids = self.tokenizer(
                f"{instruction} {knowledge} [CONTEXT] {dialogue}".strip())["input_ids"]
            if len(token_ids) > self.tokenizer.model_max_length:
                token_ids = self.tokenizer(
                    f"{instruction} [CONTEXT] ")["input_ids"]
                token_ids.extend(
                    self.tokenizer(
                        dialogue,
                        truncation=True,
                        max_length=self.tokenizer.model_max_length - len(token_ids)
                    )["input_ids"]
                )

        else:
            token_ids = self.tokenizer(
                f"{self.tokenizer.pad_token} {text}")["input_ids"]
            token_ids = token_ids[:self.tokenizer.model_max_length]

        return token_ids


class DialoGPTTokenizer(HuggingFaceTokenizerBase):
    def __init__(self, version: str) -> None:
        if version not in ["small", "medium", "large"]:
            raise ValueError("Model version must be 'small', 'medium' or 'large'!")
        super().__init__(AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{version}"))
        self.tokenizer.truncation_side = "left"

    def encode_text(
        self,
        text: Union[str, List[str]],
        *_
    ) -> List[int]:
    
        if type(text) == list:
            input_ = f"{f' {self.tokenizer.eos_token} '.join(text)} {self.tokenizer.eos_token}"
            token_ids = self.tokenizer(
                input_,
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )["input_ids"]
        else:
            token_ids = self.tokenizer(text)["input_ids"] + [self.tokenizer.eos_token_id]
            token_ids = token_ids[:self.tokenizer.model_max_length]

        return token_ids

# ---------------------------------------------------------------------------------
