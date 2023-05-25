# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
import torch
from typing import List, Tuple, Dict

# ------------------------- IMPLEMENTATION ----------------------------------------


class ModelConfig:
    def __init__(
        self,
        model_cls: str,
        model_kwargs: Dict,
        tokenizer_cls: str,
        tokenizer_kwargs: Dict
    ) -> None:
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.tokenizer_cls = tokenizer_cls
        self.tokenizer_kwargs = tokenizer_kwargs


class ModelRawData:
    def __init__(
        self,
        context: List[int],
        target: List[int],
        raw_context: List[List[str]]
    ) -> None:
        self.context = context
        self.target = target
        self.raw_context = raw_context


class RewardModelRawData:
    def __init__(
        self,
        context: List[int],
        targets: List[List[int]],
        ratings: List[Tuple[int]]
    ) -> None:

        self.context = context
        self.targets = targets
        self.ratings = ratings


class ModelBatch:
    def __init__(
        self,
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
        raw_contexts: List[List[str]]
    ) -> None:

        self.contexts = contexts
        self.context_mask = context_mask
        self.targets = targets
        self.target_mask = target_mask
        self.raw_contexts = raw_contexts


class RewardModelBatch:
    def __init__(
        self,
        contexts: torch.LongTensor,
        context_mask: torch.BoolTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
        ratings: List[Tuple[int]]
    ) -> None:

        self.contexts = contexts
        self.context_mask = context_mask
        self.targets = targets
        self.target_mask = target_mask
        self.ratings = ratings


class CustomSeq2SeqLMOutput:
    def __init__(self) -> None:
        raise NotImplementedError


class CustomCausalLMOutput:
    def __init__(self) -> None:
        raise NotImplementedError


class GenerationConfig:
    def __init__(
        self,
        max_new_tokens: int,
        beam_width: int,
        sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        length_alpha: float
    ) -> None:
        
        self.max_new_tokens = max_new_tokens
        self.beam_width = beam_width
        self.sample = sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.length_alpha = length_alpha


class PPOConfig:
    def __init__(
        self,
        clip_epsilon: float,
        kl_penalty: float,
        gamma: float,
        lam: float,
        vf_coeff: float, 
        entropy_coeff: float, 
    ) -> None:
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        self.gamma = gamma
        self.lam = lam
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

# ---------------------------------------------------------------------------------