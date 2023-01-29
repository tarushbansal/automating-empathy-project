# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
import torch
from typing import List, Optional

# ------------------------- IMPLEMENTATION ----------------------------------------

class ConceptNetRawData:
    def __init__(
        self,
        concepts: List[int],
        concept_mask: List[List[bool]],
        context_emo_intensity: List[float],
        concept_emo_intensity: List[float]
    ):
        self.concepts = concepts
        self.concept_mask = concept_mask
        self.context_emo_intensity = context_emo_intensity
        self.concept_emo_intensity = concept_emo_intensity


class ConceptNetBatchData:
    def __init__(
        self,
        concepts: torch.LongTensor,
        adjacency_mask: torch.BoolTensor,
        context_emo_intensity: torch.FloatTensor,
        concept_emo_intensity: torch.FloatTensor
    ):
        self.concepts = concepts
        self.adjacency_mask = adjacency_mask
        self.context_emo_intensity = context_emo_intensity
        self.concept_emo_intensity = concept_emo_intensity


class EncoderDecoderModelRawData:
    def __init__(
        self,
        context: List[int],
        raw_context: List[List[str]],
        target: List[int],
        emotion: Optional[List[int]],
        concept_net_data: Optional[ConceptNetRawData]
    ):
        self.context = context
        self.raw_context = raw_context
        self.target = target
        self.emotion = emotion
        self.concept_net_data = concept_net_data


class DecoderModelRawData:
    def __init__(
        self,
        dialogue: List[List[int]],
        raw_dialogue: List[List[str]],
        target: Optional[List[int]],
        emotion: Optional[List[int]],
    ):
        self.dialogue = dialogue
        self.raw_dialogue = raw_dialogue
        self.target = target
        self.emotion = emotion


class RewardModelRawData:
    def __init__(
        self,
        dialogue: List[str],
        reward: float
    ) -> None:

        self.dialogue = dialogue
        self.reward = reward

class EncoderDecoderModelBatch:
    def __init__(
        self,
        contexts: torch.LongTensor,
        raw_contexts: List[List[str]],
        targets: torch.LongTensor,
        emotions: Optional[torch.LongTensor],
        concept_net_data: Optional[ConceptNetBatchData]
    ) -> None:

        self.contexts = contexts
        self.raw_contexts = raw_contexts
        self.targets = targets
        self.emotions = emotions
        self.concept_net_data = concept_net_data


class DecoderModelBatch:
    def __init__(
        self,
        dialogues: torch.LongTensor,
        raw_dialogues: List[List[str]],
        targets: Optional[torch.LongTensor],
        emotions: Optional[torch.LongTensor],
    ) -> None:

        self.dialogues = dialogues
        self.raw_dialogues = raw_dialogues
        self.targets = targets
        self.emotions = emotions


class RewardModelBatch:
    def __init__(
        self,
        dialogues: torch.LongTensor,
        rewards: torch.FloatTensor,
        mask: torch.BoolTensor
    ) -> None:

        self.dialogues = dialogues
        self.rewards = rewards
        self.mask = mask


class GenerationConfig:
    def __init__(
        self,
        max_new_tokens: int,
        beam_width: int,
        sample: bool,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> None:
        
        self.max_new_tokens = max_new_tokens
        self.beam_width = beam_width
        self.sample = sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


class PPOConfig:
    def __init__(
        self,
        clip_epsilon: Optional[float] = None,
        kl_penalty: Optional[float] = None,
        gamma: Optional[float] = None,
        _lambda: Optional[float] = None,
        vf_coeff: Optional[float] = None, 
        entropy_coeff: Optional[float] = None, 
    ) -> None:
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        self.gamma = gamma
        self._lambda = _lambda
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

# ---------------------------------------------------------------------------------