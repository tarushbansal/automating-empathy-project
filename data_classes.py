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
        context: List[List[int]],
        target: List[int],
        emotion: Optional[List[int]],
        concept_net_data: Optional[ConceptNetRawData]
    ):
        self.context = context
        self.target = target
        self.emotion = emotion
        self.concept_net_data = concept_net_data


class DecoderModelRawData:
    def __init__(
        self,
        dialogue: List[List[int]],
        target: Optional[List[int]],
        emotion: Optional[List[int]],
    ):
        self.dialogue = dialogue
        self.target = target
        self.emotion = emotion


class EncoderDecoderModelBatch:
    def __init__(
        self,
        contexts: torch.LongTensor,
        targets: torch.LongTensor,
        emotions: Optional[torch.LongTensor],
        concept_net_data: Optional[ConceptNetBatchData]
    ) -> None:

        self.contexts = contexts
        self.targets = targets
        self.emotions = emotions
        self.concept_net_data = concept_net_data


class DecoderModelBatch:
    def __init__(
        self,
        dialogues: torch.LongTensor,
        targets: Optional[List[List[int]]],
        emotions: Optional[torch.LongTensor],
    ) -> None:

        self.dialogues = dialogues
        self.targets = targets
        self.emotions = emotions

# ---------------------------------------------------------------------------------