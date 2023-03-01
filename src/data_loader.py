# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import random
from typing import Optional, List, Union, Tuple

import torch
import torch.utils.data as data
import pytorch_lightning as pl

# User-defined Modules
from base_classes import TokenizerBase
from data_classes import (
    ModelRawData,
    ModelBatch
)

# ------------------------- IMPLEMENTATION -----------------------------------


class Dataset(data.Dataset):
    def __init__(
        self,
        tokenizer: TokenizerBase,
        contexts: List[List[str]],
        targets: List[str],
        instructions: Optional[List[str]] = None,
        knowledge: Optional[List[str]] = None
    ) -> None:
        
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.targets = targets
        self.instructions = instructions
        self.knowledge = knowledge

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(
        self, 
        idx: int
    ) -> ModelRawData:

        # Tokenize dialogue context
        context, concept_net_data = self.tokenizer.encode_text(
            self.contexts[idx], 
            None if self.instructions is None else self.instructions[idx],
            None if self.knowledge is None else self.knowledge[idx]
        )

        # Tokenize response utterance
        target, _ = self.tokenizer.encode_text(self.targets[idx])

        return ModelRawData(
            context=context,
            target=target,
            raw_context=self.contexts[idx],
            concept_net_data=concept_net_data
        )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        tokenizer: TokenizerBase,
        num_workers: int,
        model_has_encoder: bool,
        data_erasure_level: float,
        few_shot_training: bool = False
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.model_has_encoder = model_has_encoder
        self.data_erasure_level = data_erasure_level
        self.few_shot_training = few_shot_training

        if self.data_erasure_level < 0 or self.data_erasure_level > 1.0:
            raise ValueError("Data erasure level must be between 0 and 1")

    def load_data(
        self, 
        fpath: str
    ) -> Union[List[str], List[List[str]]]:

        with open(fpath) as f:
            data = json.load(f)
        return data

    def load_augmented_data(
        self, 
        fpath: str
    ) -> Optional[List[str]]:
        
        data = None
        if os.path.isfile(fpath):
            data = self.load_data(fpath)
            random.seed(42)
            for i in random.sample(
                range(len(data)), 
                k=int(self.data_erasure_level*len(data))
            ):
                data[i] = None
        return data


    def load_dataset(self, split: str):
        path_prefix = f"{self.dataset_dir}/{split}"

        contexts = self.load_data(f"{path_prefix}/contexts.json")
        targets = self.load_data(f"{path_prefix}/targets.json")
        
        # Load additional data if present
        instructions = self.load_augmented_data(f"{path_prefix}/instructions.json")
        knowledge = self.load_augmented_data(f"{path_prefix}/knowledge.json")
        
        if split != "test":
            if self.few_shot_training:
                # TODO: Implement Few Shot training if required here !!!
                pass

        return Dataset(
            self.tokenizer,
            contexts,
            targets,
            instructions,
            knowledge
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.load_dataset("train")
            self.val_dataset = self.load_dataset("val")
        if stage == "test":
            self.test_dataset = self.load_dataset("test")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, ModelBatch):
            batch.contexts = batch.contexts.to(device)
            batch.context_mask = batch.context_mask.to(device)
            batch.targets = batch.targets.to(device)
            batch.target_mask = batch.target_mask.to(device)
            data = batch.concept_net_data
            if data is not None:
                data.concepts = data.concepts.to(device)
                data.adjacency_mask = data.adjacency_mask.to(device)
                data.context_emo_intensity = data.context_emo_intensity.to(device)
                data.concept_emo_intensity = data.concept_emo_intensity.to(device)
        else:
            batch = super().transfer_batch_to_device(data, device, dataloader_idx)
        return batch

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_batch(x, self.tokenizer, self.model_has_encoder),
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_batch(x, self.tokenizer, self.model_has_encoder),
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_batch(x, self.tokenizer, self.model_has_encoder),
            num_workers=self.num_workers
        )


def collate_batch(
    batch: List[ModelRawData],
    tokenizer: TokenizerBase,
    model_has_encoder: bool
) -> ModelBatch:

    contexts, context_mask = pad_to_tensor(
        [item.context for item in batch],
        pad_token=tokenizer.PAD_IDX, 
        padding_side="right" if model_has_encoder else "left"
    )
    raw_contexts = [item.raw_context for item in batch]
    targets, target_mask = pad_to_tensor(
        [item.target for item in batch],
        pad_token=tokenizer.PAD_IDX
    )

    concept_net_data = None
    # if batch[0].concept_net_data is not None:
    #     concept_net_data = process_concept_net_data(
    #         [item.concept_net_data for item in batch],
    #         max_len_context_seq,
    #         tokenizer.PAD_IDX,
    #         len(getattr(tokenizer, "prefix", []))
    #     )

    return ModelBatch(
        contexts=contexts,
        context_mask=context_mask,
        targets=targets,
        target_mask=target_mask,
        raw_contexts=raw_contexts,
        concept_net_data=concept_net_data
    )


def pad_to_tensor(
    sequences: List[List[int]],
    pad_token: int,
    padding_side: str = "right",
    dtype: torch.dtype = torch.long
) -> Tuple[torch.Tensor]:

    max_len = max([len(seq) for seq in sequences])
    if padding_side == "right":
        sequences = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
    elif padding_side == "left":
        sequences = [[pad_token] * (max_len - len(seq)) + seq for seq in sequences]

    tensor = torch.tensor(sequences, dtype=dtype)
    mask = (tensor != pad_token)

    return tensor, mask


# def process_concept_net_data(
#     data: List[ConceptNetRawData],
#     max_len_context_seq: int,
#     pad_token: int,
#     prefix_len: int = 0
# ) -> None:

#     concepts = [item.concepts for item in data]
#     context_emo_intensity = [item.context_emo_intensity for item in data]
#     concept_emo_intensity = [item.concept_emo_intensity for item in data]
#     concept_mask = [item.concept_mask for item in data]
#     max_len_concept_seq = max([len(seq) for seq in concepts])

#     concepts = pad_to_tensor(
#         concepts, max_len_concept_seq, pad_token=pad_token)
#     context_emo_intensity = pad_to_tensor(
#         context_emo_intensity, max_len_context_seq, pad_token=0, dtype=torch.float32)
#     concept_emo_intensity = pad_to_tensor(
#         concept_emo_intensity, max_len_concept_seq, pad_token=0, dtype=torch.float32)
#     adjacency_mask = create_adjacency_mask(
#         concept_mask, max_len_context_seq, max_len_concept_seq, prefix_len)

#     return ConceptNetBatchData(
#         concepts=concepts,
#         adjacency_mask=adjacency_mask,
#         context_emo_intensity=context_emo_intensity,
#         concept_emo_intensity=concept_emo_intensity
#     )


# def create_adjacency_mask(
#     concept_mask: List[List[List[int]]],
#     max_context_len: int,
#     max_concept_len: int,
#     prefix_len: int
# ) -> torch.BoolTensor:

#     N = len(concept_mask)
#     # Pad concept_mask
#     for i in range(N):
#         concept_mask[i] = [row + [0] * (max_concept_len - len(row))
#                            for row in concept_mask[i]]
#         for _ in range(max_context_len - len(concept_mask[i])):
#             concept_mask[i].append([0] * max_concept_len)

#     # Create adjacency mask A from submask C, identity matrix I and ones / zeros
#     # A = [[1 C], [[0 1], I]]
#     concept_mask = torch.BoolTensor(concept_mask)
#     assert concept_mask.size() == (N, max_context_len, max_concept_len)
#     upper = torch.cat((
#         torch.ones(N, max_context_len, max_context_len),
#         concept_mask), dim=-1)
#     lower = torch.cat((
#         torch.zeros(N, max_concept_len, prefix_len),
#         torch.ones(N, max_concept_len, max_context_len + max_concept_len - prefix_len),
#     ), dim=-1)
#     adjacency_mask = torch.cat((upper, lower), dim=1)

#     return adjacency_mask

# -----------------------------------------------------------------------------
