# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import random
import numpy as np
from typing import Tuple, Dict, Iterable, List, Union

import torch
import torch.utils.data as data
import pytorch_lightning as pl

# User-defined Modules
from data_tokenizers import TokenizerBase

# ------------------------- IMPLEMENTATION -----------------------------------


class Dataset(data.Dataset):
    def __init__(
        self,
        data: Tuple[Iterable],
        tokenizer: TokenizerBase
    ) -> None:

        self.tokenizer = tokenizer
        self.contexts, self.targets, self.emotions = data

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict:
        """returns one data pair"""
        item = {}

        # Map emotion label to token
        item['emotion'] = self.tokenizer.emo_map[self.emotions[idx]]

        # Tokenize dialogue history / context
        context, context_ds, external_knowledge = self.tokenizer.encode_text(
            self.contexts[idx], "context"
        )
        item["context"] = context
        item["context_dialogue_state"] = context_ds
        item["external_knowledge"] = external_knowledge

        # Tokenize response utterance
        target, target_ds, _ = self.tokenizer.encode_text(
            self.targets[idx], "target"
        )
        item['target'] = target
        item['target_dialogue_state'] = target_ds

        return item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        tokenizer: TokenizerBase,
        num_workers: int
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    def load_data(self, stage: str):
        contexts = np.load(
            f"{self.dataset_dir}/{stage}/contexts.npy", allow_pickle=True)
        targets = np.load(
            f"{self.dataset_dir}/{stage}/targets.npy", allow_pickle=True)
        emotions = np.load(
            f"{self.dataset_dir}/{stage}/emotions.npy", allow_pickle=True)
        return contexts, targets, emotions

    def setup(self, stage: str) -> None:
        if stage == "fit":
            data = list(zip(*self.load_data("train")))
            # Set seed for reproducability
            random.seed(0)
            # Shuffle data to form train and validation sets
            random.shuffle(data)
            contexts, targets, emotions = zip(*data)
            split_index = int(0.9 * len(contexts))
            train_data = (
                contexts[:split_index],
                targets[:split_index],
                emotions[:split_index]
            )
            val_data = (
                contexts[split_index:],
                targets[split_index:],
                emotions[split_index:]
            )
            self.train_dataset = Dataset(train_data, self.tokenizer)
            self.val_dataset = Dataset(val_data, self.tokenizer)
        if stage == "test":
            test_data = self.load_data("test")
            self.test_dataset = Dataset(test_data, self.tokenizer)

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_batch(x, self.tokenizer),
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_batch(x, self.tokenizer),
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_batch(x, self.tokenizer),
            num_workers=self.num_workers
        )


def collate_batch(
    batch: List[Dict],
    tokenizer: TokenizerBase
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:

    context = [item["context"] for item in batch]
    target = [item["target"] for item in batch]
    emotion = [item["emotion"] for item in batch]

    max_len_context_seq = max([len(seq) for seq in context])
    max_len_target_seq = max([len(seq) for seq in target])

    collated_batch = {
        "context": pad_seq_and_convert_to_tensor(
            context, max_len_context_seq, padding_idx=tokenizer.PAD_IDX),
        "target": pad_seq_and_convert_to_tensor(
            target, max_len_target_seq, padding_idx=tokenizer.PAD_IDX),
        "emotion":  torch.LongTensor(emotion),
    }

    if tokenizer.supports_dialogue_states:
        add_dialogue_states(
            collated_batch,
            batch,
            max_len_context_seq,
            max_len_target_seq
        )

    if tokenizer.supports_external_knowledge:
        add_external_knowledge(
            collated_batch,
            batch,
            max_len_context_seq,
            tokenizer.PAD_IDX,
            len(getattr(tokenizer, "prefix", []))
        )

    return collated_batch


def create_adjacency_mask(
    concept_mask: List[List[int]],
    max_context_len: int,
    max_concept_len: int,
    prefix_len: int
) -> torch.BoolTensor:

    N = len(concept_mask)
    # Pad concept_mask
    for i in range(N):
        concept_mask[i] = [row + [0] * (max_concept_len - len(row))
                           for row in concept_mask[i]]
        for _ in range(max_context_len - len(concept_mask[i])):
            concept_mask[i].append([0] * max_concept_len)

    # Create adjacency mask A from submask C, identity matrix I and ones / zeros
    # A = [[1 C], [[0 1], I]]
    concept_mask = torch.BoolTensor(concept_mask)
    assert concept_mask.size() == (N, max_context_len, max_concept_len)
    upper = torch.cat((
        torch.ones(N, max_context_len, max_context_len),
        concept_mask), dim=-1)
    lower = torch.cat((
        torch.zeros(N, max_concept_len, prefix_len),
        torch.ones(N, max_concept_len, max_context_len + max_concept_len - prefix_len),
    ), dim=-1)
    adjacency_mask = torch.cat((upper, lower), dim=1)

    return adjacency_mask


def pad_seq_and_convert_to_tensor(
    sequences: List[int],
    max_len: int,
    padding_idx: int,
    dtype: torch.dtype = torch.long
) -> torch.Tensor:

    sequences = [seq + [padding_idx] * (max_len - len(seq)) for seq in sequences]

    return torch.tensor(sequences, dtype=dtype)


def add_dialogue_states(
    collated_batch: Dict[str, torch.Tensor],
    batch: List[Dict],
    max_len_context_seq: int,
    max_len_target_seq: int
) -> None:

    context_dialogue_state = [item["context_dialogue_state"] for item in batch]
    target_dialogue_state = [item["target_dialogue_state"]for item in batch]

    collated_batch["context_dialogue_state"] = pad_seq_and_convert_to_tensor(
        context_dialogue_state, max_len_context_seq, padding_idx=0)
    collated_batch["target_dialogue_state"] = pad_seq_and_convert_to_tensor(
        target_dialogue_state, max_len_target_seq, padding_idx=0)


def add_external_knowledge(
    collated_batch: Dict[str, torch.Tensor],
    batch: List[Dict],
    max_len_context_seq: int,
    padding_idx: int,
    prefix_len: int = 0
) -> None:

    concepts = [item["external_knowledge"]["concepts"] for item in batch]
    context_emo_intensity = [item["external_knowledge"]["context_emo_intensity"] for item in batch]
    concept_emo_intensity = [item["external_knowledge"]["concept_emo_intensity"] for item in batch]
    concept_mask = [item["external_knowledge"]["concept_mask"] for item in batch]
    max_len_concept_seq = max([len(seq) for seq in concepts])

    concepts = pad_seq_and_convert_to_tensor(
        concepts, max_len_concept_seq, padding_idx=padding_idx)
    # context_emo_intensity = pad_seq_and_convert_to_tensor(
    #     context_emo_intensity, max_len_context_seq, padding_idx=0, dtype=torch.float32)
    # concept_emo_intensity = pad_seq_and_convert_to_tensor(
    #     concept_emo_intensity, max_len_concept_seq, padding_idx=0, dtype=torch.float32)
    # adjacency_mask = create_adjacency_mask(
    #     concept_mask, max_len_context_seq, max_len_concept_seq, prefix_len)

    collated_batch["external_knowledge"] = {
        "concepts": concepts,
        # "adjacency_mask": adjacency_mask,
        # "context_emo_intensity": context_emo_intensity,
        # "concept_emo_intensity": concept_emo_intensity
    }

# -----------------------------------------------------------------------------
