# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
from typing import List, Union

import torch
import torch.utils.data as data
import pytorch_lightning as pl

# User-defined Modules
from transformers import BertTokenizer
from data_classes import RewardModelRawData, RewardModelBatch

# ------------------------- IMPLEMENTATION -----------------------------------


class RewardDataset(data.Dataset):
    def __init__(
        self,
        dialogues: List[List[str]],
        rewards: List[float],
        tokenizer: BertTokenizer
    ) -> None:
        
        self.dialogues = dialogues
        self.rewards = rewards
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> RewardModelRawData:

        dialogue = self.tokenizer(self.dialogues[idx])  
        reward = self.rewards[idx]

        return RewardModelRawData(
            dialogue=dialogue,
            reward=reward
        )


class RewardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        tokenizer: BertTokenizer,
        num_workers: int
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    def load_data(
        self, 
        fpath: str
    ) -> Union[List[str], List[List[str]]]:

        with open(fpath) as f:
            data = json.load(f)
        return data

    def load_dataset(self, stage: str):
        dialogues = self.load_data(f"{self.dataset_dir}/{stage}/dialogues.json")
        rewards = self.load_data(f"{self.dataset_dir}/{stage}/rewards.json")
        return RewardDataset(
            dialogues=dialogues,
            rewards=rewards,
            tokenizer=self.tokenizer
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.load_dataset("train")
        if stage == "test":
            self.test_dataset = self.load_dataset("test")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, RewardModelBatch):
            batch.dialogues = batch.dialogues.to(device)
            batch.rewards = batch.rewards.to(device)
        else:
            batch = super().transfer_batch_to_device(data, device, dataloader_idx)
        return batch

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            num_workers=self.num_workers
        )


    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_token_id),
            num_workers=self.num_workers
        )

def pad_seq_and_convert_to_tensor(
    sequences: List[int],
    max_len: int,
    padding_idx: int,
    dtype: torch.dtype = torch.long
) -> torch.Tensor:

    sequences = [seq + [padding_idx] * (max_len - len(seq)) for seq in sequences]

    return torch.tensor(sequences, dtype=dtype)

def collate_fn(batch: List[RewardModelRawData], pad_idx: int):
    dialogues = [item.dialogue for item in batch]
    max_len = max([len(seq) for seq in dialogues])
    dialogues = torch.LongTensor([seq + [pad_idx] * (max_len - len(seq)) for seq in dialogues])
    rewards = torch.FloatTensor([item.reward for item in batch])
    return RewardModelBatch(
        dialogues=dialogues,
        rewards=rewards
    )