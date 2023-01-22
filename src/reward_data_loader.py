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

        dialogue = "[CLS] " + " [SEP] ".join(self.dialogues[idx])
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

    def load_dataset(self, split: str):
        dialogues = self.load_data(f"{self.dataset_dir}/{split}/dialogues.json")
        rewards = self.load_data(f"{self.dataset_dir}/{split}/rewards.json")
        return RewardDataset(
            dialogues=dialogues,
            rewards=rewards,
            tokenizer=self.tokenizer
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.load_dataset("train")
            self.val_dataset = self.load_dataset("val")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, RewardModelBatch):
            batch.dialogues = batch.dialogues.to(device)
            batch.mask = batch.mask.to(device)
            batch.rewards = batch.rewards.to(device)
        else:
            batch = super().transfer_batch_to_device(data, device, dataloader_idx)
        return batch

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, self.tokenizer),
            num_workers=self.num_workers
        )


    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: collate_fn(x, self.tokenizer),
            num_workers=self.num_workers
        )


def collate_fn(batch: List[RewardModelRawData], tokenizer: BertTokenizer):
    out = tokenizer([item.dialogue for item in batch], return_tensors="pt", padding=True)
    dialogues = out.input_ids
    mask = out.attention_mask
    rewards = torch.FloatTensor([item.reward for item in batch])
    return RewardModelBatch(
        dialogues=dialogues,
        rewards=rewards,
        mask=mask
    )