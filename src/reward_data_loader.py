# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
from typing import List, Union, Tuple

import torch
import torch.utils.data as data
import pytorch_lightning as pl

# User-defined Modules
from base_classes import TokenizerBase
from data_loader import pad_seq_and_convert_to_tensor
from data_classes import RewardModelRawData, RewardModelBatch

# ------------------------- IMPLEMENTATION -----------------------------------


class RewardDataset(data.Dataset):
    def __init__(
        self,
        contexts: List[List[str]],
        responses: List[List[str]],
        ratings: List[List[Tuple[int]]],
        tokenizer: TokenizerBase
    ) -> None:
        
        self.contexts = contexts
        self.responses = responses
        self.ratings = ratings
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> RewardModelRawData:
        context = self.tokenizer.encode_text(self.contexts[idx])[0]
        targets = [] 
        for response in self.responses[idx]:
            targets.append(self.tokenizer.encode_text(response)[0])
        ratings = self.ratings[idx]

        return RewardModelRawData(
            context=context,
            targets=targets,
            ratings=ratings
        )


class RewardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: TokenizerBase,
        num_workers: int
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    def load_data(
        self, 
        fpath: str
    ) -> Union[List[str], List[List[str]]]:

        with open(fpath) as f:
            data = json.load(f)
        return data

    def load_dataset(self):
        contexts = self.load_data(f"{self.dataset_dir}/train/contexts.json")
        responses = self.load_data(f"{self.dataset_dir}/train/responses.json")
        ratings = self.load_data(f"{self.dataset_dir}/train/ratings.json")
        return RewardDataset(
            contexts=contexts,
            responses=responses,
            ratings=ratings,
            tokenizer=self.tokenizer
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.load_dataset()
            self.val_dataset = self.load_dataset()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, RewardModelBatch):
            batch.contexts = batch.contexts.to(device)
            batch.context_mask = batch.context_mask.to(device)
            batch.targets = batch.targets.to(device)
            batch.target_mask = batch.target_mask.to(device)
        else:
            batch = super().transfer_batch_to_device(data, device, dataloader_idx)
        return batch

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x[0], self.tokenizer),
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=lambda x: collate_fn(x[0], self.tokenizer),
            num_workers=self.num_workers
        )


def collate_fn(data: RewardModelRawData, tokenizer: TokenizerBase):
    contexts = torch.LongTensor([data.context for _ in range(len(data.targets))])
    context_mask = torch.ones_like(contexts)
    max_len_target_seq = max([len(seq) for seq in data.targets])
    targets = pad_seq_and_convert_to_tensor(data.targets, max_len_target_seq, tokenizer.PAD_IDX)
    target_mask = (targets != tokenizer.PAD_IDX)
    return RewardModelBatch(
        contexts=contexts,
        context_mask=context_mask,
        targets=targets,
        target_mask=target_mask,
        ratings=data.ratings
    )