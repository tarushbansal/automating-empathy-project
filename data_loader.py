# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import random
import numpy as np
from typing import Tuple, List, Dict, Iterable

import torch
import torch.utils.data as data
import pytorch_lightning as pl

# User-defined Modules
from token_indexer import TokenIndexer

# ------------------------- IMPLEMENTATION -----------------------------------

class Dataset(data.Dataset):
    def __init__(
        self, 
        data: Tuple[Iterable, Iterable, Iterable],
        data_type: str, 
        token_indexer: TokenIndexer
    ) -> None:

        if data_type not in ["train", "test", "val"]:
            raise ValueError("Data type must be one of 'train', 'test' or 'val'!")

        self.data_type = data_type
        self.token_indexer = token_indexer
        self.contexts, self.targets, self.emotions = data
    
    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict:
        """returns one data pair"""
        item = {}

        # Map emotion label to token
        item['emotion'] = self.token_indexer.emo_map[self.emotions[idx]]

        dialogue = []
        dialogue_state = []

        # Create list of utterances tokenized and encoded to indices
        # using spaCY and Byte-Pair Encoding (BPE)
        # rtype -> [[token1, token2, ...], [token1, token2, ...], ...]
        encoded_context = self.token_indexer.encode_text(self.contexts[idx])
        
        for i, enc in enumerate(encoded_context):
            # Add SOS and EOS token to every utterance
            dialogue += ([self.token_indexer.SOS_IDX] + 
                         enc + [self.token_indexer.EOS_IDX])
            ds = (self.token_indexer.DS_SPEAKER_IDX if i % 2 == 0 else
                  self.token_indexer.DS_LISTENER_IDX)
            # Add dialogue state to every token in utterance
            dialogue_state += [ds for _ in range(len(enc) + 2)] 

        # Tokenize and encode response utterance
        encoded_target = self.token_indexer.encode_text([self.targets[idx]])[0]
        # Add SOS and EOS token to target utterance
        target = ([self.token_indexer.SOS_IDX] + 
                    encoded_target + [self.token_indexer.EOS_IDX])
        # Determine response dialogue state
        ds = (self.token_indexer.DS_SPEAKER_IDX
              if len(encoded_context) % 2 == 0 
              else self.token_indexer.DS_LISTENER_IDX)

        if self.data_type == "test":
            item['target'] = target
            item['target_dialogue_state'] = [ds]
        
        else:
            dialogue += target
            dialogue_state += [ds for _ in range(len(target))]
        
        item["dialogue"] = dialogue
        item["dialogue_state"] = dialogue_state
        
        return item

def collate_batch(batch, padding_idx):
    """
    Pads each context and target sequence with zeroes and converts to tensor
    """

    def pad_and_convert_to_tensor(sequences, max_len, padding_idx):
        sequences = [seq + [padding_idx] * (max_len - len(seq)) 
                     for seq in sequences]
        return torch.LongTensor(sequences)
    
    dialogue = [item["dialogue"] for item in batch]
    dialogue_state = [item["dialogue_state"] for item in batch]
    target = [item.get("target", []) for item in batch]
    target_dialogue_state = [item.get("target_dialogue_state", []) for item in batch]
    emotion = [item["emotion"] for item in batch]

    max_len_dialog_seq = max([len(seq) for seq in dialogue])
    max_len_target_seq = max([len(seq) for seq in target])

    collated_batch = {
        "dialogue": pad_and_convert_to_tensor(
            dialogue, max_len_dialog_seq, padding_idx=padding_idx),
        "dialogue_state":  pad_and_convert_to_tensor(
            dialogue_state, max_len_dialog_seq, padding_idx=padding_idx),
        "target": pad_and_convert_to_tensor(
            target, max_len_target_seq, padding_idx=padding_idx),
        "target_dialogue_state": torch.LongTensor(target_dialogue_state),
        "emotion":  torch.LongTensor(emotion),
    }

    return collated_batch

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str, 
        batch_size: int,
        token_indexer: TokenIndexer,
        num_workers: int
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.token_indexer = token_indexer
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
            # Shuffle data to form train and validation sets
            data = list(zip(*self.load_data("train")))
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
            self.train_dataset = Dataset(train_data, "train", self.token_indexer,)
            self.val_dataset = Dataset(val_data, "val", self.token_indexer)
        if stage == "test":
            test_data = self.load_data("test")
            self.test_dataset = Dataset(test_data, "test", self.token_indexer)

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn= lambda x: collate_batch(x, self.token_indexer.PAD_IDX),
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn= lambda x: collate_batch(x, self.token_indexer.PAD_IDX),
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn= lambda x: collate_batch(x, self.token_indexer.PAD_IDX),
            num_workers=self.num_workers
        )

# -----------------------------------------------------------------------------