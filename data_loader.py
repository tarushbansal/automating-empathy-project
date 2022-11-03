# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import random
import numpy as np
from typing import Tuple, Dict, Iterable

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

        context = []
        context_dialogue_state = []

        # Create list of utterances tokenized and encoded to indices using nltk
        # rtype -> [[token1, token2, ...], [token1, token2, ...], ...]
        encoded_context = self.tokenizer.encode_text(self.contexts[idx])
        
        for i, enc in enumerate(encoded_context):
            ds = (self.tokenizer.DS_SPEAKER_IDX 
                  if i % 2 == 0
                  else self.tokenizer.DS_LISTENER_IDX)
            context += enc + [self.tokenizer.EOS_IDX] 
            context_dialogue_state += [ds for _ in range(len(enc) + 1)] 
            
        item["context"] = context
        item["context_dialogue_state"] = context_dialogue_state

        # Tokenize and encode response utterance
        encoded_target = self.tokenizer.encode_text([self.targets[idx]])[0]
        target = [self.tokenizer.SOS_IDX] + encoded_target + [self.tokenizer.EOS_IDX]

        ds = (self.tokenizer.DS_SPEAKER_IDX
              if len(encoded_context) % 2 == 0 
              else self.tokenizer.DS_LISTENER_IDX)

        item['target'] = target
        item['target_dialogue_state'] = [ds for _ in range(len(target))]
        
        return item


def collate_batch(batch, padding_idx):
    """
    Pads each context and target sequence with zeroes and converts to tensor
    """

    def pad_and_convert_to_tensor(sequences, max_len, padding_idx):
        sequences = [seq + [padding_idx] * (max_len - len(seq)) 
                     for seq in sequences]
        return torch.LongTensor(sequences)
    
    context = [item["context"] for item in batch]
    context_dialogue_state = [item["context_dialogue_state"] for item in batch]
    target = [item["target"] for item in batch]
    target_dialogue_state = [item["target_dialogue_state"]for item in batch]
    emotion = [item["emotion"] for item in batch]

    max_len_context_seq = max([len(seq) for seq in context])
    max_len_target_seq = max([len(seq) for seq in target])

    collated_batch = {
        "context": pad_and_convert_to_tensor(
            context, max_len_context_seq, padding_idx=padding_idx),
        "context_dialogue_state":  pad_and_convert_to_tensor(
            context_dialogue_state, max_len_context_seq, padding_idx=padding_idx),
        "target": pad_and_convert_to_tensor(
            target, max_len_target_seq, padding_idx=padding_idx),
        "target_dialogue_state": pad_and_convert_to_tensor(
            target_dialogue_state, max_len_target_seq, padding_idx=padding_idx),
        "emotion":  torch.LongTensor(emotion),
    }

    return collated_batch


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
            collate_fn= lambda x: collate_batch(x, self.tokenizer.PAD_IDX),
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn= lambda x: collate_batch(x, self.tokenizer.PAD_IDX),
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn= lambda x: collate_batch(x, self.tokenizer.PAD_IDX),
            num_workers=self.num_workers
        )

# -----------------------------------------------------------------------------