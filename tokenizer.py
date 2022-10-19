# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
import os
import json
from typing import List

import torch
from nltk import word_tokenize

# ------------------------- IMPLEMENTATION ----------------------------------------

class Tokenizer:
    def __init__(self, dataset_dir: str) -> None:
        """
        Including word tokens, special tokens, dialogue states, and emotion indices
        """
        dataset_path = os.path.abspath(dataset_dir)
        with open(f"{dataset_path}/vocab.json") as f:
            self.vocab = json.load(f)
        
        self.decoder = {v : k for k, v in self.vocab.items()}

        # Special tokens (Start / End of continguous dialogue sentence, Padding)
        self.PAD_IDX = self.vocab["[PAD]"]
        self.UNK_IDX = self.vocab["[UNK]"]
        self.SOS_IDX = self.vocab["[SOS]"]
        self.EOS_IDX = self.vocab["[EOS]"]
        
        # Size of vocabulary and special tokens 
        self.vocab_size = len(self.vocab)
        
        # Dialog state indices
        self.DS_SPEAKER_IDX = 1
        self.DS_LISTENER_IDX = 2
        
        # Emotion label map
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 
            'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9,
            'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17, 
            'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31
        }

        self.word_pairs = {
            "it's": "it is", "don't": "do not", "doesn't": "does not", 
            "didn't": "did not", "you'd": "you would", "you're": "you are", 
            "you'll": "you will", "i'm": "i am", "they're": "they are", 
            "that's": "that is", "what's": "what is", "couldn't": "could not", 
            "i've": "i have", "we've": "we have", "can't": "cannot", 
            "i'd": "i would", "aren't": "are not", "isn't": "is not", 
            "wasn't": "was not", "weren't": "were not", "won't": "will not", 
            "there's": "there is", "there're": "there are"
        }

    def encode_text(self, sequences: List[List[str]]) -> List[List[int]]:
        encoded_sequences = []
        for seq in sequences:
            encoded_sequence = []
            seq = seq.lower()
            for k, v in self.word_pairs.items():
                seq = seq.replace(k, v)
            tokens = word_tokenize(seq)
            for token in tokens:
                token_idx = self.vocab.get(token, self.UNK_IDX)
                encoded_sequence.append(token_idx)
            encoded_sequences.append(encoded_sequence)
        return encoded_sequences

    def decode_to_text(self, seq: torch.Tensor) -> str:
        decoded_text = ""
        for token_idx in seq:
            decoded_text += self.decoder[int(token_idx)]
        return decoded_text

# ---------------------------------------------------------------------------------