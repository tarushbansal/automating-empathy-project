# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
import json
from typing import List

from nltk import word_tokenize
from transformers import AutoTokenizer

# User-Defined Modules
from base_classes import TokenizerBase

# ------------------------- IMPLEMENTATION ----------------------------------------

class NltkTokenizer(TokenizerBase):
    def __init__(self, vocab_fpath: str) -> None:
        super().__init__()

        with open(vocab_fpath) as f:
            self.vocab = json.load(f)
        
        self.decoder = {v : k for k, v in self.vocab.items()}

        # Special tokens (Start / End of continguous dialogue sentence, Padding)
        self.PAD_IDX = self.vocab["[PAD]"]
        self.UNK_IDX = self.vocab["[UNK]"]
        self.SOS_IDX = self.vocab["[SOS]"]
        self.EOS_IDX = self.vocab["[EOS]"]
        
        # Size of vocabulary and special tokens 
        self.vocab_size = len(self.vocab)

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

    def decode_to_text(self, sequence: List[int]) -> str:
        decoded_text = ""
        for token_idx in sequence:
            decoded_text += self.decoder[token_idx]
        return decoded_text


class HuggingFaceAutoTokenizer(TokenizerBase):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens_dict = {
            'bos_token': '<BOS>', 
            'eos_token': '<EOS>', 
            'pad_token': '<PAD>'
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.PAD_IDX = self.tokenizer.pad_token_id
        self.SOS_IDX = self.tokenizer.bos_token_id
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
    
    def encode_text(self, sequences: List[List[str]]) -> List[List[int]]:
        token_ids = []
        for seq in sequences:
            token_ids.append(self.tokenizer(seq)["input_ids"])
        return token_ids
    
    def decode_to_text(self, sequence: List[int]) -> str:
        decoded_text = self.tokenizer.decode(
            sequence, 
            skip_special_tokens=True
        )
        return decoded_text        

# ---------------------------------------------------------------------------------