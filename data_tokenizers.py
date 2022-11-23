# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
import os
import json
import numpy as np
from typing import List, Union, Tuple, Optional, Dict

import nltk
from pattern.text.en import singularize
from transformers import AutoTokenizer

# User-Defined Modules
from base_classes import TokenizerBase

# ------------------------- IMPLEMENTATION ----------------------------------------

# class NltkTokenizer(TokenizerBase):
#     def __init__(self, vocab_fpath: str) -> None:
#         super().__init__()

#         with open(vocab_fpath) as f:
#             self.vocab = json.load(f)

#         self.decoder = {v : k for k, v in self.vocab.items()}

#         # Special tokens (Start / End of continguous dialogue sentence, Unknown)
#         self.UNK_IDX = self.vocab["[UNK]"]
#         self.SOS_IDX = self.vocab["[SOS]"]
#         self.EOS_IDX = self.vocab["[EOS]"]

#         # Size of vocabulary and special tokens
#         self.vocab_size = len(self.vocab)

#         self.word_pairs = {
#             "it's": "it is", "don't": "do not", "doesn't": "does not",
#             "didn't": "did not", "you'd": "you would", "you're": "you are",
#             "you'll": "you will", "i'm": "i am", "they're": "they are",
#             "that's": "that is", "what's": "what is", "couldn't": "could not",
#             "i've": "i have", "we've": "we have", "can't": "cannot",
#             "i'd": "i would", "aren't": "are not", "isn't": "is not",
#             "wasn't": "was not", "weren't": "were not", "won't": "will not",
#             "there's": "there is", "there're": "there are"
#         }

#     def encode_text(self, sequences: List[List[str]]) -> List[List[int]]:
#         encoded_sequences = []
#         for seq in sequences:
#             encoded_sequence = []
#             seq = seq.lower()
#             for k, v in self.word_pairs.items():
#                 seq = seq.replace(k, v)
#             tokens = word_tokenize(seq)
#             for token in tokens:
#                 token_idx = self.vocab.get(token, self.UNK_IDX)
#                 encoded_sequence.append(token_idx)
#             encoded_sequences.append(encoded_sequence)
#         return encoded_sequences

#     def decode_to_text(self, sequence: List[int]) -> str:
#         decoded_text = ""
#         for token_idx in sequence:
#             if token_idx == self.SOS_IDX or token_idx == self.EOS_IDX:
#                 continue
#             if token_idx == self.PAD_IDX:
#                 break
#             decoded_text += self.decoder[token_idx]
#         return decoded_text


class GODELTokenizer(TokenizerBase):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/GODEL-v1_1-base-seq2seq"
        )
        self.tokenizer.add_special_tokens({
            "bos_token": "<SOS>"
        })
        self.SOS_IDX = self.tokenizer.bos_token_id
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        instruction = ("Instruction: given a dialog context, "
                       + "you need to response empathically.")
        self.prefix = self.tokenizer(
            f"{instruction} [CONTEXT] ",
            add_special_tokens=False)["input_ids"]

    @property
    def supports_dialogue_states(self) -> bool:
        return False

    def encode_text(
        self,
        text: Union[str, List[str]],
        text_type: str
    ) -> Tuple[Optional[List[int]]]:

        if text_type == "context":
            dialog_history = f' EOS '.join(text)
            token_ids = self.tokenizer(
                dialog_history,
                add_special_tokens=False)["input_ids"]
            token_ids = self.prefix + token_ids
        elif text_type == "target":
            token_ids = self.tokenizer(
                f"{self.tokenizer.bos_token} {text} {self.tokenizer.eos_token}",
                add_special_tokens=False)["input_ids"]
        else:
            raise ValueError("Unsupported text type!")

        return token_ids, None, None

    def decode_to_text(self, sequence: List[int]) -> str:
        if len(sequence) == 0:
            return ""
        for i in range(len(sequence)):
            if sequence[i] == self.PAD_IDX:
                break
        decoded_text = self.tokenizer.decode(
            sequence[:i],
            skip_special_tokens=True
        )
        return decoded_text


class GPT2Tokenizer(TokenizerBase):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    @property
    def supports_dialogue_states(self) -> bool:
        return True

    def encode_text(
        self,
        text: Union[str, List[str]],
        text_type: str
    ) -> Tuple[List[int]]:

        if text_type == "context":
            input_ = f' {self.tokenizer.eos_token} '.join(text)
        elif text_type == "target":
            input_ = text
        else:
            raise ValueError("Unsupported text type!")

        token_ids = self.tokenizer(
            f"{input_} {self.tokenizer.eos_token}",
            add_special_tokens=False)["input_ids"]

        current = (self.DS_SPEAKER_IDX
                   if text == "context"
                   else self.DS_LISTENER_IDX)
        ds = []
        for token_id in range(len(token_ids)):
            ds.append(current)
            if token_id == self.EOS_IDX:
                current = (self.DS_LISTENER_IDX
                           if current == self.DS_SPEAKER_IDX
                           else self.DS_SPEAKER_IDX)

        return token_ids, ds, None

    def decode_to_text(self, sequence: List[int]) -> str:
        if len(sequence) == 0:
            return ""
        for i in range(len(sequence)):
            if sequence[i] == self.PAD_IDX:
                break
        decoded_text = self.tokenizer.decode(
            sequence[:i],
            skip_special_tokens=True
        )
        return decoded_text


class KnowledgeBridgedGODELTokenizer(TokenizerBase):
    def __init__(
        self,
        num_top_concepts: int,
        max_num_concepts: int
    ) -> None:

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/GODEL-v1_1-base-seq2seq"
        )
        self.tokenizer.add_special_tokens({
            "bos_token": "<SOS>"
        })
        self.SOS_IDX = self.tokenizer.bos_token_id
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
        self.external_knowledge = ExternalKnowlege(
            self.tokenizer,
            num_top_concepts,
            max_num_concepts
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        instruction = ("Instruction: given a dialog context, "
                       + "you need to response empathically.")
        self.prefix = self.tokenizer(
            f"{instruction} [CONTEXT] ",
            add_special_tokens=False)["input_ids"]

    @property
    def supports_external_knowledge(self) -> bool:
        return True

    def encode_text(
        self,
        text: Union[str, List[str]],
        text_type: str
    ) -> Tuple[Union[List[int], Dict[str, List]]]:

        external_knowledge = None

        if text_type == "context":
            # Tokenize dialog history and retrieve related concepts and emo intensities
            dialog_history = f' EOS '.join(text)
            tokens = self.tokenizer.tokenize(dialog_history)
            token_ids = self.prefix + self.tokenizer.convert_tokens_to_ids(tokens)
            # tokens = [token[1:] if token[0] == "▁" else token for token in tokens]
            external_knowledge = self.external_knowledge.retrieve(dialog_history.split(" "))

        elif text_type == "target":
            token_ids = self.tokenizer(
                f"{self.tokenizer.bos_token} {text} {self.tokenizer.eos_token}",
                add_special_tokens=False)["input_ids"]
        else:
            raise ValueError("Unsupported text type!")

        return (
            token_ids,
            None,
            external_knowledge
        )

    def decode_to_text(self, sequence: List[int]) -> str:
        if len(sequence) == 0:
            return ""
        for i in range(len(sequence)):
            if sequence[i] == self.PAD_IDX:
                break
        decoded_text = self.tokenizer.decode(
            sequence[:i],
            skip_special_tokens=True
        )
        return decoded_text


class KnowledgeBridgedGPT2Tokenizer(TokenizerBase):
    def __init__(
        self,
        num_top_concepts: int,
        max_num_concepts: int
    ) -> None:

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)
        self.external_knowledge = ExternalKnowlege(
            self.tokenizer,
            num_top_concepts,
            max_num_concepts
        )

    @property
    def supports_external_knowledge(self) -> bool:
        return True

    def encode_text(
        self,
        text: Union[str, List[str]],
        text_type: str
    ) -> Tuple[List[int]]:

        external_knowledge = None

        if text_type == "context":
            input_ = f' {self.tokenizer.eos_token} '.join(text)
        elif text_type == "target":
            input_ = text
        else:
            raise ValueError("Unsupported text type!")

        tokens = self.tokenizer.tokenize(f"{input_} {self.tokenizer.eos_token}")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # tokens = [token[1:] if token[0] == "Ġ" else token for token in tokens]
        external_knowledge = self.external_knowledge.retrieve(
            f"{input_} {self.tokenizer.eos_token}".split(" "))

        return token_ids, None, external_knowledge

    def decode_to_text(self, sequence: List[int]) -> str:
        if len(sequence) == 0:
            return ""
        for i in range(len(sequence)):
            if sequence[i] == self.PAD_IDX:
                break
        decoded_text = self.tokenizer.decode(
            sequence[:i],
            skip_special_tokens=True
        )
        return decoded_text


class ExternalKnowlege:
    def __init__(
        self,
        tokenizer: TokenizerBase,
        num_top_concepts: int,
        max_num_concepts: int,
    ) -> None:

        self.tokenizer = tokenizer
        self.num_top_concepts = num_top_concepts
        self.max_num_concepts = max_num_concepts
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.ignore_relations = set(["Antonym", "ExternalURL", "NotDesires",
                                     "NotHasProperty", "NotCapableOf", "dbpedia",
                                     "DistinctFrom", "EtymologicallyDerivedFrom",
                                     "EtymologicallyRelatedTo", "SymbolOf", "FormOf",
                                     "AtLocation", "DerivedFrom", "SymbolOf",
                                     "CreatedBy", "Synonym", "MadeOf"])
        with open("/home/tb662/rds/hpc-work/automating-empathy-project/vad.json") as f:
            self.vad = json.load(f)
        with open("/home/tb662/rds/hpc-work/automating-empathy-project/concepts.json") as f:
            self.concepts = json.load(f)

    def emotion_intensity(self, word):
        if word not in self.vad:
            return 0
        v, a, _ = self.vad[word]
        a /= 2
        return (np.linalg.norm(
            np.array([v, a]) - np.array([0.5, 0])) - 0.06467) / 0.607468

    def retrieve(
        self,
        tokens: List[str]
    ) -> Dict[str, Union[List[int], List[int]]]:

        concepts = []
        concept_pos = []
        concept_mask = []
        context_emo_intensity = []
        concept_emo_intensity = []

        for i, token in enumerate(tokens):
            token = singularize(token.lower())
            context_emo_intensity.append(self.emotion_intensity(token))
            if (token not in self.stopwords) and (token in self.concepts):
                num_concepts_added = 0
                for concept in self.concepts[token]:
                    if num_concepts_added == self.num_top_concepts:
                        break
                    if ((concept[1] not in self.ignore_relations) and
                            (self.emotion_intensity(concept[0]) >= 0.6)):
                        num_concepts_added += 1
                        concept_ids = self.tokenizer(
                            concept[0],
                            add_special_tokens=False)["input_ids"]
                        emo_intensity = self.emotion_intensity(concept[0])
                        for id in concept_ids:
                            concepts.append(id)
                            concept_emo_intensity.append(emo_intensity)
                            concept_pos.append(i)

        # Filter max_num_concepts with highest emotional intensities
        top_indices = sorted(
            range(len(concept_emo_intensity)),
            key=lambda i: concept_emo_intensity[i],
            reverse=True
        )
        for i in sorted(top_indices[self.max_num_concepts:], reverse=True):
            concepts.pop(i)
            concept_pos.pop(i)
            concept_emo_intensity.pop(i)

        # Create concept mask (upper right matrix in adjacency matrix)
        concept_mask = [[0] * len(concepts)] * len(tokens)
        for j, i in enumerate(concept_pos):
            concept_mask[i][j] = 1

        return {
            "concepts": concepts,
            "concept_mask": concept_mask,
            "context_emo_intensity": context_emo_intensity,
            "concept_emo_intensity": concept_emo_intensity
        }

# ---------------------------------------------------------------------------------
