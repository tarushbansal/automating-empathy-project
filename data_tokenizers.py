# ------------------------- IMPORT MODULES ----------------------------------------

# System Modules
import os
import json
import string
import numpy as np
from typing import List, Union, Tuple, Optional

import nltk
from pattern.text.en import singularize
from transformers import AutoTokenizer

# User-Defined Modules
from base_classes import TokenizerBase
from data_classes import ConceptNetRawData

# ------------------------- IMPLEMENTATION ----------------------------------------


class GODELTokenizer(TokenizerBase):
    def __init__(
        self,
        query_concept_net: bool = False,
        num_top_concepts: int = 5,
        max_num_concepts: int = 10
    ) -> None:

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq"
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
        self.query_concept_net = query_concept_net
        if self.query_concept_net:
            self.concept_net = QueryConceptNet(
                self.tokenizer,
                num_top_concepts,
                max_num_concepts
            )

    def encode_text(
        self,
        text: Union[str, List[str]]
    ) -> Tuple[Union[List[int], Optional[ConceptNetRawData]]]:

        external_knowledge = None
        if type(text) == list:
            dialog_history = f' EOS '.join(text)
            tokens = self.tokenizer.tokenize(dialog_history)
            token_ids = self.prefix + self.tokenizer.convert_tokens_to_ids(tokens)
            if self.query_concept_net:
                tokens = [token[1:] if token[0] == "▁" else token for token in tokens]
                external_knowledge = self.concept_net.query(tokens)

        else:
            token_ids = self.tokenizer(
                f"{self.tokenizer.bos_token} {text} {self.tokenizer.eos_token}",
                add_special_tokens=False)["input_ids"]

        return token_ids, external_knowledge


class GPT2Tokenizer(TokenizerBase):
    def __init__(self) -> None:

        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.EOS_IDX = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode_text(
        self,
        text: Union[str, List[str]]
    ) -> Tuple[Union[List[int], Optional[ConceptNetRawData]]]:
    
        if type(text) == list:
            input_ = f"{f' {self.tokenizer.eos_token} '.join(text)} {self.tokenizer.eos_token}"
        else:
            input_ = f"{text} {self.tokenizer.eos_token}"

        token_ids = self.tokenizer(
            input_, add_special_tokens=False)["input_ids"]

        return token_ids, None


class QueryConceptNet:
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
        knowledge_dir = "/home/tb662/rds/hpc-work/automating-empathy-project/knowledge_data"
        with open(f"{knowledge_dir}/vad.json") as f:
            self.vad = json.load(f)
        with open(f"{knowledge_dir}/concepts.json") as f:
            self.concepts = json.load(f)

    def emotion_intensity(self, word):
        if word not in self.vad:
            return 0
        v, a, _ = self.vad[word]
        a /= 2
        return (np.linalg.norm(
            np.array([v, a]) - np.array([0.5, 0])) - 0.06467) / 0.607468

    def query(
        self,
        tokens: List[str]
    ) -> ConceptNetRawData:

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
        concept_mask = [[False] * len(concepts)] * len(tokens)
        for j, i in enumerate(concept_pos):
            concept_mask[i][j] = 1

        return ConceptNetRawData(
            concepts=concepts,
            concept_mask=concept_mask,
            context_emo_intensity=context_emo_intensity,
            concept_emo_intensity=concept_emo_intensity
        )


class QueryConceptNetTextBased:
    def __init__(self, num_top_concepts: int, max_num_concepts: int) -> None:
        self.num_top_concepts = num_top_concepts
        self.max_num_concepts = max_num_concepts
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        knowledge_dir = "/home/tb662/rds/hpc-work/automating-empathy-project/knowledge_data"
        with open(f"{knowledge_dir}/vad.json") as f:
            self.vad = json.load(f)
        with open(f"{knowledge_dir}/concepts_nrc_vad.json") as f:
            self.concepts = json.load(f)

    def retrieve(
        self,
        words: List[str]
    ) -> str:

        # Retrive num_top_concepts for all relevant words
        concepts = []
        for word in words:
            word = singularize(word.lower()).translate(str.maketrans('', '', string.punctuation))
            if (word not in self.stopwords) and (word in self.concepts):
                concepts.extend([(concept["intensity"], concept["text"])
                                 for concept in self.concepts[word][:self.num_top_concepts]])

        # Sort by emotional intensity and retrieve top max_num_concepts
        concepts = sorted(concepts, key=lambda x: x[0])
        concepts = [concept[1] for concept in concepts[:self.max_num_concepts]]

        return " ".join(concepts)

# ---------------------------------------------------------------------------------
