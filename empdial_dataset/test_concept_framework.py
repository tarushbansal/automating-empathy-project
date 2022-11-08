import json
import numpy as np
from tqdm import tqdm

import nltk
from pattern.text.en import singularize
from transformers import AutoTokenizer

count = 0
total = 0
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
contexts = np.load("train/contexts.npy", allow_pickle=True)
concepts = json.load(open("../concepts.json"))
vad = json.load(open("../vad.json"))

stopwords = set(nltk.corpus.stopwords.words('english'))
ignore_relations = set(["Antonym", "ExternalURL", "NotDesires", 
                                "NotHasProperty", "NotCapableOf", "dbpedia", 
                                "DistinctFrom", "EtymologicallyDerivedFrom", 
                                "EtymologicallyRelatedTo", "SymbolOf", "FormOf", 
                                "AtLocation", "DerivedFrom", "SymbolOf",
                                "CreatedBy", "Synonym", "MadeOf"])

def emotion_intensity(word):
    if word not in vad:
        return 0
    v, a, _ = vad[word]
    a /= 2
    return (np.linalg.norm(
        np.array([v, a]) - np.array([0.5, 0])) - 0.06467) / 0.607468

for context in tqdm(contexts):
    for utt in context:
        tokens = tokenizer.tokenize(utt)
        for token in tokens:
            total += 1
            if token[0] == "▁":
                token = token[1:]
            token = singularize(token.lower())
            if (token not in stopwords) and (token in concepts):
                concept_found = False
                for concept in concepts[token]:
                    if ((concept[1] not in ignore_relations) and 
                        (emotion_intensity(concept[0]) >= 0.6)):
                        concept_found = True
                if concept_found:
                    count += 1

print(f"Total percentage of conceptualised tokens: {count * 100 / total}%%")