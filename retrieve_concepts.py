# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
import requests
import numpy as np
from tqdm import tqdm
from typing import List, Dict

# ------------------------- IMPLEMENTATION ------------------------------------


def retrieve_concepts(
    queries: List[str],
    emo_thresh: float,
    max_req_failures: int
) -> Dict[str, List[str]]:

    print(f"Retrieving concepts for {len(queries)} NRC_VAD words from ConceptNet5...")
    concepts = {}
    failed_reqs = 0
    for word in tqdm(queries):
        word_concepts = []
        endpoint = f"c/en/{word}?offset=0&limit=1000"
        while endpoint is not None:
            try:
                obj = requests.get(
                    f'http://api.conceptnet.io/{endpoint}').json()
            except requests.exceptions.RequestException:
                failed_reqs += 1
                print(f"ERROR: {failed_reqs} failed requests recorded!", end=" ")
                print(f"Maximum {max_req_failures} are allowed")
                if failed_reqs > max_req_failures:
                    print("ERROR: Exiting loop as failure threshold has been breached!")
                    return concepts
                continue
            for edge in obj["edges"]:
                concept_intensity = emotion_intensity(edge["end"]["label"])
                if concept_intensity < emo_thresh or edge["rel"]["label"] in ignore_relations:
                    continue
                if edge["surfaceText"] is not None:
                    word_concepts.append((concept_intensity, edge["surfaceText"]))
            endpoint = dict.get(dict.get(obj, "view", {}), "nextPage", None)
        if len(word_concepts) != 0:
            word_concepts = sorted(word_concepts, key=lambda x: x[0], reverse=True)
            concepts[word] = [concept[1] for concept in word_concepts]

    return concepts


def emotion_intensity(word: str) -> float:
    if word not in vad:
        return 0
    v, a, _ = vad[word]
    a /= 2
    return (np.linalg.norm(
        np.array([v, a]) - np.array([0.5, 0])) - 0.06467) / 0.607468


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrc_vad_fpath", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--emo_thresh", type=float, default=0.6)
    parser.add_argument("--max_req_failures", type=int, default=3)
    cli_args = parser.parse_args()

    ignore_relations = set(["Antonym", "ExternalURL", "NotDesires",
                            "NotHasProperty", "NotCapableOf", "dbpedia",
                            "DistinctFrom", "EtymologicallyDerivedFrom",
                            "EtymologicallyRelatedTo", "SymbolOf", "FormOf",
                            "AtLocation", "DerivedFrom", "SymbolOf",
                            "CreatedBy", "Synonym", "MadeOf"])

    with open(os.path.abspath(cli_args.nrc_vad_fpath)) as f:
        vad = json.load(f)

    if cli_args.output_dir is None:
        dirpath = os.getcwd()
        print("Output directory not specified. Setting to current working directory")
    else:
        dirpath = os.path.abspath(cli_args.output_dir)
        if not os.path.isdir(dirpath):
            raise FileNotFoundError("Specified output directory does not exist!")

    preretrieved = {}
    fpath = os.path.join(dirpath, "concepts_nrc_vad.json")
    if os.path.isfile(fpath):
        with open(fpath) as f:
            preretrieved = json.load(f)

    queries = [word for word in vad
               if emotion_intensity(word) >= cli_args.emo_thresh
               and word not in preretrieved]
    concepts = retrieve_concepts(
        queries,
        cli_args.emo_thresh,
        cli_args.max_req_failures
    )

    with open(fpath, "w") as f:
        json.dump({**preretrieved, **concepts}, f)
        print(f"Concepts successfully retrieved and saved at '{fpath}'")

# -----------------------------------------------------------------------------------
