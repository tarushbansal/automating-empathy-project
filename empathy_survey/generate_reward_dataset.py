# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
from collections import OrderedDict

# ------------------------- IMPLEMENTATION -----------------------------------

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/tb662/rds/hpc-work/automating-empathy-project/datasets")
    cli_args = parser.parse_args()

    working_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(cli_args.output_dir)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError("Specified output directory does not exist!")
    os.makedirs(f"{output_dir}/reward_dataset/train", exist_ok=True)

    data = list(json.load(open(f"{working_dir}/results/response_ratings.json")).values())

    contexts, responses, ratings = [], [], []
    for item in data:
        contexts.append(item["context"])
        ordered_res = OrderedDict(item["responses"].items())
        responses.append(list(ordered_res.values()))
        pairwise_ratings = []
        for rating in item["ratings"]:
            pairwise_ratings.append((
                list(ordered_res.keys()).index(rating["A"]), 
                list(ordered_res.keys()).index(rating["B"]), 
                rating["ratings"][0]
            ))
        ratings.append(pairwise_ratings)
    with open(f"{output_dir}/reward_dataset/train/contexts.json", "w") as f:
        json.dump(contexts, f)
    with open(f"{output_dir}/reward_dataset/train/responses.json", "w") as f:
        json.dump(responses, f)
    with open(f"{output_dir}/reward_dataset/train/ratings.json", "w") as f:
        json.dump(ratings, f)

    print(f"Pairwise ratings for responses to {len(data)} unique prompt(s) saved at {output_dir}/reward_dataset/train")
