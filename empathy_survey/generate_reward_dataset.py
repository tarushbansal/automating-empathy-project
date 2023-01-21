# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse

# ------------------------- IMPLEMENTATION -----------------------------------

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    cli_args = parser.parse_args()

    working_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(cli_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    response_ratings = json.load(open(f"{working_dir}/results/response_ratings.json"))

    dialogues = []
    rewards = []
    for item in response_ratings.values():
        for response, ratings in item["ratings"].values():
            dialogues.append(item["context"] + [response])
            rewards.append(0.8 * ratings[0] + 0.1 * ratings[1] + 0.1 * ratings[2])
            
    with open(f"{output_dir}/dialogues.json", "w") as f:
        json.dump(dialogues, f)
    
    with open(f"{output_dir}/rewards.json", "w") as f:
        json.dump(rewards, f)
