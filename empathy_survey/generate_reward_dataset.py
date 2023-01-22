# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse

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
    os.makedirs(f"{output_dir}/reward_dataset/val", exist_ok=True)

    responses = list(json.load(open(f"{working_dir}/results/response_ratings.json")).values())
    split_idx = int(0.9 * len(responses))
    data_split = {
        "train": responses[:split_idx],
        "val": responses[split_idx:]
    }

    for split, responses in data_split.items():
        dialogues = []
        rewards = []
        for item in responses:
            for response, ratings in item["ratings"].values():
                dialogues.append(item["context"] + [response])
                rewards.append(0.8 * ratings[0] + 0.1 * ratings[1] + 0.1 * ratings[2])
                
        with open(f"{output_dir}/reward_dataset/{split}/dialogues.json", "w") as f:
            json.dump(dialogues, f)
        
        with open(f"{output_dir}/reward_dataset/{split}/rewards.json", "w") as f:
            json.dump(rewards, f)
    
        print(f"{len(dialogues)} dialogues and rewards saved at {output_dir}/reward_dataset/{split}")
