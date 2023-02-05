# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
from tqdm import tqdm

import torch

# User-defined Modules
from data_classes import RewardModelBatch
from reward_model_supervisor import RewardModelSupervisor
from transformers import GPT2Tokenizer, GPT2Model
from utils.train_utils import load_ckpt_path

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_dialogue_model_dir", type=str, default=None, required=True)
    parser.add_argument("--pretrained_reward_model_dir", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    cli_args = parser.parse_args()

    return cli_args


def main():
    # Parse command line arguments
    cli_args = parse_args()

    model_dir = os.path.abspath(cli_args.pretrained_dialogue_model_dir)
    if not os.path.isdir(model_dir):
        raise ValueError(f"Specified model directory does not exist!")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise model and tokenizer
    reward_model = RewardModelSupervisor.load_from_checkpoint(
        load_ckpt_path(cli_args.pretrained_reward_model_dir),
        model=GPT2Model.from_pretrained("gpt2-large")).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Prepare data
    test_data = json.load(open(f"{model_dir}/test_data.json"))
    dialogues = [f" {tokenizer.eos_token} ".join(item["context"] + [item["prediction"]])
                 + f" {tokenizer.eos_token}" for item in test_data]

    # Compute rewards
    rewards = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dialogues), cli_args.batch_size)):
            input = tokenizer(
                dialogues[i:i+cli_args.batch_size], 
                return_tensors="pt",
                padding=True
            )
            batch = RewardModelBatch(
                dialogues=input.input_ids.to(device), 
                rewards=None,
                mask=input.attention_mask.to(device)
            )
            rewards.extend(reward_model.forward(batch).tolist())
    
    mean_reward = sum(rewards)/len(rewards)
    print(f"Mean reward for dialogue model: {mean_reward}")
    with open(f"{model_dir}/rewards.json", "w") as f:
        json.dump({"mean": mean_reward, "rewards": 
                   [{i: reward} for i, reward in enumerate(rewards)]}, f)
        print(f"All rewards saved at {model_dir}/rewards.json")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------