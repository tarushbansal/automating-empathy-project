# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
from tqdm import tqdm

import torch

# User-defined Modules
from data_classes import RewardModelBatch
from data_loader import pad_seq_and_convert_to_tensor
from setup import get_model_supervisor_and_config

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
    reward_model = get_model_supervisor_and_config(
        pretrained_model_dir=cli_args.pretrained_reward_model_dir,
        reward_model=True
    )
    reward_model = reward_model.to(device)
    reward_model.eval()

    # Prepare data
    test_data = json.load(open(f"{model_dir}/test_data.json"))

    # Compute rewards
    rewards = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), cli_args.batch_size)):
            contexts = [reward_model.tokenizer.encode_text(item["context"])[0] 
                        for item in test_data[i:i+cli_args.batch_size]]
            max_len_context_seq = max([len(seq) for seq in contexts])
            contexts = pad_seq_and_convert_to_tensor(
                contexts,
                max_len_context_seq,
                pad_token=reward_model.tokenizer.PAD_IDX
            )
            contexts = contexts.to(device)
            targets = [reward_model.tokenizer.encode_text(item["output"])[0]
                       for item in test_data[i:i+cli_args.batch_size]]
            max_len_target_seq = max([len(seq) for seq in targets])
            targets = pad_seq_and_convert_to_tensor(
                targets,
                max_len_target_seq,
                pad_token=reward_model.tokenizer.PAD_IDX
            )
            targets = targets.to(device)
            batch = RewardModelBatch(
                contexts=contexts,
                context_mask=(contexts != reward_model.tokenizer.PAD_IDX),
                targets=targets,
                target_mask=(targets != reward_model.tokenizer.PAD_IDX),
                ratings=None
            )
            rewards.extend(reward_model.forward(batch).tolist())
    
    mean_reward = sum(rewards) / len(rewards)
    var_reward = sum([(reward - mean_reward) ** 2 for reward in rewards]) / len(rewards)
    print(f"Mean reward for dialogue model: {mean_reward}")
    print(f"Variance in reward for dialogue model: {var_reward}")
    with open(f"{model_dir}/rewards.json", "w") as f:
        json.dump({"mean": mean_reward, "var": var_reward, "rewards": 
                   {id: rewards[i] for i, id in enumerate([item["id"] for item in test_data])}}, f)
        print(f"All rewards saved at {model_dir}/rewards.json")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------