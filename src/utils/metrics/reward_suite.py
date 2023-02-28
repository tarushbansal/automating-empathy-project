# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import sys
from tqdm import tqdm
from typing import Dict, List, Union

import torch

# User-Defined Modules
sys.path.append("/home/tb662/rds/hpc-work/automating-empathy-project/src")
from data_classes import RewardModelBatch
from reward_model_supervisor import RewardModelSupervisor

# ------------------------- IMPLEMENTATION -----------------------------------


def compute_reward_metrics(
    test_data: Dict[str, Union[List[str], str]],
    reward_model: RewardModelSupervisor,
    device: torch.device
) -> Dict[str, float]:

    print("Computing rewards...")
    with torch.no_grad():
        for item in tqdm(test_data):
            contexts = torch.tensor(
                [reward_model.tokenizer.encode_text(item["context"])[0]],
                dtype=torch.long,
                device=device
            )
            targets = torch.tensor(
                [reward_model.tokenizer.encode_text(item["target"])[0]],
                dtype=torch.long,
                device=device
            )
            batch = RewardModelBatch(
                contexts=contexts,
                context_mask=(contexts != reward_model.tokenizer.PAD_IDX),
                targets=targets,
                target_mask=(targets != reward_model.tokenizer.PAD_IDX),
                ratings=None
            )
            item["reward"] = float(reward_model.forward(batch)[0])
    
    rewards = [item["reward"] for item in test_data]
    mean_reward = sum(rewards) / len(rewards)
    var_reward = sum([(reward - mean_reward) ** 2 for reward in rewards]) / len(rewards)
    
    return {
        "mean_reward": mean_reward,
        "var_reward": var_reward
    }

# ------------------------------------------------------------------------------------