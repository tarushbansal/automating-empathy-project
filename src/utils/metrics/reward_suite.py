# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import sys
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

import torch

# User-Defined Modules
sys.path.append("/home/tb662/rds/hpc-work/automating-empathy-project/src")
from utils.train import load_ckpt_path
from data_classes import RewardModelBatch
from reward_model_supervisor import RewardModelSupervisor
from data_loader import pad_to_tensor

# ------------------------- IMPLEMENTATION -----------------------------------


def compute_reward_metrics(
    reward_model_dir: Optional[str] = None,
    test_data: Optional[Dict] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    
    if reward_model_dir is None:
        print("USER WARNING: No reward model directory supplied!")
        return {}
    if test_data is None:
        print("USER WARNING: No test data supplied!")
        return {}
    if device is None:
        raise ValueError("Device must be specified to load reward model!")
    
    print(f"Loading reward model from '{reward_model_dir}'")
    reward_model = RewardModelSupervisor.load_from_checkpoint(
        load_ckpt_path(reward_model_dir),
        strict=False
    )
    reward_model.to(device)
    reward_model.eval()

    print("Computing rewards...")
    with torch.no_grad():
        for item in tqdm(test_data):
            contexts, context_mask = pad_to_tensor(
                [reward_model.tokenizer.encode_text(item["context"])[0],
                 reward_model.tokenizer.encode_text(item["context"])[0]], 
                reward_model.tokenizer.PAD_IDX
            )
            targets, target_mask = pad_to_tensor(
                [reward_model.tokenizer.encode_text(item["target"])[0],
                 reward_model.tokenizer.encode_text(item["output"])[0]], 
                 reward_model.tokenizer.PAD_IDX
            )
            batch = RewardModelBatch(
                contexts=contexts.to(device),
                context_mask=context_mask.to(device),
                targets=targets.to(device),
                target_mask=target_mask.to(device),
                ratings=None
            )
            rewards = reward_model.forward(batch)
            item["reward_target"] = float(rewards[0])
            item["reward_output"] = float(rewards[1])
    
    target_rewards = np.array([item["reward_target"] for item in test_data])
    mean_reward_target = float(np.mean(target_rewards))
    var_reward_target = float(np.var(target_rewards))

    output_rewards = np.array([item["reward_output"] for item in test_data])
    mean_reward_output = float(np.mean(output_rewards))
    var_reward_output = float(np.var(output_rewards))
    
    return {
        "mean_reward_target": mean_reward_target,
        "var_reward_target": var_reward_target,
        "mean_reward_output": mean_reward_output,
        "var_reward_output": var_reward_output
    }

# ------------------------------------------------------------------------------------