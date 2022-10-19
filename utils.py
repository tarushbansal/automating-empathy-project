# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- IMPLEMENTATION -----------------------------------

def load_val_ckpt_path(trained_model_dir: str) -> Optional[str]:
    ckpt_path = None
    if trained_model_dir is not None:
        if os.path.isdir(trained_model_dir):
            load_ckpt_dir = os.path.join(trained_model_dir, "checkpoints")
            if os.path.isdir(load_ckpt_dir):
                files = os.listdir(load_ckpt_dir)
                for fname in files:
                    if fname.startswith("train_loss") and fname.endswith(".ckpt"):
                        ckpt_path = os.path.join(load_ckpt_dir, fname)
            if ckpt_path is None:
                raise ValueError(
            "No checkpoint files found in the specified model directory!")
        else:
            raise ValueError("Specified model directory does not exist!")
    
    return ckpt_path

def beam_search(
    model: nn.Module,
    batch: Dict,
    beam_width: int,
    min_seq_len: int,
    max_seq_len: int,
    sos_token: int,
    eos_token: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # Set model in evaluation mode (Important to turn off dropout layers!!)
    model.eval()

    # Objective is to determine the most probable output sequence from the model
    N, seq_len = batch["dialogue"].size(dim=0), 1
    batch_beam = torch.empty(N, beam_width, seq_len, dtype=torch.long).fill_(sos_token)
    batch_beam_prob = torch.ones(N, beam_width)
    eos_detected = torch.zeros(N, beam_width, dtype=torch.bool)

    # Loop until EOS token detected for all beams in batch
    while (seq_len < min_seq_len or not torch.all(eos_detected)) and (seq_len < max_seq_len):

        sequences = torch.empty(N, beam_width ** 2, seq_len + 1, dtype=torch.long)
        prob_sequences = torch.empty(N, beam_width ** 2)

        # Determine all possible output sequences and probabilities for current beam 
        for i in range(beam_width):
            input_seq = torch.cat((batch["dialogue"], batch_beam[:, i, :]), dim=1)
            dialogue_state = torch.cat(
                (batch["dialogue_state"], batch["target_dialogue_state"].expand(-1, seq_len)), dim=1)
            out = model(
                input_seq=input_seq,
                dialogue_state=dialogue_state,
                emotion_label=batch["emotion"]
            )
            conditional_p, top_responses = torch.topk(
                F.softmax(out[:, -1, :], dim=-1), beam_width, dim=-1)
            conditional_p = conditional_p.masked_fill(eos_detected[:, i:i+1], 1)
            for j in range(beam_width):
                prob_sequences[:, i * beam_width + j] = (
                    batch_beam_prob[:, i] * conditional_p[:, j])
                sequences[:, i * beam_width + j, :] = torch.cat(
                        (batch_beam[:, i, :], top_responses[:, j:j+1]), dim=-1)

        # Choose {beam_width} number of sequences with highest probabilities
        batch_beam_prob, top_indices = torch.topk(
            prob_sequences, beam_width, dim=-1)
        top_indices = top_indices.unsqueeze(2).expand(N, beam_width, seq_len + 1)
        batch_beam = torch.gather(sequences, 1, top_indices)

        # Check which beams in batch have reached the EOS token
        for i in range(N):
            for j in range(beam_width):
                if int(batch_beam[i, j, -1]) == eos_token:
                    eos_detected[i, j] = True
        
        # Increment target sequence length and check if maximum limit has been exceeded
        seq_len += 1

    # Return most probable sequence for each beam in batch
    return batch_beam[:, 0, :], batch_beam_prob[:, 0]

#----------------------------------------------------------------------------