# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import sys
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# User-defined Modules
from model_supervisor import ModelSupervisor
from tokenizer import Tokenizer
from utils import load_val_ckpt_path, beam_search

# ------------------------- IMPLEMENTATION -----------------------------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_dir", type=str, default=None, required=True)
    parser.add_argument("--trained_model_dir", type=str, default=None, required=True)
    parser.add_argument("--predict_beam_width", type=int, default=2)
    cli_args = parser.parse_args()

    # Load checkpoint file path from trained model directory
    ckpt_path = load_val_ckpt_path(cli_args.trained_model_dir)

    # Initialise token indexer
    tokenizer = Tokenizer(cli_args.vocab_dir)

    # Load model supervisor from checkpoint file
    model_supervisor = ModelSupervisor.load_from_checkpoint(
        ckpt_path, tokenizer=tokenizer)

    # Generate response from model using stdin
    while True:
        emotion_label = input("Emotion Label: ")
        if emotion_label in tokenizer.emo_map:
            break
        print("Emotion label not supported! Try again")

    batch = {}

    dialogue = [[]]
    dialogue_state = [[]]
    batch["emotion"] = torch.LongTensor([tokenizer.emo_map[emotion_label]])
    batch["target_dialogue_state"] = torch.LongTensor([[tokenizer.DS_LISTENER_IDX]])

    while True:
        speaker_utterance = input(f"[{emotion_label}] Speaker: ")
        if speaker_utterance.strip() == "<quit>":
            break
        speaker_utterance = tokenizer.encode_text([speaker_utterance])[0]
        dialogue[0].extend(speaker_utterance)
        dialogue_state[0].extend([
            tokenizer.DS_SPEAKER_IDX for _ in range(len(speaker_utterance))])

        batch["dialogue"] = torch.LongTensor(dialogue)
        batch["dialogue_state"] = torch.LongTensor(dialogue_state)
        
        response, prob = beam_search(
            model=model_supervisor.model,
            batch=batch,
            beam_width=cli_args.predict_beam_width,
            min_seq_len=3,
            max_seq_len=10,
            sos_token=tokenizer.SOS_IDX,
            eos_token=tokenizer.EOS_IDX
        )

        decoded_reponse = "".join(tokenizer.decode_to_text(list(response[0])))
        print(f"[{emotion_label}] Empathetic model: {decoded_reponse} [Response probability: {prob[0]:.3f}]")
        print("")

        dialogue[0].extend(response[0])
        dialogue_state[0].extend([
            tokenizer.DS_LISTENER_IDX for _ in range(len(response[0]))])




if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------