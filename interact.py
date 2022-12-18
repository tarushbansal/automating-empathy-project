# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import math
import argparse

import torch

# User-defined Modules
from base_classes import TokenizerBase
from model_supervisor import ModelSupervisor
from data_loader import collate_batch
from utils import load_val_ckpt_path, load_config

# ------------------------- IMPLEMENTATION -----------------------------------


def initialise_interface():
    os.system("clear")
    print("---- Welcome to this interface to interact with a dialogue model! -------")
    print("")
    print("Supply an emotion label for the conversation to get started and " +
          "type a speaker utterance when prompted for a multi-turn dialogue.")
    print("")
    print("Command keys:")
    print("---- <clear> - Clear conversation history and start a new conversation.")
    print("---- <quit>  - Exit interface")
    print("")


def emotion_loop(tokenizer: TokenizerBase) -> torch.Tensor:
    while True:
        emotion_label = input("Emotion Label: ")
        if emotion_label in tokenizer.emo_map:
            break
        print("Emotion label not supported! Try again\n")

    print(f"Emotion context set to '{emotion_label}'\n")

    return torch.LongTensor([tokenizer.emo_map[emotion_label]])


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, default=None, required=True)
    parser.add_argument("--pred_beam_width", type=int, default=1)
    parser.add_argument("--max_pred_seq_len", type=int, default=200)
    cli_args, _ = parser.parse_known_args()

    # Load checkpoint file path from trained model directory
    ckpt_path = load_val_ckpt_path(cli_args.pretrained_model_dir)

    # Initialise model and tokenizer from config file
    config = load_config(cli_args.pretrained_model_dir)
    tokenizer_cls = getattr(__import__("data_tokenizers"), config["tokenizer"]["cls"])
    tokenizer = tokenizer_cls(**config["tokenizer"]["kwargs"])

    model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
    model = model_cls(tokenizer=tokenizer, **config["model"]["kwargs"])
    model_supervisor = ModelSupervisor.load_from_checkpoint(
        ckpt_path,
        tokenizer=tokenizer,
        model=model,
        pred_beam_width=cli_args.pred_beam_width,
        max_pred_seq_len=cli_args.max_pred_seq_len,
    )

    # Run main interface loop
    context = []
    initialise_interface()
    emotion = emotion_loop(tokenizer)

    while True:
        speaker_utterance = input(f"Speaker: ")
        if speaker_utterance.strip() == "<quit>":
            os.system("clear")
            break
        if speaker_utterance.strip() == "<clear>":
            context = []
            initialise_interface()
            emotion = emotion_loop(tokenizer)
            continue

        context.append(speaker_utterance)
        enc_context, context_ds, external_knowledge = tokenizer.encode_text(context, "context")
        batch = collate_batch([{
            "context": enc_context,
            "context_dialogue_state": context_ds,
            "external_knowledge": external_knowledge,
            "target": [],
            "target_dialogue_state": [tokenizer.DS_LISTENER_IDX],
            "emotion": emotion
        }], tokenizer)

        response = model_supervisor.beam_search(batch)
        decoded_reponse = tokenizer.decode_to_text(response)
        print(f"Dialogue Model: {decoded_reponse}")
        print("")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
