# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import math
import argparse

import torch

# User-defined Modules
from model_supervisor import ModelSupervisor
from utils import load_val_ckpt_path, load_config
from base_classes import TokenizerBase

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

    # Initialise target dialogue state to listener (if supported)
    batch = {}
    batch["target_dialogue_state"] = None
    if tokenizer.supports_dialogue_states:
        batch["target_dialogue_state"] = torch.LongTensor([[tokenizer.DS_LISTENER_IDX]])

    # Run main interface loop
    context = []
    initialise_interface()
    batch["emotion"] = emotion_loop(tokenizer)

    while True:
        speaker_utterance = input(f"Speaker: ")
        if speaker_utterance.strip() == "<quit>":
            os.system("clear")
            break
        if speaker_utterance.strip() == "<clear>":
            context = []
            initialise_interface()
            batch["emotion"] = emotion_loop(tokenizer)
            continue
        
        context.append(speaker_utterance)
        enc_context, context_ds = tokenizer.encode_text(context, "context")
        batch["context"] = torch.LongTensor([enc_context])
        batch["context_dialogue_state"] = None
        if tokenizer.supports_dialogue_states:
            batch["context_dialogue_state"] = torch.LongTensor([context_ds])

        response, log_prob = model_supervisor.beam_search(batch)
        decoded_reponse = tokenizer.decode_to_text(response[0])
        ppl = 1 / math.exp(log_prob[0] / len(response[0]))
        print(f"Dialogue Model: {decoded_reponse} (Perplexity: {ppl:.2f})")
        print("")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------