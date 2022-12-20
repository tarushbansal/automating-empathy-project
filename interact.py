# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import math
import argparse

import torch

# User-defined Modules
from base_classes import TokenizerBase
from model_supervisor import ModelSupervisor
from data_classes import EncoderDecoderModelRawData, DecoderModelRawData
from utils import load_val_ckpt_path, load_config
from data_loader import collate_decoder_batch, collate_encoder_decoder_batch

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
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
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
        generation_kwargs={
        "max_new_tokens": cli_args.max_new_tokens,
        "beam_width": cli_args.beam_width,
        "sample": cli_args.sample,
        "temperature": cli_args.temperature,
        "top_p": cli_args.top_p,
        "top_k": cli_args.top_k,
        }
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
        enc_context, concept_net_data = tokenizer.encode_text(context)
        if model.has_encoder:
            batch = collate_encoder_decoder_batch([
                EncoderDecoderModelRawData(
                    context=enc_context,
                    target=[],
                    emotion=emotion,
                    concept_net_data=concept_net_data
                )
            ])
        else:
            batch = collate_decoder_batch([
                DecoderModelRawData(
                    dialogue=context,
                    gen_target=None
                )
            ])

        response = model_supervisor.generate(batch).tolist()[0]
        decoded_reponse = tokenizer.decode_to_text(response)
        print(f"Dialogue Model: {decoded_reponse}")
        print("")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
