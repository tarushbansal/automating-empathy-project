# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import argparse

import torch

# User-defined Modules
from model_supervisor import ModelSupervisor
from utils import load_val_ckpt_path, load_config

# ------------------------- IMPLEMENTATION -----------------------------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_dir", type=str, default=None, required=True)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--max_pred_seq_len", type=int, default=100)
    cli_args, _ = parser.parse_known_args()

    # Load checkpoint file path from trained model directory
    ckpt_path = load_val_ckpt_path(cli_args.trained_model_dir)

    # Initialise token indexer
    config = load_config(cli_args.trained_model_dir)
    tokenizer_cls = getattr(__import__("data_tokenizers"), config["tokenizer"]["cls"])
    tokenizer = tokenizer_cls(**config["tokenizer"]["kwargs"])

    # Load model supervisor from checkpoint file
    model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
    model = model_cls(**config["model"]["kwargs"])
    model_supervisor = ModelSupervisor.load_from_checkpoint(
        ckpt_path, 
        tokenizer=tokenizer, 
        model=model, 
        beam_width=cli_args.beam_width, 
        max_pred_seq_len=cli_args.max_pred_seq_len
    )

    # Generate response from model using stdin
    while True:
        emotion_label = input("Emotion Label: ")
        if emotion_label in tokenizer.emo_map:
            break
        print("Emotion label not supported! Try again")

    batch = {}

    context = [[]]
    context_dialogue_state = [[]]
    batch["emotion"] = torch.LongTensor([tokenizer.emo_map[emotion_label]])
    batch["target_dialogue_state"] = torch.LongTensor([[tokenizer.DS_LISTENER_IDX]])

    while True:
        speaker_utterance = input(f"[{emotion_label}] Speaker: ")
        if speaker_utterance.strip() == "<quit>":
            break
        speaker_utterance = (
            [tokenizer.SOS_IDX] + \
            tokenizer.encode_text([speaker_utterance])[0] + \
            [tokenizer.EOS_IDX]
        )
        context[0].extend(speaker_utterance)
        context_dialogue_state[0].extend([
            tokenizer.DS_SPEAKER_IDX for _ in range(len(speaker_utterance))])

        batch["context"] = torch.LongTensor(context)
        batch["context_dialogue_state"] = torch.LongTensor(context_dialogue_state)

        response, prob = model_supervisor.predict_step(batch)

        response = response[0].tolist()
        decoded_reponse = tokenizer.decode_to_text(
            response[1:response.index(tokenizer.EOS_IDX)])
        print(f"[{emotion_label}] Empathetic model: {decoded_reponse}")
        print(f"[{emotion_label}] Response probability: {prob[0]:.3f}")
        print("")

        context[0].extend(response)
        context_dialogue_state[0].extend([
            tokenizer.DS_LISTENER_IDX for _ in range(len(response))])

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------