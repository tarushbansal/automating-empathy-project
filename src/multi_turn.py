# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
from typing import Optional
import matplotlib.pyplot as plt

import torch

# User-defined Modules
from data_loader import collate_batch
from setup import get_model_supervisor_and_config
from data_classes import ModelRawData, RewardModelBatch, GenerationConfig

# ------------------------- IMPLEMENTATION -----------------------------------


def initialise_interface(id: int) -> Optional[str]:
    os.system("clear")
    print("------- Welcome to this interface to study a model's performance " + 
          "over multiple conversation turns and sessions! -------")
    print("")
    print("Supply an instruction to the model to get started.")
    print("NOTE that you can leave the instruction prompt empty in which case " + 
          "it will default to a preset value depending on the model in concern.")
    print("")
    print(f"----- Session ID {id} -----")
    print("")
    instruction = input("Instruction: ").strip()
    print("")

    return (None if instruction == "" else instruction) 


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dialogue_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrained_dialogue_model_dir", type=str, default=None)
    parser.add_argument("--pretrained_reward_model_dir", type=str, default=None)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_sessions", type=int, default=5)
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument('--sample', action='store_true', default=None)
    parser.add_argument('--no_sample', dest='sample', action='store_false')   
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--length_alpha", type=float, default=None)
    
    cli_args = parser.parse_args()

    return cli_args


def main():
    # Parse command line arguments
    cli_args = parse_args()

    # Sanity checks
    if cli_args.max_turns < 1 or cli_args.max_sessions < 1:
        raise ValueError("Turns or sessions cannot be less than 1!")

    if cli_args.pretrained_dialogue_model_dir is None:
        if cli_args.output_dir is None:
            raise ValueError(
                "Output directory must be specified for saving multi-turn results for new models!")
        output_dir = os.path.abspath(cli_args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.abspath(cli_args.pretrained_dialogue_model_dir)

    # Define generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=cli_args.max_new_tokens,
        beam_width=cli_args.beam_width,
        sample=cli_args.sample,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        top_k=cli_args.top_k,
        length_alpha=cli_args.length_alpha
    )

    # Set up dialogue and reward models
    dialogue_model, _ = get_model_supervisor_and_config(
        cli_args.dialogue_model,
        cli_args.pretrained_dialogue_model_dir
    )
    dialogue_model.generation_config = generation_config
    reward_model, _ = get_model_supervisor_and_config(
        pretrained_model_dir=cli_args.pretrained_reward_model_dir,
        reward_model=True
    )

    # Run main interface loop
    data = []
    context, rewards = [], []
    session_id = 0
    instruction = initialise_interface(session_id)

    while True:
        if len(context) == 2 * cli_args.max_turns:
            input(">>> Maximum conversation turns reached! Press any button to continue to the next session.")
            data.append({
                "session_id": session_id, 
                "instruction": instruction, 
                "dialogue":  context, 
                "rewards": rewards
            })
            session_id += 1
            if session_id > cli_args.max_sessions - 1:
                break
            context, rewards = [], []
            instruction = initialise_interface(session_id)
        
        speaker_utterance = input(f"Speaker: ").strip()
        context.append(speaker_utterance)
        enc_context = dialogue_model.tokenizer.encode_text(
            context,
            instruction
        )

        batch = collate_batch([
            ModelRawData(
                context=enc_context,
                raw_context=context,
                target=[],
                emotion=None
            )], 
            dialogue_model.tokenizer, 
            dialogue_model.model.has_encoder
        )

        response = dialogue_model.generate(batch.contexts, batch.context_mask)
        response = dialogue_model.tokenizer.decode(response)[0]

        enc_contexts = torch.LongTensor([reward_model.tokenizer.encode_text(context)[0]]) 
        enc_targets = torch.LongTensor([reward_model.tokenizer.encode_text(response)[0]])
        reward_batch = RewardModelBatch(
            contexts=enc_contexts,
            context_mask=torch.ones_like(enc_contexts),
            targets=enc_targets,
            target_mask=torch.ones_like(enc_targets),
            ratings=None
        )
        reward = float(reward_model.forward(reward_batch)[0])
        rewards.append(reward)

        print(f"Dialogue Model: {response} [Reward = {reward:.3f}]")

        context.append(response)
        print("")
    
    os.system("clear")
    print(">>> Maximum number of sessions reached!\n")
    with open(f"{output_dir}/multi_turn.json", "w") as f:
        json.dump(data, f)
    
    for item in data:
        id, rewards = item["session_id"], item["rewards"]
        plt.plot(range(len(rewards)), rewards, label=f"Session ID = {id}")
    plt.xlabel("Turn")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(f"{output_dir}/multi_turn.png")

    print(f"Plots, rewards and dialogue history from all sessions saved at '{output_dir}'")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------