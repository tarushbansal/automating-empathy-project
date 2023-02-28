# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse
from typing import Optional

# User-defined Modules
from data_loader import collate_batch
from setup import get_model_supervisor_and_config
from data_classes import ModelRawData, GenerationConfig

# ------------------------- IMPLEMENTATION -----------------------------------


def initialise_interface() -> Optional[str]:
    os.system("clear")
    print("------- Welcome to this interface to interact with a dialogue model! -------")
    print("")
    print("Supply an instruction to the model to get started.")
    print("NOTE that you can leave the instruction prompt empty in which case " + 
          "it will default to a preset value depending on the model in concern.")
    print("")
    print("Command keys:")
    print("---- <new>  - Clear current conversation history and start a new session.")
    print("---- <quit> - Exit interface")
    print("")
    instruction = input("Instruction: ").strip()
    print("")

    return (None if instruction == "" else instruction)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
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

    # Set up dialogue model and configuration
    model_supervisor = get_model_supervisor_and_config(
        model=cli_args.model,
        pretrained_model_dir=cli_args.pretrained_model_dir,
        kwargs={"generation_config": generation_config}
    )
    tokenizer = model_supervisor.tokenizer

    # Run main interface loop
    context = []
    instruction = initialise_interface()

    while True:
        speaker_utterance = input(f"Speaker: ").strip()
        if speaker_utterance == "<quit>":
            os.system("clear")
            break
        if speaker_utterance == "<new>":
            context = []
            instruction = initialise_interface()
            continue

        context.append(speaker_utterance)
        enc_context, concept_net_data = tokenizer.encode_text(
            context,
            instruction
        )

        batch = collate_batch([
            ModelRawData(
                context=enc_context,
                raw_context=context,
                target=[],
                concept_net_data=concept_net_data
            )], 
            tokenizer, 
            model_supervisor.model.has_encoder
        )

        response = model_supervisor.generate(batch.contexts, batch.context_mask)
        decoded_reponse = tokenizer.decode(response)[0]
        print(f"Dialogue Model: {decoded_reponse}")

        context.append(decoded_reponse)
        print("")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
