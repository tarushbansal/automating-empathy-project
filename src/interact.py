# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse

# User-defined Modules
from dialogue_model_supervisor import DialogueModelSupervisor
from utils.train_utils import load_ckpt_path, load_config
from data_loader import collate_batch
from data_classes import ModelRawData, GenerationConfig

# ------------------------- IMPLEMENTATION -----------------------------------


def initialise_interface():
    os.system("clear")
    print("---- Welcome to this interface to interact with a dialogue model! -------")
    print("")
    print("Supply an instruction to the model to get started.")
    print("Note that if you leave the instruction prompt empty it defaults to " +
          "'Instruction: given a dialog context, you need to respond empathetically.'")
    print("")
    print("Command keys:")
    print("---- <new>  - Clear current conversation history and start a new session.")
    print("---- <quit> - Exit interface")
    print("")


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
    ckpt_path = load_ckpt_path(cli_args.pretrained_model_dir)

    # Initialise model and tokenizer from config file
    config = load_config(cli_args.pretrained_model_dir)
    tokenizer_cls = getattr(__import__("custom_tokenizers"), config["tokenizer"]["cls"])
    tokenizer = tokenizer_cls(**config["tokenizer"]["kwargs"])

    model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
    model = model_cls(tokenizer=tokenizer, **config["model"]["kwargs"])
    model.eval()
    model_supervisor = DialogueModelSupervisor.load_from_checkpoint(
        ckpt_path,
        strict=False,
        tokenizer=tokenizer,
        model=model,
        batch_size=1,
        generation_config=GenerationConfig(
            max_new_tokens=cli_args.max_new_tokens,
            beam_width=cli_args.beam_width,
            sample=cli_args.sample,
            temperature=cli_args.temperature,
            top_p=cli_args.top_p,
            top_k=cli_args.top_k,
        )
    )

    # Run main interface loop
    context = []
    initialise_interface()
    instruction = input("Instruction: ").strip()

    while True:
        speaker_utterance = input(f"Speaker: ").strip()
        if speaker_utterance == "<quit>":
            os.system("clear")
            break
        if speaker_utterance == "<new>":
            context = []
            initialise_interface()
            instruction = input("Instruction: ").strip()
            continue

        context.append(speaker_utterance)
        enc_context, concept_net_data = tokenizer.encode_text(
            context,
            None if instruction == "" else instruction
        )

        batch = collate_batch([
            ModelRawData(
                context=enc_context,
                raw_context=context,
                target=[],
                emotion=None,
                concept_net_data=concept_net_data
            )], tokenizer
        )

        response = model_supervisor.generate(batch)[0]
        decoded_reponse = tokenizer.decode_to_text(response)
        print(f"Dialogue Model: {decoded_reponse}")
        print("")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
