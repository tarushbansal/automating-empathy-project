# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse

import pytorch_lightning as pl

# User-defined Modules
from data_loader import DataModule
from data_classes import GenerationConfig
from setup import get_model_supervisor_and_config

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metric_n_grams", type=int, default=4)
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

    if not os.path.isdir(cli_args.dataset_dir):
        raise ValueError(f"Specified dataset directory does not exist!")

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
    model_supervisor, _ = get_model_supervisor_and_config(
        cli_args.model,
        cli_args.pretrained_model_dir
    )
    model_supervisor.metric_n_grams = cli_args.metric_n_grams
    if cli_args.pretrained_model_dir is None:
        if cli_args.output_dir is None:
            raise ValueError(
                "Output directory must be specified for saving test results for new models!")
        os.makedirs(cli_args.output_dir, exist_ok=True)
        model_supervisor.test_output_dir = cli_args.output_dir
    else:
        model_supervisor.test_output_dir = cli_args.pretrained_model_dir
    model_supervisor.generation_config = generation_config

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             tokenizer=model_supervisor.tokenizer,
                             model_has_encoder=model_supervisor.model.has_encoder,
                             num_workers=max(1, os.cpu_count() // 4))

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto", 
        devices=1,
        logger=None
    )

    # Test the model
    trainer.test(
        model_supervisor, 
        data_module
    )

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------