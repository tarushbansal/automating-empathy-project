# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse

import torch
import pytorch_lightning as pl

# User-defined Modules
from data_loader import DataModule
from model_supervisor import ModelSupervisor
from utils import load_val_ckpt_path, load_config

# ------------------------- IMPLEMENTATION -----------------------------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pred_beam_width", type=int, default=1)
    parser.add_argument("--max_pred_seq_len", type=int, default=200)
    parser.add_argument("--pred_n_grams", type=int, default=4)
    cli_args, _ = parser.parse_known_args()

    if not os.path.isdir(cli_args.dataset_dir):
        raise ValueError(f"Specified dataset directory does not exist!")

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
        batch_size=cli_args.batch_size,
        test_output_dir=os.path.abspath(cli_args.pretrained_model_dir), 
        pred_beam_width=cli_args.pred_beam_width,
        max_pred_seq_len=cli_args.max_pred_seq_len,
        pred_n_grams=cli_args.pred_n_grams
    )

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             tokenizer=tokenizer,
                             num_workers=max(1, os.cpu_count() // 4))

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto", 
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp_find_unused_parameters_false",
        logger=None
    )

    # Test the model
    trainer.test(
        model_supervisor, 
        data_module, 
        ckpt_path=load_val_ckpt_path(cli_args.pretrained_model_dir)    
    )

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------