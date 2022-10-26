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
    parser.add_argument("--pretrained_model_dir", type=str, default=None, required=True)
    parser.add_argument("--pred_beam_width", type=int, default=1)
    parser.add_argument("--max_pred_seq_len", type=int, default=1000)
    parser.add_argument("--bleu_n_grams", type=int, default=4)
    cli_args, _ = parser.parse_known_args()

    # Load checkpoint file path from trained model directory
    ckpt_path = load_val_ckpt_path(cli_args.pretrained_model_dir)

    # Initialise model and tokenizer from config file
    config = load_config(cli_args.pretrained_model_dir)
    tokenizer_cls = getattr(__import__("data_tokenizers"), config["tokenizer"]["cls"])
    tokenizer = tokenizer_cls(**config["tokenizer"]["kwargs"])

    model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
    model = model_cls(**config["model"]["kwargs"])
    model_supervisor = ModelSupervisor.load_from_checkpoint(
        ckpt_path, 
        tokenizer=tokenizer, 
        model=model, 
        pred_beam_width=cli_args.pred_beam_width,
        max_pred_seq_len=cli_args.max_pred_seq_len,
        bleu_n_grams=cli_args.bleu_n_grams
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
    )

    # Test the model
    test_metrics = trainer.test(
        model_supervisor, 
        data_module, 
        ckpt_path=load_val_ckpt_path(cli_args.pretrained_model_dir)    
    )

    print(test_metrics)

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------