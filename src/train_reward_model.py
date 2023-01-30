# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse
from typing import Optional, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# User-defined Modules
from reward_data_loader import RewardDataModule
from reward_model_supervisor import RewardModelSupervisor
from transformers import GPT2Tokenizer, GPT2Model

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--initial_lr", type=float, default=0.00001)

    cli_args = parser.parse_args()

    return cli_args


def get_model_checkpoints(ckpt_dir: str) -> Optional[List[ModelCheckpoint]]:
    if ckpt_dir is None:
        return None

    return [
        ModelCheckpoint(
            monitor="train_loss_epoch",
            dirpath=ckpt_dir,
            filename="{train_loss_epoch:.2f}-{epoch}",
            every_n_epochs=1
        ),
        ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath=ckpt_dir,
            filename="{val_loss_epoch:.2f}-{epoch}",
            every_n_epochs=1
        )
    ]


def main():
    # Parse command line arguments
    cli_args = parse_args()

    if not os.path.isdir(cli_args.dataset_dir):
        raise ValueError(f"Specified dataset directory does not exist!")

    # Set up tensorboard logger
    logger = TensorBoardLogger(
        save_dir=cli_args.output_dir,
        name="tensorboard_logs"
    )

    # Initialise model and tokenizer from config
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model = GPT2Model.from_pretrained("gpt2-large")
    model_supervisor = RewardModelSupervisor(
        model=model,
        initial_lr=cli_args.initial_lr
    )

    # Set up data module
    data_module = RewardDataModule(
        dataset_dir=cli_args.dataset_dir,
        batch_size=cli_args.batch_size,
        tokenizer=tokenizer,
        num_workers=max(1, os.cpu_count() // 4)
    )

    # Set up model checkpointing
    callbacks = get_model_checkpoints(f"{logger.log_dir}/checkpoints")

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=-1 if torch.cuda.is_available() else 1,
        num_nodes=cli_args.num_nodes,
        strategy="ddp_find_unused_parameters_false",
        max_epochs=cli_args.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    # Train the model
    trainer.fit(
        model_supervisor,
        data_module
    )


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
