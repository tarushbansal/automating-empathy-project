# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# User-defined Modules
from data_loader import DataModule
from setup import get_model_supervisor_and_config
from utils.train_utils import get_model_checkpoints, SaveConfigCallback

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=0.00001)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
    parser.add_argument("--few_shot_training", action="store_true")

    cli_args = parser.parse_args()

    return cli_args


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

    # Set up dialogue model and configuration
    model_supervisor, config = get_model_supervisor_and_config(
        cli_args.model,
        cli_args.pretrained_model_dir,
        cli_args.initial_lr
    )

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             tokenizer=model_supervisor.tokenizer,
                             num_workers=max(1, os.cpu_count() // 4),
                             model_has_encoder=model_supervisor.model.has_encoder,
                             few_shot_training=cli_args.few_shot_training)

    # Set up model checkpointing
    ckpt_dir = f"{logger.log_dir}/checkpoints"
    checkpoint_callback = get_model_checkpoints(ckpt_dir)
    callbacks = []
    if checkpoint_callback is not None:
        callbacks.extend(checkpoint_callback)
    callbacks.append(SaveConfigCallback(config=config))

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

    # Train / Validate the model
    trainer.fit(
        model_supervisor,
        data_module
    )


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
