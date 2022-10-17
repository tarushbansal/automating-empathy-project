# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# User-defined Modules
from model_supervisor import ModelSupervisor
from data_loader import DataModule
from token_indexer import TokenIndexer
from utils import load_val_ckpt_path

# ------------------------- IMPLEMENTATION -----------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=0.001)
    parser.add_argument("--max_seq_len", type=int, default=2000)
    parser.add_argument("--trained_model_dir", type=str, default=None)

    cli_args = parser.parse_args()

    return cli_args

def get_model_checkpoints(ckpt_dir: str) -> Optional[ModelCheckpoint]:
    if ckpt_dir is None:
        return None
    
    return [
        ModelCheckpoint(
            monitor="train_loss",
            dirpath=ckpt_dir,
            filename="{train_loss:.2f}-{epoch}",
            every_n_train_steps=1
        ),
        ModelCheckpoint(
            monitor="avg_val_loss",
            dirpath=ckpt_dir,
            filename="{avg_val_loss:.2f}-{epoch}"
        ),
    ]

def main():
    # Parse command line arguments
    cli_args = parse_args()

    if not os.path.isdir(cli_args.dataset_dir):
        raise ValueError(f"Specified dataset directory does not exist!")

    # Initialise token indexer
    token_indexer = TokenIndexer()

    # Set up model supervisor
    model_supervisor = ModelSupervisor(
        max_seq_len=cli_args.max_seq_len,
        token_indexer=token_indexer,
        initial_lr=cli_args.initial_lr
    )

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             token_indexer=token_indexer,
                             num_workers=os.cpu_count())

    # Set up model checkpointing and logging
    logger = TensorBoardLogger(
        save_dir=cli_args.output_dir,
        name="tensorboard_logs"
    )
    
    ckpt_dir = f"{logger.log_dir}/checkpoints"
    checkpoint_callback = get_model_checkpoints(ckpt_dir)
    callbacks = []
    if checkpoint_callback is not None:
        callbacks.extend(checkpoint_callback)

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto", 
        devices=-1 if torch.cuda.is_available() else 1,
        num_nodes=cli_args.num_nodes,
        strategy="ddp_find_unused_parameters_false",
        max_epochs=cli_args.max_epochs, 
        logger=logger, 
        callbacks=callbacks
    )

    # Train / Validate the model
    trainer.fit(
        model_supervisor, 
        data_module, 
        ckpt_path=load_val_ckpt_path(cli_args.trained_model_dir)
    )

    # Test the model
    trainer.test(model_supervisor, data_module)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------