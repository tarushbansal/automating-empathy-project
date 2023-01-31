# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import os
import argparse
from typing import Optional, List, Tuple, Union, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# User-defined Modules
from base_classes import DialogueModelBase, TokenizerBase
from data_loader import DataModule
from dialogue_model_supervisor import DialogueModelSupervisor
from utils.train_utils import (
    load_ckpt_path,
    load_config,
    SaveConfigCallback
)

# ------------------------- IMPLEMENTATION -----------------------------------


def get_model_supervisor_and_config(
        model: Optional[str] = None,
        pretrained_model_dir: Optional[str] = None,
        initial_lr: Optional[float] = None
) -> Tuple[Union[DialogueModelSupervisor, Dict]]:
    
    if pretrained_model_dir is not None:
        config = load_config(pretrained_model_dir)
        tokenizer_cls = getattr(__import__("custom_tokenizers"), config["tokenizer"]["cls"])
        tokenizer_kwargs = config["tokenizer"]["kwargs"]
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
        model_kwargs = config["model"]["kwargs"]
        model = model_cls(tokenizer=tokenizer, **model_kwargs)
        model_supervisor = DialogueModelSupervisor.load_from_checkpoint(
            load_ckpt_path(pretrained_model_dir),
            strict=False,
            tokenizer=tokenizer,
            model=model,
            initial_lr=initial_lr
        )
    else:
        if model is None:
            raise ValueError(
                "Either a pretrained or new model must be specified for training!")

        dirname = os.path.dirname(os.path.abspath(__file__))
        with open(f"{dirname}/configs.json") as f:
            model_config = json.load(f).get(model, {})

        size_suffixes = ["_SMALL", "_MEDIUM", "_LARGE"]
        for suffix in size_suffixes:
            if model.endswith(suffix):
                model = model.replace(suffix, "")
                break

        model_cls = getattr(__import__("dialogue_models"), model)

        if not issubclass(model_cls, DialogueModelBase):
            raise ValueError("Model must be derived from base class 'DialogueModelBase'!")
        
        tokenizer_name = model_cls.tokenizer_cls()
        if tokenizer_name is None:
            raise ValueError(
                "Must specify the tokenizer associated with the model as a static method!")

        tokenizer_cls = getattr(__import__("custom_tokenizers"), tokenizer_name)
        if not issubclass(tokenizer_cls, TokenizerBase):
            raise ValueError(
                "Tokenizer must be derived from base class 'TokenizerBase'!")
        tokenizer_kwargs = model_config.pop("tokenizer_kwargs", {})
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        model_kwargs = model_config
        model = model_cls(tokenizer=tokenizer, **model_kwargs)
        model_supervisor = DialogueModelSupervisor(
            tokenizer=tokenizer,
            model=model,
            initial_lr=initial_lr
        )
    config = {
        "model": {
            "cls": model_cls.__name__,
            "kwargs": model_kwargs
        },
        "tokenizer": {
            "cls": tokenizer_cls.__name__,
            "kwargs": tokenizer_kwargs,
        }
    }
    return model_supervisor, config

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
        ),
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
