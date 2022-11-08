# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import os
import argparse
from typing import Optional, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# User-defined Modules
from data_loader import DataModule
from model_supervisor import ModelSupervisor
from base_classes import DialogueModelBase, TokenizerBase
from utils import (
    load_val_ckpt_path, 
    load_config, 
    SaveConfigCallback
)

# ------------------------- IMPLEMENTATION -----------------------------------

def get_model_cls() -> DialogueModelBase:
    single_arg_parser = argparse.ArgumentParser()
    single_arg_parser.add_argument("--dialogue_model", type=str, default=None, required=True)
    args, _ = single_arg_parser.parse_known_args()

    model_cls = getattr(__import__("dialogue_models"), args.dialogue_model)

    if not issubclass(model_cls, DialogueModelBase):
        raise ValueError("Model must be derived from base class 'DialogueModelBase'!")

    return model_cls

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--dialogue_model", type=str, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=0.0001)
    parser.add_argument("--pred_beam_width", type=int, default=1)
    parser.add_argument("--max_pred_seq_len", type=int, default=200)
    parser.add_argument("--pred_n_grams", type=int, default=4)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)

    cli_args = parser.parse_args()

    return cli_args

def get_model_checkpoints(ckpt_dir: str) -> Optional[List[ModelCheckpoint]]:
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
    
    # Set up tensorboard logger
    logger = TensorBoardLogger(
        save_dir=cli_args.output_dir,
        name="tensorboard_logs"
    )

    # Initialise model and tokenizer from config
    if cli_args.pretrained_model_dir is not None:
        config = load_config(cli_args.pretrained_model_dir)
        tokenizer_cls = getattr(__import__("data_tokenizers"), config["tokenizer"]["cls"])
        tokenizer_kwargs = config["tokenizer"]["kwargs"]
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
        model_kwargs = config["model"]["kwargs"]
        model = model_cls(tokenizer=tokenizer, **model_kwargs)

    else:
        if cli_args.dialogue_model is None:
            raise ValueError(
                "Either a new or pretrained dialogue model must be specified!")

        dirname = os.path.dirname(os.path.abspath(__file__))
        with open(f"{dirname}/configs.json") as f:
            model_config = json.load(f).get(cli_args.dialogue_model, {})

        model_cls = get_model_cls()
        tokenizer_name = model_cls.tokenizer_cls()
        if tokenizer_name is None:
            raise ValueError(
                "Must specify the tokenizer associated with the model as a static method!")
        
        tokenizer_cls = getattr(__import__("data_tokenizers"), tokenizer_name)
        if not issubclass(tokenizer_cls, TokenizerBase):
            raise ValueError(
                "Tokenizer must be derived from base class 'TokenizerBase'!")
        tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        model_kwargs = model_config.get("model_kwargs", {})
        model = model_cls(tokenizer=tokenizer, **model_kwargs)
    
    # Set up model supervisor
    model_supervisor = ModelSupervisor(
        tokenizer=tokenizer,
        model=model,
        batch_size=cli_args.batch_size,
        initial_lr=cli_args.initial_lr,
        test_output_dir=logger.log_dir,
        pred_beam_width=cli_args.pred_beam_width,
        max_pred_seq_len=cli_args.max_pred_seq_len,
        pred_n_grams=cli_args.pred_n_grams
    )

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             tokenizer=tokenizer,
                             num_workers=max(1, os.cpu_count() // 4))

    # Set up model checkpointing
    ckpt_dir = f"{logger.log_dir}/checkpoints"
    checkpoint_callback = get_model_checkpoints(ckpt_dir)
    callbacks = []
    if checkpoint_callback is not None:
        callbacks.extend(checkpoint_callback)
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
        data_module, 
        ckpt_path=load_val_ckpt_path(cli_args.pretrained_model_dir)
    )

    # Test the model
    trainer.test(
        model_supervisor, 
        data_module,
        ckpt_path=load_val_ckpt_path(logger.log_dir)    
    )


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------