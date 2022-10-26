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
    SaveConfigCallback,
    SaveTestMetricsCallback
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
    parser.add_argument("--bleu_n_grams", type=int, default=4)
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
    
    if cli_args.pretrained_model_dir is not None:
        # Initialise model and tokenizer from config file
        config = load_config(cli_args.pretrained_model_dir)
        tokenizer_cls = getattr(__import__("data_tokenizers"), config["tokenizer"]["cls"])
        tokenizer_kwargs = config["tokenizer"]["kwargs"]
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
        model_kwargs = config["model"]["kwargs"]
        model = model_cls(**model_kwargs)

    else:
        if cli_args.dialogue_model is None:
            raise ValueError(
                "Either a new or pretrained dialogue model must be specified!")

        # Read model config
        dirname = os.path.dirname(os.path.abspath(__file__))
        with open(f"{dirname}/configs.json") as f:
            configs = json.load(f)
            if cli_args.dialogue_model not in configs:
                raise ValueError(
                    f"Configuration not found for dialogue model {cli_args.dialogue_model}!")
            model_config = configs[cli_args.dialogue_model]

        # Additional check for Hugging Face models
        if (cli_args.dialogue_model in ["HuggingFaceEncoderDecoderModel", "HuggingFaceDecoderModel"]) and (
            model_config["model_kwargs"]["model_name"] != model_config["tokenizer_kwargs"]["model_name"]):
            raise ValueError("Pretrained model names supplied to tokenizer and model don't match!")

        # Get tokenizer class and kwargs, then instantiate tokenizer
        tokenizer_cls = getattr(__import__("data_tokenizers"), model_config["tokenizer_cls"])
        if not issubclass(tokenizer_cls, TokenizerBase):
            raise ValueError("Tokenizer must be derived from base class 'TokenizerBase'!")
        tokenizer_kwargs = model_config["tokenizer_kwargs"]
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        # Set up model kwargs and instantiate
        model_kwargs = {
            "vocab_size": tokenizer.vocab_size,
            "num_emo_labels": tokenizer.num_emo_labels,
            "padding_idx": tokenizer.PAD_IDX,
            **model_config["model_kwargs"]
        }
        model_cls = get_model_cls()
        model = model_cls(**model_kwargs)
    
    # Set up model supervisor
    model_supervisor = ModelSupervisor(
        tokenizer=tokenizer,
        model=model,
        initial_lr=cli_args.initial_lr,
        pred_beam_width=cli_args.pred_beam_width,
        max_pred_seq_len=cli_args.max_pred_seq_len,
        bleu_n_grams=cli_args.bleu_n_grams
    )

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             tokenizer=tokenizer,
                             num_workers=max(1, os.cpu_count() // 4))

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
    callbacks.extend([
        SaveConfigCallback(config=config), 
        SaveTestMetricsCallback(logger.log_dir)
    ])

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
    test_metrics = trainer.test(model_supervisor, data_module)
    print(test_metrics)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------