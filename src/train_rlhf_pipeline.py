# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
from typing import Optional, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# User-defined Modules
from data_loader import DataModule
from dialogue_model_supervisor import DialogueModelSupervisor
from reward_model_supervisor import RewardModelSupervisor
from rlhf_supervisor import RLHFSupervisor
from transformers import GPT2Model, GPT2Tokenizer
from utils.train_utils import load_ckpt_path, load_config, SaveConfigCallback
from data_classes import GenerationConfig, PPOConfig
from train_dialogue_model import get_model_cls
from custom_tokenizers import TokenizerBase

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--pretrained_reward_model_dir", type=str, default=None, required=True)
    parser.add_argument("--dialogue_model", type=str, default=None)
    parser.add_argument("--pretrained_dialogue_model_dir", type=str, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=0.00001)
    parser.add_argument("--few_shot_training", action="store_true")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--clip_epsilon", type=int, default=0.2)
    parser.add_argument("--kl_penalty", type=int, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--vf_coeff", type=float, default=0.1)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)

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

    generation_config = GenerationConfig(
        max_new_tokens=cli_args.max_new_tokens,
        beam_width=cli_args.beam_width,
        sample=cli_args.sample,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        top_k=cli_args.top_k
    )

    # Initialise model and tokenizer from config
    if cli_args.pretrained_dialogue_model_dir is not None:
        config = load_config(cli_args.pretrained_dialogue_model_dir)
        tokenizer_cls = getattr(__import__("custom_tokenizers"), config["tokenizer"]["cls"])
        tokenizer_kwargs = config["tokenizer"]["kwargs"]
        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        model_cls = getattr(__import__("dialogue_models"), config["model"]["cls"])
        model_kwargs = config["model"]["kwargs"]
        model = model_cls(tokenizer=tokenizer, **model_kwargs)
        dialogue_model = DialogueModelSupervisor.load_from_checkpoint(
            load_ckpt_path(cli_args.pretrained_dialogue_model_dir),
            tokenizer=tokenizer,
            model=model,
            initial_lr=cli_args.initial_lr,
            generation_config=generation_config
        )

    else:
        if cli_args.dialogue_model is None:
            raise ValueError(
                "Either a pretrained or new dialogue model must be specified for training!")

        dirname = os.path.dirname(os.path.abspath(__file__))
        with open(f"{dirname}/configs.json") as f:
            model_config = json.load(f).get(cli_args.dialogue_model, {})

        model_cls = get_model_cls()
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
        dialogue_model = DialogueModelSupervisor(
            tokenizer=tokenizer,
            model=model,
            initial_lr=cli_args.initial_lr,
            generation_config=generation_config
        )
    
    reward_model = RewardModelSupervisor.load_from_checkpoint(
        load_ckpt_path(cli_args.pretrained_reward_model_dir),
        model=GPT2Model.from_pretrained("gpt2-large"),
        initial_lr=cli_args.initial_lr
    )
    reward_model.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    reward_model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    rlhf_supervisor = RLHFSupervisor(
        dialogue_model=dialogue_model,
        reward_model=reward_model,
        ppo_config=PPOConfig(
            clip_epsilon=cli_args.clip_epsilon,
            kl_penalty=cli_args.kl_penalty,
            gamma=cli_args.gamma,
            lam=cli_args.lam,
            vf_coeff=cli_args.vf_coeff,
            entropy_coeff=cli_args.entropy_coeff
        ),
        initial_lr=cli_args.initial_lr
    )

    # Set up data module
    data_module = DataModule(dataset_dir=cli_args.dataset_dir,
                             batch_size=cli_args.batch_size,
                             tokenizer=tokenizer,
                             num_workers=max(1, os.cpu_count() // 4),
                             model_has_encoder=model.has_encoder,
                             few_shot_training=cli_args.few_shot_training)

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
        callbacks=callbacks
    )

    # Train / Validate the model
    trainer.fit(
        rlhf_supervisor,
        data_module
    )


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
