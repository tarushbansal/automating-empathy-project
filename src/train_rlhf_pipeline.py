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
from data_loader import DataModule
from dialogue_model_supervisor import DialogueModelSupervisor
from reward_model_supervisor import RewardModelSupervisor
from rlhf_supervisor import RLHFSupervisor
from transformers import BertModel, BertTokenizer
from utils.train_utils import load_ckpt_path, load_config, SaveConfigCallback
from data_classes import GenerationConfig, PPOConfig

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--pretrained_dialogue_model_dir", type=str, default=None, required=True)
    parser.add_argument("--pretrained_reward_model_dir", type=str, default=None, required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=0.0001)
    parser.add_argument("--few_shot_training", action="store_true")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--clip_epsilon", type=int, default=0.2)
    parser.add_argument("--value_clip_range", type=float, default=0.2)
    parser.add_argument("--kl_penalty", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--_lambda", type=float, default=0.95)
    parser.add_argument("--vf_coeff", type=float, default=0.1)

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
            every_n_train_steps=1
        ),
        ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath=ckpt_dir,
            filename="{val_loss_epoch:.2f}-{epoch}"
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
        batch_size=cli_args.batch_size,
        initial_lr=cli_args.initial_lr,
        generation_config=GenerationConfig(
            max_new_tokens=cli_args.max_new_tokens,
            beam_width=cli_args.beam_width,
            sample=cli_args.sample,
            temperature=cli_args.temperature,
            top_p=cli_args.top_p,
            top_k=cli_args.top_k
        )
    )
    reward_model = RewardModelSupervisor.load_from_checkpoint(
        load_ckpt_path(cli_args.pretrained_reward_model_dir),
        model=BertModel.from_pretrained("bert-large-uncased"),
        batch_size=cli_args.batch_size,
        initial_lr=cli_args.initial_lr
    )
    reward_model.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    rlhf_supervisor = RLHFSupervisor(
        dialogue_model=dialogue_model,
        reward_model=reward_model,
        batch_size=cli_args.batch_size,
        ppo_config=PPOConfig(
            clip_epsilon=cli_args.clip_epsilon,
            value_clip_range=cli_args.value_clip_range,
            kl_penalty=cli_args.kl_penalty,
            gamma=cli_args.gamma,
            _lambda=cli_args._lambda,
            vf_coeff=cli_args.vf_coeff 
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
    callbacks.extend([
        SaveConfigCallback(config=config),
        # EarlyStopping(
        #     monitor="avg_val_loss",
        #     min_delta=0.01,
        #     mode="min",
        #     patience=1
        # )
    ]
    )


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
        rlhf_supervisor,
        data_module
    )


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
