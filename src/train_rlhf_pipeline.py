# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# User-defined Modules
from data_loader import DataModule
from rlhf_supervisor import RLHFSupervisor
from data_classes import GenerationConfig, PPOConfig
from setup import get_model_supervisor
from utils.train import get_model_checkpoints

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--reward_model_dir", type=str, default=None, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pretrained_model_dir", type=str, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=0.00001)
    parser.add_argument("--data_erasure_level", type=float, default=1.0)
    parser.add_argument("--few_shot_training", action="store_true")
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument('--sample', action='store_true', default=None)
    parser.add_argument('--no_sample', dest='sample', action='store_false')   
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--length_alpha", type=float, default=None)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--kl_penalty", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--vf_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)

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

    # Define generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=cli_args.max_new_tokens,
        beam_width=cli_args.beam_width,
        sample=cli_args.sample,
        temperature=cli_args.temperature,
        top_p=cli_args.top_p,
        top_k=cli_args.top_k,
        length_alpha=cli_args.length_alpha
    )

    # Set up dialogue and reward model as well as RLHF pipeline
    dialogue_model = get_model_supervisor(
        model=cli_args.model,
        pretrained_model_dir=cli_args.pretrained_model_dir,
        kwargs={"generation_config": generation_config}
    )

    reward_model = get_model_supervisor(
        pretrained_model_dir=cli_args.reward_model_dir,
        reward_model=True
    )

    num_devices = cli_args.num_nodes * torch.cuda.device_count() if torch.cuda.is_available() else 1
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
        initial_lr=cli_args.initial_lr,
        batch_size=cli_args.batch_size * num_devices,
        data_erasure_level=cli_args.data_erasure_level
    )

    # Set up data module
    data_module = DataModule(
        dataset_dir=cli_args.dataset_dir,
        batch_size=cli_args.batch_size,
        tokenizer=dialogue_model.tokenizer,
        num_workers=max(1, os.cpu_count() // 4),
        model_has_encoder=dialogue_model.model.has_encoder,
        data_erasure_level=cli_args.data_erasure_level,
        few_shot_training=cli_args.few_shot_training
    )

    # Set up model checkpointing
    ckpt_dir = f"{logger.log_dir}/checkpoints"
    checkpoint_callback = get_model_checkpoints(ckpt_dir)
    checkpoint_callback += [
        ModelCheckpoint(
            monitor="val_reward_epoch",
            dirpath=ckpt_dir,
            filename="{val_reward_epoch:.2f}-{epoch}",
            every_n_epochs=1,
            mode="max"
        )
    ]
    callbacks = []
    if checkpoint_callback is not None:
        callbacks.extend(checkpoint_callback)

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        num_nodes=cli_args.num_nodes,
        strategy="ddp_find_unused_parameters_false",
        max_epochs=cli_args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    # Train / Validate the model
    trainer.fit(
        rlhf_supervisor,
        data_module
    )


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
