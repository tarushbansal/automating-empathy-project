# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
from typing import Optional, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

# ------------------------- IMPLEMENTATION -----------------------------------

class SaveConfigCallback(pl.Callback):
    """Saves a config file to the log_dir when training starts."""
    def __init__(self, config: dict,) -> None:
        self.config = config

    @rank_zero_only
    def on_train_start(
        self, 
        trainer: pl.Trainer, 
        *_
    ) -> None:

        if trainer.log_dir is None:
            raise ValueError("No log directory specified to save config file!")
        
        if not os.path.isdir(trainer.log_dir):
            os.makedirs(trainer.log_dir)
        
        output_fpath = f"{trainer.log_dir}/config.json"
        with open(output_fpath, "w") as f:
            json.dump(self.config, f)


def load_config(trained_model_dir: str) -> dict:
    config = None
    config_fpath = os.path.join(trained_model_dir, "config.json")
    if os.path.isfile(config_fpath):
        with open(config_fpath) as f:
            config = json.load(f)
    else:
        raise ValueError("Specified directory does not have a 'config.json' file!")
    
    return config


def load_ckpt_path(trained_model_dir: Optional[str]) -> Optional[str]:
    ckpt_path = None
    if trained_model_dir is not None:
        if os.path.isdir(trained_model_dir):
            load_ckpt_dir = os.path.join(trained_model_dir, "checkpoints")
            if os.path.isdir(load_ckpt_dir):
                files = os.listdir(load_ckpt_dir)
                for fname in sorted(files):
                    if fname.startswith("val_loss_epoch") and fname.endswith(".ckpt"):
                        ckpt_path = os.path.join(load_ckpt_dir, fname)
                        break
                    elif fname.startswith("train_loss_epoch") and fname.endswith(".ckpt"):
                        ckpt_path = os.path.join(load_ckpt_dir, fname)
                        break
            if ckpt_path is None:
                raise ValueError(
            "No valid checkpoint files found in the specified model directory! " + 
            "Valid formats are 'val_loss_epoch*.ckpt' or 'train_loss_epoch*.ckpt'")
        else:
            raise ValueError("Specified model directory does not exist!")
    
    return ckpt_path


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

#----------------------------------------------------------------------------