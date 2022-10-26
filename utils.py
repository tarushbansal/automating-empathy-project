# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
from typing import Optional

import pytorch_lightning as pl

# ------------------------- IMPLEMENTATION -----------------------------------

class SaveConfigCallback(pl.Callback):
    """Saves a config file to the log_dir when training starts."""
    def __init__(
        self,
        config: dict,
    ) -> None:
        self.config = config

    def setup(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        stage: Optional[str] = None
    ) -> None:

        if trainer.log_dir is None:
            raise ValueError(
                "No log directory specified to save config file!")
        output_fpath = f"{trainer.log_dir}/config.json"

        # Save the file on rank 0 (to avoid race conditions)
        if trainer.is_global_zero:
            if not os.path.isdir(trainer.log_dir):
                os.makedirs(trainer.log_dir)
            with open(output_fpath, "w") as f:
                json.dump(self.config, f)


def load_config(trained_model_dir: str) -> dict:
    config = None
    config_fpath = os.path.join(trained_model_dir, "config.json")
    if os.path.isfile(config_fpath):
        with open(config_fpath) as f:
            config = json.load(f)
    else:
        raise ValueError("Specified directory doesn't have a config.json file!")
    
    return config


def load_val_ckpt_path(trained_model_dir: Optional[str]) -> Optional[str]:
    ckpt_path = None
    if trained_model_dir is not None:
        if os.path.isdir(trained_model_dir):
            load_ckpt_dir = os.path.join(trained_model_dir, "checkpoints")
            if os.path.isdir(load_ckpt_dir):
                files = os.listdir(load_ckpt_dir)
                for fname in sorted(files):
                    if fname.startswith("avg_val_loss") and fname.endswith(".ckpt"):
                        ckpt_path = os.path.join(load_ckpt_dir, fname)
                        break
                    elif fname.startswith("train_loss") and fname.endswith(".ckpt"):
                        ckpt_path = os.path.join(load_ckpt_dir, fname)
                        break
            if ckpt_path is None:
                raise ValueError(
            "No checkpoint files found in the specified model directory!")
        else:
            raise ValueError("Specified model directory does not exist!")
    
    return ckpt_path

#----------------------------------------------------------------------------