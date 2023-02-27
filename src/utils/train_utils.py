# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
from typing import Optional, List

from pytorch_lightning.callbacks import ModelCheckpoint

# ------------------------- IMPLEMENTATION -----------------------------------


def load_ckpt_path(trained_model_dir: Optional[str]) -> Optional[str]:
    ckpt_path = None
    if trained_model_dir is not None:
        if os.path.isdir(trained_model_dir):
            load_ckpt_dir = os.path.join(trained_model_dir, "checkpoints")
            if os.path.isdir(load_ckpt_dir):
                files = os.listdir(load_ckpt_dir)
                for fname in sorted(files):
                    if fname.endswith(".ckpt"):
                        ckpt_path = os.path.join(load_ckpt_dir, fname)
                        break
            if ckpt_path is None:
                raise ValueError(
            "No valid checkpoint files found in the specified model directory! " + 
            "Valid formats are '*.ckpt'")
        else:
            raise ValueError("Specified model directory does not exist!")
    
    return ckpt_path


def get_model_checkpoints(ckpt_dir: str) -> Optional[List[ModelCheckpoint]]:
    if ckpt_dir is None:
        return None

    return [
        ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath=ckpt_dir,
            filename="{val_loss_epoch:.2f}-{epoch}",
            every_n_epochs=1
        )
    ]

#----------------------------------------------------------------------------