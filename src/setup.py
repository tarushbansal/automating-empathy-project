# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import os
from typing import Optional, Tuple, Union, Dict

# User-defined Modules
from reward_model_supervisor import RewardModelSupervisor
from dialogue_model_supervisor import DialogueModelSupervisor
from utils.train import load_ckpt_path
from data_classes import ModelConfig

# ------------------------- IMPLEMENTATION -----------------------------------


def get_model_supervisor(
    model: Optional[str] = None,
    pretrained_model_dir: Optional[str] = None,
    kwargs: Dict = {},
    reward_model: bool = False
) -> Tuple[Union[DialogueModelSupervisor, RewardModelSupervisor, Dict]]:
    
    # Sanity checks
    if model is None and pretrained_model_dir is None:
        raise ValueError( "Either a pretrained or a new model must be specified!")
    if model is not None and pretrained_model_dir is not None:
        raise ValueError("Cannot specify both a new and a pretrained model!")

    supervisor_cls = RewardModelSupervisor if reward_model else DialogueModelSupervisor
    if model is not None:
        # Instantiate new model from config.json file
        dirname = os.path.dirname(os.path.abspath(__file__))
        with open(f"{dirname}/configs.json") as f:
            config = json.load(f).get(model)

        model_supervisor = supervisor_cls(
            config=ModelConfig(
                model_cls=config["model"]["cls"],
                model_kwargs=config["model"]["kwargs"],
                tokenizer_cls=config["tokenizer"]["cls"],
                tokenizer_kwargs=config["tokenizer"]["kwargs"]).__dict__,
            **kwargs
        )
    else:
        # Load pretrained model
        model_supervisor = supervisor_cls.load_from_checkpoint(
            load_ckpt_path(pretrained_model_dir),
            strict=False,
            **kwargs
        )

    return model_supervisor

# -----------------------------------------------------------------------------