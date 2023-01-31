# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import json
import os
from typing import Optional, Tuple, Union, Dict

# User-defined Modules
from base_classes import DialogueModelBase, TokenizerBase
from dialogue_model_supervisor import DialogueModelSupervisor
from utils.train_utils import load_ckpt_path, load_config

# ------------------------- IMPLEMENTATION -----------------------------------


def get_model_supervisor_and_config(
        model: Optional[str] = None,
        pretrained_model_dir: Optional[str] = None,
        initial_lr: Optional[float] = None
) -> Tuple[Union[DialogueModelSupervisor, Dict]]:
    
    # Sanity checks
    if model is None and pretrained_model_dir is None:
        raise ValueError( "Either a pretrained or a new model must be specified!")
    if model is not None and pretrained_model_dir is not None:
        raise ValueError("Cannot specify both a new and a pretrained model!")

    if model is not None:
        # Instantiate new model from config.json file
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
    else:
        # Load pretrained model from known configuration
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

# -----------------------------------------------------------------------------