# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Iterable

import torch

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)

# User-defined Modules
from metric_utils import compute_test_metrics
from generation_utils import generate_map

# ------------------------- IMPLEMENTATION -----------------------------------

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--model_name", type=str, default=None, required=True)
    parser.add_argument("--model_type", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pred_beam_width", type=int, default=1)
    parser.add_argument("--max_pred_seq_len", type=int, default=200)
    parser.add_argument("--pred_n_grams", type=int, default=4)
    cli_args, _ = parser.parse_known_args()

    dataset_dir = os.path.abspath(cli_args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Specified dataset directory does not exist!")

    # Set appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialise model and tokenizer 
    if cli_args.model_type == "EncoderDecoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(cli_args.model_name)
    elif cli_args.model_type == "Decoder":
        model = AutoModelForCausalLM.from_pretrained(cli_args.model_name)
    else:
        raise ValueError("Model type unsupported! Must be either'EncoderDecoder' or 'Decoder'")
    
    if cli_args.model_name not in generate_map:
        raise ValueError(
            "Please define a generate function for the specified model!")   
    
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(cli_args.model_name)
    model_generation = generate_map[cli_args.model_name](
        model, 
        tokenizer,
        device, 
        cli_args.pred_beam_width, 
        cli_args.max_pred_seq_len
    )
    
    # Make modifications to tokenizer to allow padding
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("WARNING: Using EOS token for padding. All EOS tokens added to" +
              " model inputs will be masked!")

    # Set up data module
    context_data = np.load(f"{dataset_dir}/test/contexts.npy", allow_pickle=True)
    target_data = np.load(f"{dataset_dir}/test/targets.npy", allow_pickle=True)
    
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, context: Iterable) -> None:
            self.context = context
        def __len__(self) -> int:
            return len(self.context)
        def __getitem__(self, idx: int) -> list:
            return self.context[idx]

    test_dataloader = torch.utils.data.DataLoader(
        TestDataset(context_data),
        batch_size=cli_args.batch_size,
        num_workers=max(1, os.cpu_count() // 4),
        collate_fn=lambda x: x
    )

    targets = [[seq.split(" ")] for seq in target_data]
    enc_targets = [tokenizer(seq).input_ids for seq in target_data]
    predictions, enc_predictions = [], []

    print("Generating predictions from model...")
    for batch in tqdm(test_dataloader):
        outputs = model_generation.generate(batch)
        enc_predictions.extend(outputs)
        predictions.extend([tokenizer.decode(enc, skip_special_tokens=True).split(" ")
                            for enc in outputs])
    print("Done.")

    print("Computing test metrics")
    test_metrics = compute_test_metrics(
        targets,
        predictions,
        enc_targets,
        enc_predictions,
        cli_args.pred_n_grams,
        None,
        model.get_input_embeddings()
    )
    print("Test metrics computed.")

    os.makedirs(cli_args.output_dir, exist_ok=True)
    dir = os.path.abspath(cli_args.output_dir)
    
    if os.path.isdir(dir):
        with open(f"{dir}/test_predictions.txt", "w") as f:
            for context, target, prediction in zip(context_data, target_data, predictions):
                prediction = " ".join(prediction)
                f.write(f"Context: {context}; Target: {target}; Predicted: {prediction}\n")
        with open(f"{dir}/test_metrics.json", "w") as f:
            json.dump(test_metrics, f)
        
    print(f"Test predictions and metrics saved at '{dir}'")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------