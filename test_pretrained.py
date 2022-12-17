# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
from tqdm import tqdm
from typing import Iterable

import torch

# User-defined Modules
from metric_utils import compute_test_metrics

# ------------------------- IMPLEMENTATION -----------------------------------


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--dialogue_model", type=str, default=None, required=True)
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

    gen_cls = getattr(__import__("generation"), cli_args.dialogue_model)
    model_generation = gen_cls(
        device,
        cli_args.pred_beam_width,
        cli_args.max_pred_seq_len
    )
    model = model_generation.model
    tokenizer = model_generation.tokenizer

    # Set up data module
    with open(f"{dataset_dir}/test/contexts.json") as f:
        contexts = json.load(f)
    with open(f"{dataset_dir}/test/targets.json") as f:
        targets = json.load(f)
    with open(f"{dataset_dir}/test/emotions.json") as f:
        emotions = json.load(f)

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, contexts: Iterable) -> None:
            self.contexts = contexts

        def __len__(self) -> int:
            return len(self.contexts)

        def __getitem__(self, idx: int) -> list:
            return self.contexts[idx]

    test_dataloader = torch.utils.data.DataLoader(
        TestDataset(contexts),
        batch_size=cli_args.batch_size,
        num_workers=max(1, os.cpu_count() // 4),
        collate_fn=lambda x: x
    )

    print("Generating predictions from pretrained model...")
    predictions, enc_predictions = [], []
    for batch in tqdm(test_dataloader):
        outputs = model_generation.generate(batch)
        enc_predictions.extend(outputs)
        predictions.extend([tokenizer.decode(enc, skip_special_tokens=True)
                            for enc in outputs])
    print("Done.")

    os.makedirs(cli_args.output_dir, exist_ok=True)
    dir = os.path.abspath(cli_args.output_dir)
    print(f"Created test output directory at '{dir}'")

    N = len(contexts)
    test_data = []
    for i in range(N):
        entry = {
            "id": i,
            "context": contexts[i],
            "target": targets[i],
            "prediction": predictions[i],
            "emotion": emotions[i],
            "pred_emotion": None,
            "concepts": None
        }
        test_data.append(entry)

    fname = f"{dir}/test_data.json"
    with open(fname, "w") as f:
        json.dump(test_data, f)

    print(f"Test data saved at '{fname}'")

    print("Computing test metrics")
    enc_targets = [tokenizer(seq).input_ids for seq in targets]
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

    fname = f"{dir}/test_metrics.json"
    with open(fname, "w") as f:
        json.dump(test_metrics, f)

    print(f"Test metrics saved at '{fname}'")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
