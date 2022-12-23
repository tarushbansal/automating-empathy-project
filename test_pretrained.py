# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import math
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple

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
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--bleu_n_grams", type=int, default=4)
    cli_args, _ = parser.parse_known_args()

    dataset_dir = os.path.abspath(cli_args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Specified dataset directory does not exist!")

    # Set appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_cls = getattr(__import__("pretrained_gen"), cli_args.dialogue_model)
    model_generation = gen_cls(
        device,
        cli_args.beam_width,
        cli_args.max_new_tokens
    )
    model = model_generation.model
    tokenizer = model_generation.tokenizer

    # Set up data module
    with open(f"{dataset_dir}/test/contexts.json") as f:
        contexts = json.load(f)
    with open(f"{dataset_dir}/test/targets.json") as f:
        targets = json.load(f)

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, contexts: list, targets: list) -> None:
            self.contexts = contexts
            self.targets = targets

        def __len__(self) -> int:
            return len(self.contexts)

        def __getitem__(self, idx: int) -> list:
            return self.contexts[idx], self.targets[idx]

    def collate_fn(batch: List[Tuple]) -> Dict[str, Tuple]:
        contexts, targets = zip(*batch)
        return {
            "contexts": contexts,
            "targets": targets
        }

    test_dataloader = torch.utils.data.DataLoader(
        TestDataset(contexts, targets),
        batch_size=cli_args.batch_size,
        num_workers=max(1, os.cpu_count() // 4),
        collate_fn=collate_fn
    )

    print("Generating predictions from pretrained model...")
    predictions, enc_predictions, cross_entropy = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            preds, nll = model_generation.generate(batch)
            cross_entropy.append(nll)
            enc_predictions.extend(preds)
            predictions.extend([tokenizer.decode(enc, skip_special_tokens=True) for enc in preds])
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
        }
        test_data.append(entry)

    fname = f"{dir}/test_data.json"
    with open(fname, "w") as f:
        json.dump(test_data, f)

    print(f"Test data saved at '{fname}'")

    print("Computing test metrics...")
    enc_targets = [tokenizer(seq).input_ids for seq in targets]
    test_metrics = compute_test_metrics(
        targets,
        predictions,
        enc_targets,
        enc_predictions,
        model.get_input_embeddings(),
        cli_args.bleu_n_grams,
    )
    test_metrics["ppl"] = math.exp(sum(cross_entropy) / len(cross_entropy))
    print("Test metrics computed.")

    fname = f"{dir}/test_metrics.json"
    with open(fname, "w") as f:
        json.dump(test_metrics, f)

    print(f"Test metrics saved at '{fname}'")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
