# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import openai
import backoff
import argparse
from tqdm import tqdm
from openai.error import RateLimitError

import torch

# User-defined Modules
from utils.metrics.full_suite import compute_test_metrics

# ------------------------- IMPLEMENTATION -----------------------------------

openai.organization = "org-EfQ17TvbVssFVjhyWwHXaVwk"
openai.api_key = "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"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/tb662/rds/hpc-work/automating-empathy-project/pretrained_models/GPT3")
    parser.add_argument("--force_new", action='store_true', help="Flag to ignore and overwrite pre-saved test data")
    parser.add_argument("--emo_classifier_dir", type=str, default=None)
    parser.add_argument("--intent_classifier_dir", type=str, default=None)
    parser.add_argument("--epitome_dir", type=str, default=None)
    parser.add_argument("--reward_model_dir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--pred_n_grams", type=int, default=4)
    parser.add_argument("--max_giveup_time", type=int, default=600)

    cli_args = parser.parse_args()

    return cli_args

def main():
    cli_args = parse_args()

    if (cli_args.temperature < 1 and cli_args.top_p < 1):
        print("WARNING: Setting both temperature and top_p less than 1.0 is not recommended!")
    
    # Create output directory if does not exist
    output_dir = os.path.abspath(cli_args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created test output directory at '{output_dir}'")
    
    # Check if any test data is saved at the output directory
    test_data = []
    test_fpath = os.path.join(output_dir, "test_data.json")
    if (not cli_args.force_new) and (os.path.isfile(test_fpath)):
        test_data = json.load(open(test_fpath))
        print(f"{len(test_data)} tested samples detected at '{test_fpath}'")
        print("Generation will continue for the remaining test samples.")

    # Load remaining contexts and targets from test dataset
    data_path = os.path.abspath(cli_args.dataset_dir)
    contexts = json.load(open(f"{data_path}/test/contexts.json"))
    targets = json.load(open(f"{data_path}/test/targets.json"))
    tested_ids = set([item["id"] for item in test_data])
    remaining_ids = [i for i in range(len(contexts)) if i not in tested_ids]

    # Backoff function to implement exponential backoff if rate limit is exceeded
    # The max_time argument gives the maximum elapsed time while trying a request before giving up
    @backoff.on_exception(backoff.expo, RateLimitError, max_time=cli_args.max_giveup_time)
    def completions_with_backoff(**kwargs):
        response = openai.Completion.create(**kwargs)
        return response

    if len(remaining_ids) > 0:
        print("Generating predictions from GPT3...")
        prefix = "The following is a conversation between two people. The speaker and listener are helpful, friendly, and empathetic.\n"
        for i in tqdm(remaining_ids):
            prompt = prefix
            for j, utt in enumerate(contexts[i]):
                if j % 2 == 0:
                    prompt += f"Speaker: {utt}\n"
                else:
                    prompt += f"Listener: {utt}\n"
            prompt += "Speaker: " if len(contexts[i]) % 2 == 0 else "Listener: "

            response = completions_with_backoff(
                engine="text-davinci-003",
                prompt=prompt,
                suffix="\n",
                temperature=cli_args.temperature,
                max_tokens=cli_args.max_tokens,
                top_p=cli_args.top_p,
            )

            test_entry = {
                "id": i,
                "input": prompt,
                "context": contexts[i],
                "target": targets[i],
                "output": response['choices'][0]['text']
            }
            test_data.append(test_entry)
            with open(test_fpath, "w") as f:
                json.dump(test_data, f)

        with open(test_fpath, "w") as f:
            test_data = sorted(test_data, key=lambda x:x["id"])
            json.dump(test_data, f)
            print(f"All test data saved at '{test_fpath}'")
    else:
        print("No samples left to generate!")

    test_metrics, classwise_metrics = compute_test_metrics(
        test_data,
        torch.device("cpu"),
        cli_args.emo_classifier_dir,
        cli_args.intent_classifier_dir,
        cli_args.epitome_dir,
        cli_args.reward_model_dir,
        None,
        None,
        None,
        cli_args.pred_n_grams,
    )

    fname = f"{output_dir}/test_metrics.json"
    with open(fname, "w") as f:
        json.dump(test_metrics, f)
    
    fname = f"{output_dir}/classwise_test_metrics.json"
    with open(fname, "w") as f:
        json.dump(classwise_metrics, f)

    print(f"All main and classwise test metrics saved at '{output_dir}'")

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------