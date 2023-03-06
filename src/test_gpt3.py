# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import openai
import backoff
import argparse
from tqdm import tqdm
from openai.error import RateLimitError

# User-defined Modules
from utils.metrics.full_suite import compute_test_metrics

# ------------------------- IMPLEMENTATION -----------------------------------

openai.organization = "org-d2uKBRnpSfSqcHfeWCRLDIMR"
openai.api_key = "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"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/tb662/rds/hpc-work/automating-empathy-project/pretrained_models/GPT3")
    parser.add_argument("--reuse_cache", action='store_true')
    parser.add_argument("--emo_classifier_dir", type=str, default=None)
    parser.add_argument("--intent_classifier_dir", type=str, default=None)
    parser.add_argument("--epitome_dir", type=str, default=None)
    parser.add_argument("--reward_model_dir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--metric_n_grams", type=int, default=4)
    parser.add_argument("--max_giveup_time", type=int, default=600)

    cli_args = parser.parse_args()

    return cli_args

def main():
    cli_args = parse_args()

    if (cli_args.temperature < 1 and cli_args.top_p < 1):
        print("WARNING: Setting both temperature and top_p less than 1.0 is not recommended!")
    
    # Create output directory if does not exist
    output_dir = os.path.abspath(cli_args.output_dir)
    test_fpath = os.path.join(output_dir, "test_data.json")
    if cli_args.reuse_cache:
        test_data = json.load(open(test_fpath))
        print(f"{len(test_data)} cached samples detected at '{test_fpath}'")
        print("WARNING: Generation will continue for the remaining test samples. " + 
              "Make sure cached data matches the specified dataset!")
    else:
        test_data = []
        os.makedirs(output_dir)
        print(f"Created test output directory at '{output_dir}'")

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
    else:
        print("No samples left to test!")

    test_metrics, classwise_metrics = compute_test_metrics(
        test_data=test_data,
        emo_classifier_dir=cli_args.emo_classifier_dir,
        intent_classifier_dir=cli_args.intent_classifier_dir,
        epitome_dir=cli_args.epitome_dir,
        reward_model_dir=cli_args.reward_model_dir,
        metric_n_grams=cli_args.metric_n_grams,
    )

    with open(test_fpath, "w") as f:
        test_data = sorted(test_data, key=lambda x:x["id"])
        json.dump(test_data, f)
        print(f"All test data saved at '{test_fpath}'")

    fname = f"{output_dir}/test_metrics.json"
    with open(fname, "w") as f:
        json.dump(test_metrics, f)
    
    fname = f"{output_dir}/classwise_test_metrics.json"
    with open(fname, "w") as f:
        json.dump(classwise_metrics, f)

    print(f"All test metrics saved at directory '{output_dir}'")

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------