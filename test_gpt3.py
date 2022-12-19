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
from metric_utils import compute_test_metrics

# ------------------------- IMPLEMENTATION -----------------------------------

openai.organization = "org-V9hIr5Y4NaEAvSMDkk6QeLhw"
openai.api_key = "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"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/tb662/rds/hpc-work/pretrained_models/GPT3")
    parser.add_argument("--force_new", action='store_true', help="Flag to ignore and overwrite pre-saved test data")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--pred_n_grams", type=int, default=4)
    parser.add_argument("--max_giveup_time", type=int, default=600)
    cli_args = parser.parse_args()

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
        print(f"Test data detected at '{test_fpath}'")
        print("Generation will continue for the remaining test samples.")

    # Load contexts and targets from test dataset
    data_path = os.path.abspath(cli_args.dataset_dir)
    contexts = json.load(open(f"{data_path}/test/contexts.json"))[len(test_data):]
    targets = json.load(open(f"{data_path}/test/targets.json"))[len(test_data):]

    # Backoff function to implement exponential backoff if rate limit is exceeded
    # The max_time argument gives the maximum elapsed time while trying a request before giving up
    @backoff.on_exception(backoff.expo, RateLimitError, max_time=cli_args.max_giveup_time)
    def completions_with_backoff(**kwargs):
        response = openai.Completion.create(**kwargs)
        return response

    print("Generating predictions from GPT3...")
    prefix = "The following is a conversation between two people. The listener is helpful, friendly, and empathetic.\n"
    for i, context in tqdm(enumerate(contexts), total=len(contexts)):
        gpt_prompt = prefix
        for j, utt in enumerate(context):
            if j % 2 == 0:
                gpt_prompt += f"Speaker: {utt}\n"
            else:
                gpt_prompt += f"Listener: {utt}\n"
        gpt_prompt += "Listener: "

        response = completions_with_backoff(
            engine="text-davinci-003",
            prompt=gpt_prompt,
            suffix="\n",
            temperature=cli_args.temperature,
            max_tokens=cli_args.max_tokens,
            top_p=cli_args.top_p,
        )

        test_entry = {
            "id": i,
            "context": contexts[i],
            "target": targets[i],
            "prediction": response['choices'][0]['text']
        }
        test_data.append(test_entry)

        with open(test_fpath, "w") as f:
            json.dump(test_data, f)

    print("Done.")
    print(f"All test data saved at '{test_fpath}'")

    print("Computing test metrics...")
    test_metrics = compute_test_metrics(
        [item["target"] for item in test_data],
        [item["prediction"] for item in test_data],
        None,
        None,
        None,
        cli_args.pred_n_grams,
    )
    print("Test metrics computed.")

    fname = f"{output_dir}/test_metrics.json"
    with open(fname, "w") as f:
        json.dump(test_metrics, f)

    print(f"Test metrics saved at '{fname}'")

# ----------------------------------------------------------------------------