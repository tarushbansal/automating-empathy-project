# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import time
import openai
import argparse
from tqdm import tqdm

# User-defined Modules
from metric_utils import compute_test_metrics

# ------------------------- IMPLEMENTATION -----------------------------------

openai.organization = "org-jip1qiNxxKzVhTcIXFs5CDW4"
openai.api_key = "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"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/tb662/rds/hpc-work/pretrained_models/GPT3")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--pred_n_grams", type=int, default=4)
    cli_args = parser.parse_args()

    if (cli_args.temperature < 1 and cli_args.top_p < 1):
        print("WARNING: Setting both temperature and top_p less than 1.0 is not recommended!")
    
    data_path = os.path.abspath(cli_args.dataset_dir)
    contexts = json.load(open(f"{data_path}/test/contexts.json"))
    targets = json.load(open(f"{data_path}/test/targets.json"))

    print("Generating predictions from GPT3...")
    test_data, predictions = [], []
    prefix = "The following is a conversation between two people. The listener is helpful, friendly, and empathetic.\n"
    for i, context in tqdm(enumerate(contexts), total=len(contexts)):
        gpt_prompt = prefix
        for j, utt in enumerate(context):
            if j % 2 == 0:
                gpt_prompt += f"Speaker: {utt}\n"
            else:
                gpt_prompt += f"Listener: {utt}\n"
        gpt_prompt += "Listener: "

        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=gpt_prompt,
        suffix="\n",
        temperature=cli_args.temperature,
        max_tokens=cli_args.max_tokens,
        top_p=cli_args.top_p,
        )

        predictions.append(response['choices'][0]['text'])
        test_entry = {
            "id": i,
            "context": contexts[i],
            "target": targets[i],
            "prediction": predictions[i],
        }
        test_data.append(test_entry)

        time.sleep(1.5)
    
    print("Done.")

    output_dir = os.path.abspath(cli_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created test output directory at '{output_dir}'")

    fname = f"{output_dir}/test_data.json"
    with open(fname, "w") as f:
        json.dump(test_data, f)

    print(f"Test data saved at '{fname}'")

    print("Computing test metrics...")
    test_metrics = compute_test_metrics(
        targets,
        predictions,
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