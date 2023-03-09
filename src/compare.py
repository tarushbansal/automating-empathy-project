# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import sys
import csv
import json
import random
import argparse

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samples", type=int, default=5)
    cli_args = parser.parse_args()

    return cli_args


def main(
    cli_args: argparse.Namespace
) -> None:
    
    text_data = {}
    metric_data = {}
    src = os.path.abspath("/home/tb662/rds/hpc-work")

    for root, _, filenames in os.walk(src):
        if ("test_data.json" in filenames) and ("ignore" not in filenames):
            test_data = json.load(open(f"{root}/test_data.json"))
            model_name = root.replace(src, "")[1:]
            model_name = model_name.replace("automating-empathy-project/", "")
            model_name = model_name.replace(f"/tensorboard_logs/version_", "_v")
            if os.path.isfile(f"{root}/test_metrics.json"):
                metric_data[model_name] = json.load(open(f"{root}/test_metrics.json"))
            for item in test_data:
                id = item["id"]
                if id not in text_data:
                    text_data[id] = {"context": item["context"]}
                text_data[id][model_name] = (
                    item["output"], 
                    item.get("reward_output", float("nan")),
                    item.get("emotion_output", "unk"),
                    item.get("empintent_output", "unk"),
                    item.get("epitome_IP_output", "unk"),
                    item.get("epitome_EX_output", "unk"),
                    item.get("epitome_ER_output", "unk")
                )
    
    metrics = set()
    for model in metric_data:
        for metric in metric_data[model]:
            if metric.endswith("_target"):
                continue
            metrics.add(metric)
    metrics = sorted(metrics)
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(f"{dirname}/metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model"] + list([metric.replace("_output", "") for metric in metrics]))
        for model in metric_data:
            data = [model]
            for metric in metrics:
                data.append(metric_data[model].get(metric, None))
            writer.writerow(data)

    for id in random.sample(list(text_data.keys()), cli_args.num_samples):
        print(f"\n---- Sample ID {id} -----\n")
        context = text_data[id].pop("context")
        for i in range(len(context)):
            if i % 2 == 0:
                print("Speaker:", end=" ")
            else:
                print("Listener:", end=" ")
            print(context[i])
        print("")
        for model in text_data[id]:
            output, reward, emo, intent, IP, EX, ER = text_data[id][model]
            print(f"{model}: {output} " +
                  f"[Emotion: {emo}; EmpIntent: {intent}; IP: {IP}; EX: {EX}; ER: {ER}; Reward: {reward:.3f}]"
            )
        print("")


if __name__ == "__main__":
    # Parse command line arguments
    cli_args = parse_args()

    dirname = os.path.dirname(os.path.abspath(__file__))
    fname = f"{dirname}/results.txt"
    with open(fname, "w") as sys.stdout:
        main(cli_args)  

# -----------------------------------------------------------------------------
