# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import sys
import json
import random
import argparse

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--dump", action="store_true")
    cli_args = parser.parse_args()

    return cli_args


def main(
    cli_args: argparse.Namespace
) -> None:
    print(f"\n----- Reward Statistics (mean, var) -----\n")
    comparison_data = {}
    src = os.path.abspath("/home/tb662/rds/hpc-work")

    for root, _, filenames in os.walk(src):
        if ("test_data.json" in filenames) and ("ignore" not in filenames):
            test_data = json.load(open(f"{root}/test_data.json"))
            model_name = root.replace(src, "")[1:]
            model_name = model_name.replace("automating-empathy-project/", "")
            model_name = model_name.replace(f"/tensorboard_logs/version_", "_v")
            rewards = None
            if os.path.isfile(f"{root}/rewards.json"):
                rewards = json.load(open(f"{root}/rewards.json"))
                mean_reward, var_reward = rewards["mean"], rewards["var"]
                print(f"{model_name}: ({mean_reward:.3f}, {var_reward:.3f})")
            for item in test_data:
                id, context, output = item["id"], item["context"], item["output"]
                reward = float("nan") if rewards is None else rewards["rewards"][str(id)]
                if id not in comparison_data:
                    comparison_data[id] = {"context": context}
                comparison_data[id][model_name] = (output, reward)
    
    for id in list(comparison_data.keys()):
        if len(comparison_data[id].keys()) <= 2:
            del comparison_data[id]

    for id in random.sample(list(comparison_data.keys()), cli_args.num_samples):
        print(f"\n---- Sample ID {id} -----\n")
        context = comparison_data[id].pop("context")
        for i in range(len(context)):
            if i % 2 == 0:
                print("Speaker:", end=" ")
            else:
                print("Listener:", end=" ")
            print(context[i])
        print("")
        for model in comparison_data[id]:
            output, reward = comparison_data[id][model]
            print(f"{model}: {output} [Reward = {reward:.3f}]")
        print("")


if __name__ == "__main__":
    # Parse command line arguments
    cli_args = parse_args()

    if cli_args.dump:
        dirname = os.path.dirname(os.path.abspath(__file__))
        fname = f"{dirname}/results.txt"
        with open(fname, "w") as sys.stdout:
            main(cli_args)
    else:
        main(cli_args)
        

# -----------------------------------------------------------------------------
