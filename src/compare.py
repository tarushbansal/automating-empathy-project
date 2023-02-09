# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import random
import argparse

# ------------------------- IMPLEMENTATION -----------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    cli_args = parser.parse_args()

    return cli_args


def main():
    # Parse command line arguments
    cli_args = parse_args()

    print(f"\n-----Mean Rewards-----\n")
    comparison_data = {}
    src = os.path.abspath("/home/tb662/rds/hpc-work")

    for root, _, filenames in os.walk(src):
        if ("rewards.json" in filenames) and ("test_data.json" in filenames):
            test_data = json.load(open(f"{root}/test_data.json"))
            rewards = json.load(open(f"{root}/rewards.json"))
            mean_reward = rewards["mean"]
            model_name = root.replace(src, "")[1:]
            model_name = model_name.replace("automating-empathy-project/", "")
            model_name = model_name.replace(f"/tensorboard_logs/version_", "_v")
            print(f"{model_name}: {mean_reward:.3f}")
            for item, reward in list(zip(test_data, rewards["rewards"].values())):
                id, context, response = item["id"], item["context"], item["prediction"]
                if id not in comparison_data:
                    comparison_data[id] = {"context": context}
                comparison_data[id][model_name] = (response, reward)
    
    for id in random.sample(list(comparison_data.keys()), cli_args.num_samples):
        print(f"\n----Sample ID {id}-----\n")
        context = comparison_data[id].pop("context")
        for i in range(len(context)):
            if i % 2 == 0:
                print("Speaker:", end=" ")
            else:
                print("Listener:", end=" ")
            print(context[i])
        print("")
        for model in sorted(comparison_data[id], key = lambda x: comparison_data[id][x][1], reverse=True):
            response, reward = comparison_data[id][model]
            print(f"{model}: {response} [Reward = {reward:.3f}]")
        print("")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
