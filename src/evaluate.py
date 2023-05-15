# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Union

# ------------------------- IMPLEMENTATION -----------------------------------


def initialise_interface() -> None:
    os.system("clear")
    print("------- Welcome to this interface to conduct a human evaluation of automated dialogue models! -------")
    print("")
    print("In each turn you will be presented with a randomly sampled dialogue context "
          "and a series of human / dialogue model generated responses which you need to "
          "rate according to the following criteria:")
    print("")
    print("1) Empathy: How appropriately the response explores, expresses, and interpets feelings and emotions.")
    print("2) Specificity: How specific the response is with respect to the dialogue context.")
    print("")
    print("NOTE-A: that the definition of 'How appropriately', although subjective, should be best interpreted "
          "as which response demonstrates the respective criterion as and when required given the "
          "dialogue context i.e., always 'exploring', 'reacting' or showing 'interpretation' is not "
          "desirable in an empathetic context.")
    print("NOTE-B: Please provide the ratings on a scale of 0 to 5 (5 being the highest) "
          "separated by commas in the order the responses are displayed "
          "For instance: '5,3,2' for the first, second, and third response respectively. "
          "All spaces in the input will be ignored.")
    print("\n")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dirs", type=str, nargs="+")
    parser.add_argument("--model_names", type=str, nargs="+")
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--num_samples", type=int, default=30)
    cli_args = parser.parse_args()

    return cli_args


def load_data(
    model_dirs: List[str], 
    model_names: List[str]
) -> Dict[str, Union[int, List[str], str]]:
    
    assert len(model_dirs) == len(model_names)
    assert len(set(model_names)) == len(model_names)
    
    data = {}
    for model_dir, model_name in zip(model_dirs, model_names):
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError("Specified model directory does not exist!")
        fnames = os.listdir(model_dir)
        if ("test_data.json" not in fnames):
            raise FileNotFoundError("Test data does not exist in specified model directory!")
        test_data = json.load(open(f"{model_dir}/test_data.json"))
        for item in test_data:
            id = item["id"]
            if id not in data:
                data[id] = {"context": item["context"], "models": {}}
            data[id]["models"][model_name] = {
                "output": item["output"],
                "diff_ER": (item["epitome_ER_output"] - item["epitome_ER_target"]) ** 2,
                "diff_EX": (item["epitome_EX_output"] - item["epitome_EX_target"]) ** 2,
                "diff_IP": (item["epitome_IP_output"] - item["epitome_IP_target"]) ** 2,
                "reward": item["reward_output"]
            }
    
    return data


def get_rating(type: str, length: int) -> List[int]:
    try:
        rating = [
            int(char) for char in 
            input(f"{type} Rating: ").replace(" ", "").split(",")
        ]
        if (len(rating) == length) and all([r <= 5 for r in rating]):
            return rating
        else:
            raise AssertionError
    except (AssertionError, ValueError):
        print("Invalid input! Refer to NOTE-B in the header.\n")
        return get_rating(type, length)


def main() -> None:

    # Parse command line arguments
    cli_args = parse_args()
    
    output_dir = os.path.abspath(cli_args.output_dir)
    if cli_args.reuse_cache:
        if os.path.isfile(f"{output_dir}/human_eval.json") and os.path.isfile(f"{output_dir}/model_paths.json"):
            with open(f"{output_dir}/model_paths.json") as f:
                model_paths = json.load(f)
                model_dirs, model_names = list(model_paths.values()), list(model_paths.keys())
                data = load_data(model_dirs, model_names)
            with open(f"{output_dir}/human_eval.json") as f:
                eval_data = json.load(f)
                initial_len = len(eval_data)
                for item in eval_data:
                    data.pop(item["id"])
            print("Successfully loaded cached evaluation data.\n")
        else:
            raise FileNotFoundError(
                "No cached evaluation data found at the specified output directory!")
    else:
        eval_data = []
        initial_len = 0
        model_dirs, model_names = cli_args.model_dirs, cli_args.model_names
        data = load_data(model_dirs, model_names)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model_paths.json", "w") as f:
            model_paths = {
                name: os.path.abspath(path)
                for path, name in zip(model_dirs, model_names)
            }
            json.dump(model_paths, f) 
    data = [{"id": k, **v} for k, v in data.items()]

    initialise_interface()
    criteria = ["Empathy", "Specificity"]
    while len(eval_data) < initial_len + cli_args.num_samples:
        print(f"----- Sample {len(eval_data) + 1}/{initial_len + cli_args.num_samples} -----")
        sample = data.pop(random.choice(range(len(data))))
        for i, utt in enumerate(sample["context"]):
            if i % 2 == 0:
                print("Speaker:", end=" ")
            else:
                print("Listener:", end=" ")
            print(utt)
        print("")
        if input("Leave empty to accept sample: ") != "":
            print("")
            continue
        print("")
        models = list(sample["models"].keys())
        random.shuffle(models)
        for i, model in enumerate(models):
            ind = chr(ord("A") + i)
            output = sample["models"][model]["output"]
            print(f"Model {ind}: {output}")
        print("")
        ratings = []
        for crit in criteria:
            rating = get_rating(crit, len(models))
            ratings.append(rating)
        for crit, rating in zip(criteria, ratings):
            for i in range(len(rating)):
                sample["models"][models[i]][crit] = rating[i]
        eval_data.append(sample)
        with open(f"{output_dir}/human_eval.json", "w") as f:
            json.dump(eval_data, f) 
        print("\n")

    os.system("clear")
    scores = {model: {crit: [] for crit in criteria} for model in model_names}
    for model in scores:
        for crit in criteria:
            ratings = np.array([item["models"][model][crit] for item in eval_data])
            scores[model][crit] = {
                "mean": np.mean(ratings),
                "var": np.var(ratings)
            }

    with open(f"{output_dir}/model_scores.json", "w") as f:
        json.dump(scores, f)
    
    diff_ER, diff_EX, diff_IP, reward, empathy, specificity = ([] for _ in range(6))
    for item in eval_data:
        for model in item["models"]:
            diff_ER.append(item["models"][model]["diff_ER"] if model != "GOLD" else None)
            diff_EX.append(item["models"][model]["diff_EX"] if model != "GOLD" else None)
            diff_IP.append(item["models"][model]["diff_IP"] if model != "GOLD" else None)
            reward.append(item["models"][model]["reward"])
            empathy.append(item["models"][model]["Empathy"])
            specificity.append(item["models"][model]["Specificity"])
    df = pd.DataFrame(
        np.array([diff_ER, diff_EX, diff_IP, reward, empathy, specificity]).T, 
        columns=["diff_ER", "diff_EX", "diff_IP", "reward", "empathy", "specificity"]
    )
    corr = df.corr(numeric_only=False)
    
    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, _ = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # More details at https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        corr,          # The data to plot
        mask=mask,     # Mask some cells
        cmap=cmap,     # What colors to plot the heatmap as
        annot=True,    # Should the values be plotted in the cells?
        vmax=1,       # The maximum value of the legend. All higher vals will be same color
        vmin=-1,      # The minimum value of the legend. All lower vals will be same color
        center=0,      # The center value of the legend. With divergent cmap, where white is
        square=True,   # Force cells to be square
        linewidths=.5, # Width of lines that divide cells
        cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
    )

    # You can save this as a png with
    f.savefig(f"{output_dir}/heatmap_colored_correlation_matrix.png")

    print(f"All human evaluation results saved at directory '{output_dir}'\n")

    for crit in criteria:
        print(f"----- {crit} Scores -----")
        for model, score in sorted(
            [(model, scores[model][crit]) for model in scores], 
            key=lambda x: x[1]["mean"], 
            reverse=True
        ):
            mean = score["mean"]
            print(f"{model}: {mean:.3f}")
        print("")
    
    print("Pearson's Correlation Matrix:\n")
    print(corr)


if __name__ == "__main__":
    main()  

# -----------------------------------------------------------------------------
