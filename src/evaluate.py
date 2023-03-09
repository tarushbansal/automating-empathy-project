# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import random
import argparse
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
            data[id]["models"][model_name] = {"output": item["output"]}
    
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
            ratings = [item["models"][model][crit] for item in eval_data]
            scores[model][crit] = sum(ratings) / len(ratings)

    with open(f"{output_dir}/model_scores.json", "w") as f:
        json.dump(scores, f)

    print(f"All human evaluation results saved at directory '{output_dir}'\n")

    for crit in criteria:
        print(f"----- {crit} Scores -----")
        for model, score in sorted(
            [(model, scores[model][crit]) for model in scores], 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"{model}: {score:.3f}")
        print("")


if __name__ == "__main__":
    main()  

# -----------------------------------------------------------------------------
