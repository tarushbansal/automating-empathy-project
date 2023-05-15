# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- IMPLEMENTATION -----------------------------------

def compute_expected_score(ratingA, ratingB):
    return 1 / (1 + 10 ** ((ratingB - ratingA) / 400))

def update_rating(rating, actual_score, expected_score, K):
    return rating + K * (actual_score - expected_score)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", "--learning_rate", type=float, default=32)
    cli_args = parser.parse_args()

    working_dir = os.path.dirname(os.path.abspath(__file__))

    with open(f"{working_dir}/results/survey_results.json") as f:
        results = json.load(f)

    # Load linked model directory
    model_dir = os.path.join(working_dir, "model_data")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Model data directory missing!")

    # Load all model tables from model_data directory
    model_tables = []
    for item in os.listdir(model_dir):
        path = os.path.join(model_dir, item)
        if os.path.isfile(path) and item.endswith(".json"):
            with open(path) as f:
                model_table = json.load(f)
                model_tables.append(model_table)
                model_id = model_table["model_id"]
    
    # Assign default ratings to all models
    model_ratings = {model_table["model_id"]: [1000] * 3 for model_table in model_tables}
    response_ratings = {}

    # Run ELO rating algorithm on all rated samples
    samples = [[], [], []]
    for item in results:
        id = item["sample"]["id"]
        if id not in response_ratings:
            response_ratings[id] = {"context": item["sample"]["context"], "responses": {}, "ratings": []}
        response_ratings[id]["responses"][item["modelA"]] = item["sample"]["responseA"]
        response_ratings[id]["responses"][item["modelB"]] = item["sample"]["responseB"]
        pairwise_rating = {
            "A": item["modelA"], 
            "B": item["modelB"], 
            "ratings": item["ratings"]
        }
        response_ratings[id]["ratings"].append(pairwise_rating)

        for i in range(3):
            samples[i].append(item["ratings"][i])
            # Rate models
            modelA_rating = model_ratings[item["modelA"]][i]
            modelB_rating = model_ratings[item["modelB"]][i]
            
            expected_score = compute_expected_score(modelA_rating, modelB_rating)
            model_ratings[item["modelA"]][i] = update_rating(
                modelA_rating, 
                (2 - item["ratings"][i]) / 4, 
                expected_score,
                K=cli_args.learning_rate
            )

            expected_score = compute_expected_score(modelB_rating, modelA_rating)
            model_ratings[item["modelB"]][i] = update_rating(
                modelB_rating, 
                (item["ratings"][i] + 2) / 4, 
                expected_score,
                K=cli_args.learning_rate
            )
    
    criteria = ["Empathy", "Relevance", "Fluency"]
    df = pd.DataFrame(
        np.array(samples).T, 
        columns=criteria
    )
    corr = df.corr()

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

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
    f.savefig(f"{working_dir}/results/heatmap_colored_correlation_matrix.png")
    
    print(f"\n----------------------Model ELO Ratings----------------------------\n")
    for i, crit in enumerate(criteria):
        print(f"-----------{crit}------------")
        ratings = []
        for model_table in model_tables:
            id, name, version = (
                model_table["model_id"], 
                model_table["model_name"], 
                model_table["version"]
            )
            ratings.append([f"{name}_v{version}", model_ratings[id][i]])
        for item in sorted(ratings, key=lambda x: x[1], reverse=True):
            print(f"{item[0]}: {item[1]}")
        print("")
    
    with open(f"{working_dir}/results/model_ratings.json", "w") as f:
        json.dump(model_ratings, f)
        print(f"All ratings saved at '{working_dir}/results/model_ratings.json'")
    
    with open(f"{working_dir}/results/response_ratings.json", "w") as f:
        json.dump(response_ratings, f)
        print(f"All ratings saved at '{working_dir}/results/response_ratings.json'")