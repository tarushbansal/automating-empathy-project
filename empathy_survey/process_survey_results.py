# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import argparse

# ------------------------- IMPLEMENTATION -----------------------------------

def compute_expected_score(ratingA, ratingB):
    return 1 / (1 + 10 ** ((ratingB - ratingA) / 400))

def update_rating(rating, actual_score, expected_score, K):
    return rating + K * (actual_score - expected_score)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", "--learning_rate", type=float, default=1.0)
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
    model_ratings = {model_table["model_id"]: [0] * 3 for model_table in model_tables}
    response_ratings = {}

    # Run ELO rating algorithm on all rated samples
    for item in results:
        id = item["sample"]["id"]
        if id not in response_ratings:
            response_ratings[id] = {"context": item["sample"]["context"], "ratings": {}}
        if item["modelA"] not in response_ratings[id]["ratings"]:
            response_ratings[id]["ratings"][item["modelA"]] = (item["sample"]["responseA"], [0] * 3)
        if item["modelB"] not in response_ratings[id]["ratings"]:
            response_ratings[id]["ratings"][item["modelB"]] = (item["sample"]["responseB"], [0] * 3)

        for i in range(3):
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

            # Rate responses
            responseA_rating = response_ratings[id]["ratings"][item["modelA"]][1][i]
            responseB_rating = response_ratings[id]["ratings"][item["modelB"]][1][i]
            
            expected_score = compute_expected_score(responseA_rating, responseB_rating)
            response_ratings[id]["ratings"][item["modelA"]][1][i] = update_rating(
                responseA_rating, 
                (2 - item["ratings"][i]) / 4, 
                expected_score,
                K=cli_args.learning_rate
            )

            expected_score = compute_expected_score(responseB_rating, responseA_rating)
            response_ratings[id]["ratings"][item["modelB"]][1][i] = update_rating(
                responseB_rating, 
                (item["ratings"][i] + 2) / 4, 
                expected_score,
                K=cli_args.learning_rate
            )
    
    print(f"\n----------------------Model ELO Ratings----------------------------\n")
    for i, dim in enumerate(["Empathy", "Relevance", "Fluency"]):
        print(f"-----------{dim}------------")
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