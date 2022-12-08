# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import uuid
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from stat import S_IREAD
from typing import List, Dict

# User-Defined Modules
from dynamodb_utils import (
    retrieve_dynamodb_table,
    fill_dynamodb_table,
    create_dynamodb_table
)

# ------------------------- IMPLEMENTATION -----------------------------------

logger = logging.getLogger(__name__)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load linked model directory
model_dir = os.path.join(working_dir, "model_data")
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    logger.info(f"Created new directory to store model tables at {model_dir}")

# Preset backend sample status codes
backend_config = f"{working_dir}/backend_config.json"
if not os.path.isfile(backend_config):
    raise FileNotFoundError("File 'backend_config.json' does not exist in working directory")
with open(backend_config) as f:
    try:
        PENDING = json.load(f)["status_codes"]["PENDING"]
    except KeyError:
        raise KeyError("Status code 'PENDING' not specified in 'backend_config.json'")


def create_model_table(id, name, version, responses):
    table = {
        "model_id": id,
        "model_name": name,
        "version": version,
        "responses": responses
    }
    fpath = os.path.join(model_dir, f"{id}.json")
    with open(fpath, "w") as f:
        json.dump(table, f)
        logger.info(f"Created new model table '{name}' at {fpath}")
    os.chmod(fpath, S_IREAD)

    return table


def preprocess_test_data(data: List[Dict]):
    # Set seed important to generate same samples and creat sample pairs
    # If new seed is set or number of samples are increased, it is
    # likely model responses will only be compared with 'GOLD' targets
    random.seed(0)
    samples = random.sample(data, k=100)
    data = {
        sample["id"]: {
            "context": sample["context"],
            "response": sample["prediction"]
        } for sample in samples
    }
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, required=True)
    parser.add_argument("--model_dir", type=str, default=None, required=True)
    cli_args = parser.parse_args()

    # Sanity checks for command line arguments
    if cli_args.model_name == "GOLD":
        raise ValueError(
            "Model name 'GOLD' is reserved for target dataset responses")
    dir = os.path.abspath(cli_args.model_dir)
    if os.path.isdir(dir):
        test_data_path = os.path.join(dir, "test_data.json")
        if not os.path.isfile(test_data_path):
            raise FileNotFoundError("Test data does not exist at the specified model directory!")
    else:
        raise FileNotFoundError("Specified model directory does not exist!")

    # Load all model tables from model_data directory
    model_tables = []
    for item in os.listdir(model_dir):
        path = os.path.join(model_dir, item)
        if os.path.isfile(path) and item.endswith(".json"):
            with open(path) as f:
                model_table = json.load(f)
                model_tables.append(model_table)
                model_id = model_table["model_id"]
                logger.info(f"Discovered model table with id {model_id}")

    # Create GOLD table if not present
    dataset_dir = "/home/tb662/rds/hpc-work/automating-empathy-project/empdial_dataset/test"
    if "GOLD" not in [model_table["model_name"] for model_table in model_tables]:
        logger.info("Table for target model 'GOLD' not present! Creating new one...")
        targets = np.load(f"{dataset_dir}/targets.npy", allow_pickle=True)
        gold_table = create_model_table(
            id=str(uuid.uuid4()),
            name="GOLD",
            version=None,
            responses={str(i): target for i, target in enumerate(targets)}
        )
        model_tables.append(gold_table)

    # Load model predictions from specified file
    with open(test_data_path) as f:
        test_data = json.load(f)
    test_data = preprocess_test_data(test_data)
    logger.info("Loaded and preprocessed new model predictions")

    # Load dialogue contexts
    contexts = np.load(f"{dataset_dir}/contexts.npy", allow_pickle=True)
    logger.info("Loaded all dialogue contexts")

    # Assign defaults to new model table
    new_model_version = 0
    new_model_id = str(uuid.uuid4())

    # Create pair-wise samples
    logger.info("Creating survey sample pairs for new model predictions...")
    samples = []
    for id in tqdm(test_data):
        for model_table in model_tables:
            if model_table["model_name"] == cli_args.model_name:
                new_model_version = model_table["version"] + 1
            if str(id) in model_table["responses"]:
                response_pair = [
                    (new_model_id, test_data[id]["response"]),
                    (model_table["model_id"], model_table["responses"][str(id)])
                ]
                random.seed()
                random.shuffle(response_pair)
                sample = {
                    "id": str(uuid.uuid4()),
                    "modelA": response_pair[0][0],
                    "modelB": response_pair[1][0],
                    "sample": {
                        "id": id,
                        "context": contexts[id],
                        "responseA": response_pair[0][1],
                        "responseB": response_pair[1][1]},
                    "status": PENDING
                }
                samples.append(sample)
    logger.info(f"Created {len(samples)} new survey samples")

    if len(samples) != 0:
        # Retrieve / Create 'samples' DynamoDB table configured with backend
        sample_table = retrieve_dynamodb_table("samples")
        if sample_table is None:
            table_schema = [{"name": "id", "key_type": "HASH", "type": "str"}]
            sample_table = create_dynamodb_table("samples", table_schema)

        # Load new samples into table
        fill_dynamodb_table(sample_table, samples)

        # Create new model table
        create_model_table(
            id=new_model_id,
            name=cli_args.model_name,
            version=new_model_version,
            responses={str(id): test_data[id]["response"] for id in test_data}
        )

# ---------------------------------------------------------------------------
