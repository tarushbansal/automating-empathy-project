# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import uuid
import random
import logging
import argparse
from tqdm import tqdm
from stat import S_IREAD

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
    dataset_dir = "/home/tb662/rds/hpc-work/automating-empathy-project/datasets/empathetic_dialogues"
    if "GOLD" not in [model_table["model_name"] for model_table in model_tables]:
        logger.info("Table for target model 'GOLD' not present! Creating new one...")
        targets = json.load(open(f"{dataset_dir}/test/targets.json"))
        gold_table = create_model_table(
            id=str(uuid.uuid4()),
            name="GOLD",
            version=None,
            responses={str(i): target for i, target in enumerate(targets)}
        )
        model_tables.append(gold_table)

    # Load model predictions from specified file
    with open(test_data_path) as f:
        responses = {item["id"]: item["prediction"] for item in json.load(f)}
    logger.info("Loaded and preprocessed new model predictions")

    # Load dialogue contexts
    contexts = json.load(open(f"{dataset_dir}/test/contexts.json"))
    logger.info("Loaded all dialogue contexts")
    
    # Create new model table
    model_id = str(uuid.uuid4())
    create_model_table(
        id=model_id,
        name=cli_args.model_name,
        version=len([True for model_table in model_tables
                     if model_table["name"] == cli_args.model_name]),
        responses=responses
    )

    # Retrieve / Create 'samples' DynamoDB table configured with backend
    sample_table = retrieve_dynamodb_table("samples")
    if sample_table is None:
        table_schema = [{"name": "id", "key_type": "HASH", "type": "S"}]
        sample_table = create_dynamodb_table("samples", table_schema)
        pushed_ids = []
    else:
        pushed_ids = set([item["sample"]["id"] for item in sample_table.scan()["Items"]])

    if len(pushed_ids) != 0:
        # Create pair-wise samples
        logger.info("Creating survey sample pairs for new model predictions...")
        samples = []
        for id in tqdm(pushed_ids):
            for model_table in model_tables:
                response_pair = [
                    (model_id, responses[id]),
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

        # Load new samples into table
        fill_dynamodb_table(sample_table, samples)

# ---------------------------------------------------------------------------
