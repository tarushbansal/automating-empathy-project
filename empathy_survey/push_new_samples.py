# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import uuid
import random
import logging
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_per_model", type=int, default=None, required=True)
    cli_args = parser.parse_args()

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

    # Load dialogue contexts
    dataset_dir = "/home/tb662/rds/hpc-work/automating-empathy-project/datasets/empathetic_dialogues"
    contexts = json.load(open(f"{dataset_dir}/test/contexts.json"))
    logger.info("Loaded all dialogue contexts")

    # Retrieve / Create 'samples' DynamoDB table configured with backend
    sample_table = retrieve_dynamodb_table("samples")
    if sample_table is None:
        table_schema = [{"name": "id", "key_type": "HASH", "type": "str"}]
        sample_table = create_dynamodb_table("samples", table_schema)
        new_response_ids = random.sample(range(len(contexts)), k=cli_args.num_per_model)
    else:
        new_response_ids = range(len(contexts))
        for item in sample_table.scan()["Items"]:
            new_response_ids.pop(item["sample"]["id"])
        if len(new_response_ids) == 0:
            raise ValueError("All unique model responses have been pushed!")
        if len(new_response_ids) < cli_args.num_per_model:
            logger.info(
                f"USER WARNING: Only {len(new_response_ids)} unique responses ", 
                "per model are yet to be pushed!")
        new_response_ids = random.sample(new_response_ids, k=cli_args.num_per_model)

    # Create pair-wise samples
    logger.info("Creating survey sample pairs for all model predictions...")
    samples = []
    for id in new_response_ids:
        for i, model_table_A in enumerate(model_tables):
            for model_table_B in model_tables[i+1:]:
                response_pair = [
                    (model_table_A["model_id"], model_table_A["responses"][str(id)]),
                    (model_table_B["model_id"], model_table_B["responses"][str(id)])
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
        # Load new samples into table
        fill_dynamodb_table(sample_table, samples)

# ---------------------------------------------------------------------------