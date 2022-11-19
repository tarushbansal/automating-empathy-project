# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import logging
from stat import S_IREAD
from boto3.dynamodb.conditions import Attr

# User-Defined Modules
from dynamodb_utils import retrieve_dynamodb_table

# ------------------------- IMPLEMENTATION -----------------------------------

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Present backend variable
    COMPLETE = 2

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    samples_table = retrieve_dynamodb_table("samples")
    if samples_table is None:
        raise NotImplementedError("Failed to get ratings! Table 'samples' does not exist.")

    rated_samples = samples_table.scan(FilterExpression=Attr("status").eq(COMPLETE))["Items"]
    logger.info(f"{len(rated_samples)} rated survey samples found")

    if len(rated_samples) != 0:
        dirname = os.path.dirname(os.path.abspath(__file__))
        ratings_fpath = os.path.join(dirname, "ratings.json")
        with open(ratings_fpath, "w") as f:
            json.dump(rated_samples, f)
            logger.info(f"Saved ratings at {ratings_fpath}")
        os.chmod(ratings_fpath, S_IREAD)

# ----------------------------------------------------------------------------
