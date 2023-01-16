# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import os
import json
import logging
from decimal import Decimal
from boto3.dynamodb.conditions import Attr

# User-Defined Modules
from dynamodb_utils import retrieve_dynamodb_table

# ------------------------- IMPLEMENTATION -----------------------------------

logger = logging.getLogger(__name__)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Preset backend sample status codes
backend_config = f"{working_dir}/backend_config.json"
if not os.path.isfile(backend_config):
    raise FileNotFoundError("File 'backend_config.json' does not exist in working directory")
with open(backend_config) as f:
    try:
        COMPLETE = json.load(f)["status_codes"]["COMPLETE"]
    except KeyError:
        raise KeyError("Status code 'COMPLETE' not specified in 'backend_config.json'")


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    samples_table = retrieve_dynamodb_table("samples")
    if samples_table is None:
        raise NotImplementedError("Failed to get results! Table 'samples' does not exist.")

    results = samples_table.scan(FilterExpression=Attr("status").eq(COMPLETE))["Items"]
    logger.info(f"{len(results)} rated survey samples found")

    if len(results) != 0:
        dirname = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(dirname, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/survey_results.json", "w") as f:
            json.dump(results, f, cls=DecimalEncoder)
            logger.info(f"Saved results at '{results_dir}/survey_results.json'")

# ----------------------------------------------------------------------------
