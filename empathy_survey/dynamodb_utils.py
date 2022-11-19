# ------------------------- IMPORT MODULES -----------------------------------

# System Modules
import logging
from tqdm import tqdm

import boto3
from botocore.exceptions import ClientError

# ------------------------- IMPLEMENTATION -----------------------------------

# Define resources
logger = logging.getLogger(__name__)
dynamodb = boto3.resource(
    'dynamodb',
    region_name="us-east-1",
    aws_access_key_id="AKIA4O74A5D3WIJCSDHX",
    aws_secret_access_key="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")

# Amazon DynamoDB rejects a get batch larger than 100 items.
MAX_GET_SIZE = 100


def retrieve_dynamodb_table(table_name):
    """
    :param table_name: The name of the table to check.
    :return: Returns table if exists else returns None.
    """
    table = None
    try:
        _table = dynamodb.Table(table_name)
        _table.load()
        logger.info(f"Found table '{table_name}'.")
    except ClientError as err:
        if err.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.info(f"Table '{table_name}' does not exist!")
        else:
            logger.error(
                "Couldn't check for existence of table '%s'. Here's why: %s: %s",
                table_name,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise
    else:
        table = _table
    return table


def create_dynamodb_table(table_name, schema):
    """
    Creates an Amazon DynamoDB table with the specified schema.

    :param table_name: The name of the table.
    :param schema: The schema of the table. The schema defines the format
                   of the keys that identify items in the table.
    :return: The newly created table.
    """
    try:
        logger.info("Creating table '%s'...", table_name)
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{
                'AttributeName': item['name'], 'KeyType': item['key_type']
            } for item in schema],
            AttributeDefinitions=[{
                'AttributeName': item['name'], 'AttributeType': item['type']
            } for item in schema],
            ProvisionedThroughput={'ReadCapacityUnits': 10, 'WriteCapacityUnits': 10}
        )
        table.wait_until_exists()
        logger.info("Created table '%s'.", table.name)
    except ClientError:
        logger.exception("Failed to create table.")
        raise
    else:
        return table


def fill_dynamodb_table(table, table_data):
    """
    Fills an Amazon DynamoDB table with the specified data, using the Boto3
    Table.batch_writer() function to put the items in the table.
    Inside the context manager, Table.batch_writer builds a list of
    requests. On exiting the context manager, Table.batch_writer starts sending
    batches of write requests to Amazon DynamoDB and automatically
    handles chunking, buffering, and retrying.

    :param table: The table to fill.
    :param table_data: The data to put in the table. Each item must contain at least
                       the keys required by the schema that was specified when the
                       table was created.
    """
    try:
        with table.batch_writer() as writer:
            logger.info("Loading data into table '%s'...", table.name)
            for item in tqdm(table_data):
                writer.put_item(Item=item)
            logger.info("Successfully loaded data into table '%s'.", table.name)
    except ClientError:
        logger.exception("Couldn't load data into table '%s'.", table.name)
        raise

# ----------------------------------------------------------------------------
