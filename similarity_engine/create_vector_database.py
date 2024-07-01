import json
import logging
import os

from vector_database_setup import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path=None):
    if config_path is None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dir_path, "config.json")

    with open(config_path, "r") as file:
        return json.load(file)


def main():
    config = load_config()

    vector_db = VectorDatabase(config, create_index=True, load_vectors=True)

    # Insert metadata about the vectors
    # NOTE: track_ids must correspond one by one to the vectors guys :D
    vector_db.insert_metadata()

    # Save the index to the specified path in the config json
    vector_db.save_index()

    logger.info("Index creation and metadata insertion completed successfully.")


if __name__ == "__main__":
    main()
