import json

from create_vector_database import VectorDatabase


def load_config(config_path="config.json"):
    with open(config_path, "r") as file:
        return json.load(file)


def main():
    config = load_config()

    vector_db = VectorDatabase(config)

    # random placeholder for now, will fill with normal track ids later
    track_ids = list(range(len(vector_db.vectors)))

    # Insert metadata about the vectors
    # NOTE: track_ids must correspond one by one to the vectors guys :D
    vector_db.insert_metadata(track_ids)

    # Save the index to the specified path in the config json
    vector_db.save_index()

    print("Index creation and metadata insertion completed successfully.")


if __name__ == "__main__":
    main()
