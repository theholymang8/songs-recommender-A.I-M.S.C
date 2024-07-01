import logging
import os
import sys

import faiss
import numpy as np
import pandas as pd
import torch
from deep_audio_features.models.cnn import load_cnn
from sqlalchemy import create_engine

# Assuming the script is run from within the root directory of the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from similarity_engine.similarity_search import SimilaritySearch, load_config
from utils.audio_utils import process_file, process_file_custom

# Configure logging
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def fetch_track_details(vector_ids, config):
    """
    Fetch track details for given vector IDs from the database and print them.

    Args:
    vector_ids (list of int): List of vector IDs from the similarity search.
    config (dict): Database configuration details.

    Returns:
    None: This function prints the query results.
    """
    # Create database engine
    db_config = config["database"]
    engine = create_engine(
        f"mysql+mysqldb://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    # Create placeholders for each vector ID
    placeholders = ", ".join(["%s"] * len(vector_ids))

    # Define the SQL query with placeholders
    query = f"""
    SELECT t.track_id, t.title, a.name as artist, YEAR(t.date_created) as released, al.title as album, t.genre_top as genre
    FROM tracks t
    INNER JOIN vector_metadata vm ON t.track_id = vm.track_id
    INNER JOIN artists a ON t.track_id = a.track_id
    INNER JOIN albums al ON t.track_id = al.track_id
    WHERE vm.vector_id IN ({placeholders});
    """

    # Execute the query
    results = pd.read_sql_query(query, engine, params=tuple(vector_ids))

    # Check if results are empty
    if results.empty:
        print("No tracks found for the given vector IDs.")
        return

    # Print results in a formatted manner
    print("Track Details:\n")
    print(results.to_string(index=False))

    # Close the connection
    engine.dispose()

    return results


def process_audio_to_embeddings(audio_file_path, model_paths):
    """
    Process an audio file to generate concatenated embeddings.

    Args:
    audio_file_path (str): Path to the audio file.
    model_path (str): Path to the machine learning model used for processing.

    Returns:
    numpy.array: Concatenated embeddings from the processed audio file.
    """

    embeddings_from_inference = []
    for model in model_paths:
        # Generate embeddings from the audio file
        embeddings = process_file(audio_file_path, model)
        embeddings_from_inference.append(embeddings)
    concatenated_inference_embedding = np.concatenate(
        embeddings_from_inference, axis=None
    )

    return concatenated_inference_embedding


def process_audio_to_embeddings_model(audio_file_path, models):
    concatenated_inference_embedding = process_file_custom(audio_file_path, models)

    return concatenated_inference_embedding


def load_embeddings(file_path):
    """
    Load precomputed embeddings from a file.

    Args:
    file_path (str): Path to the numpy file containing embeddings.

    Returns:
    numpy.array: Loaded embeddings.
    """
    return np.load(file_path).astype("float32")


def perform_similarity_search(embedding, config):
    """
    Initialize the similarity search engine and find top similar items.

    Args:
    embedding (numpy.array): Query embedding for similarity search.
    config (dict): Configuration for the similarity search engine.

    Returns:
    tuple: Indices and distances of top similar items.
    """
    v_db = SimilaritySearch(config)
    return v_db.find_similar_embeddings(embedding, top_k=6)


def main():
    model_path = "./models/genre.pt"  # Model path, update if necessary
    audio_file_path = (
        "./test_wav_files/000574.wav"  # Audio file path, update if necessary
    )
    save_path = "./test_wav_files/test_query_574_IVFF.npy"  # Path to save embeddings, update if necessary

    # Load configuration
    loaded_config = load_config()

    model_paths = {
        "genre": "./models/genre.pt",
        "instrument": "./models/instruments.pt",
        "emotion": "./models/mood.pt",
    }

    models = {
        "genre": {
            "properties": {
                "model": None,
                "hop_length": None,
                "window_length": None,
                "max_seq_length": None,
                "zero_pad": None,
                "spec_size": None,
                "fuse": None,
            }
        },
        "instrument": {
            "properties": {
                "model": None,
                "hop_length": None,
                "window_length": None,
                "max_seq_length": None,
                "zero_pad": None,
                "spec_size": None,
                "fuse": None,
            }
        },
        "emotion": {
            "properties": {
                "model": None,
                "hop_length": None,
                "window_length": None,
                "max_seq_length": None,
                "zero_pad": None,
                "spec_size": None,
                "fuse": None,
            }
        },
    }

    LAYERS_DROPPED = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for key, model_path in model_paths.items():
        model, hop_length, window_length = load_cnn(model_path)
        model = model.to(device)
        models[key]["properties"]["model"] = model
        models[key]["properties"]["hop_length"] = hop_length
        models[key]["properties"]["window_length"] = window_length
        models[key]["properties"]["max_seq_length"] = model.max_sequence_length
        models[key]["properties"]["zero_pad"] = model.zero_pad
        models[key]["properties"]["fuse"] = model.fuse

    # Process and save embeddings
    test_query = process_audio_to_embeddings_model(audio_file_path, models=models)

    # test_query = load_embeddings(save_path)

    if loaded_config["faiss"]["index_type"] == "Cosine":
        faiss.normalize_L2(test_query)

    # Load precomputed embeddings for testing
    # test_query_embedding = "./test_wav_files/test_query.npy"
    # loaded_embedding = load_embeddings(test_q368y)

    # np.save("./test_wav_files/"+ "test_query_574_IVFF.npy", test_query)

    # Perform similarity search and print results
    top_neighbors_indices, _ = perform_similarity_search(test_query, loaded_config)

    # Assuming top_neighbors_indices returns a list of indices
    vector_ids = top_neighbors_indices.flatten().tolist()

    # Fetch and display track details from the database
    _ = fetch_track_details(vector_ids, loaded_config)


if __name__ == "__main__":
    main()
