import sys
import os

# Assuming the script is run from within the root directory of the project
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.audio_utils import process_file


from pydub import AudioSegment
import os
import numpy as np
from deep_audio_features.bin import basic_test as btest
from similarity_engine.similarity_search import SimilaritySearch, load_config
from utils.audio_utils import process_file
from sqlalchemy import create_engine
import pandas as pd


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
    db_config = config['database']
    engine = create_engine(f"mysql+mysqldb://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

    # Create placeholders for each vector ID
    placeholders = ', '.join(['%s'] * len(vector_ids))

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

def process_audio_to_embeddings(audio_file_path, model_path):
    """
    Process an audio file to generate concatenated embeddings.

    Args:
    audio_file_path (str): Path to the audio file.
    model_path (str): Path to the machine learning model used for processing.

    Returns:
    numpy.array: Concatenated embeddings from the processed audio file.
    """
    # Generate embeddings from the audio file
    embeddings = process_file(audio_file_path, model_path)
    return embeddings

def load_embeddings(file_path):
    """
    Load precomputed embeddings from a file.
    
    Args:
    file_path (str): Path to the numpy file containing embeddings.

    Returns:
    numpy.array: Loaded embeddings.
    """
    return np.load(file_path).astype('float32')

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
    return v_db.find_similar_embeddings(embedding)

def main():

    model_path = "./models/genre.pt"  # Model path, update if necessary
    audio_file_path = "./test_wav_files/000182.wav"  # Audio file path, update if necessary
    save_path = "../test_wav_files/test_query.npy"  # Path to save embeddings, update if necessary

    # Load configuration
    loaded_config = load_config()

    # Process and save embeddings
    test_query = process_audio_to_embeddings(audio_file_path, model_path)

    # Load precomputed embeddings for testing
    # test_query_embedding = "./test_wav_files/test_query.npy"
    # loaded_embedding = load_embeddings(test_query)

    #np.save("./test_wav_files/"+ "test_query_182.npy", test_query)
    

    # Perform similarity search and print results
    top_neighbors_indices, _ = perform_similarity_search(test_query, loaded_config)

    # Assuming top_neighbors_indices returns a list of indices
    vector_ids = top_neighbors_indices.flatten().tolist()

    # Fetch and display track details from the database
    fetch_track_details(vector_ids, loaded_config)
    

    

if __name__ == '__main__':
    main()
