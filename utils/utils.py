import os

import numpy as np
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def concatenate_segmented_embeddings(
    embeddings_folder: str, destination_folder: str
) -> None:
    """
    Reads embeddings from subfolders within a specified directory, concatenates them,
    and saves the concatenated embeddings into a new file in a destination directory.

    Args:
    embeddings_folder (str): The path to the folder containing subfolders of embeddings.
    destination_folder (str): The path to the folder where concatenated embeddings will be saved.
    """
    # Ensure the destination directory exists, create if it doesn't
    os.makedirs(destination_folder, exist_ok=True)

    # List all subfolders in the specified embeddings directory
    embeddings_subfolders = os.listdir(embeddings_folder)

    # Iterate through each subfolder to process embeddings
    for subfolder in embeddings_subfolders:
        current_folder = os.path.join(embeddings_folder, subfolder)

        # Continue only if the current item is a directory
        if not os.path.isdir(current_folder):
            continue

        # List all files in the current subfolder
        embeddings_files = os.listdir(current_folder)

        list_of_embeddings = []

        # Load each file, convert to float32, and collect all embeddings
        for file_ in embeddings_files:
            file_path = os.path.join(current_folder, file_)
            try:
                embedding = np.load(file_path).astype("float32")
                list_of_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        # Concatenate embeddings from the current subfolder, if any are available
        if list_of_embeddings:
            concatenated_embedding = np.concatenate(list_of_embeddings, axis=None)
            save_file_name = os.path.join(destination_folder, f"{subfolder}.npy")
            np.save(save_file_name, concatenated_embedding)
            logger.info(f"Saved concatenated embeddings to {save_file_name}")
        else:
            logger.error(f"No embeddings files found in {current_folder}")


def find_search_query_from_saved_embeddings(audio_file_name: str):

    # Define the base path for saved embeddings
    SAVED_EMBEDDINGS_PATH_GENRE = os.path.abspath("./concatenated_embeddings/genre-classification/")
    SAVED_EMBEDDINGS_PATH_INSTRUMENT = os.path.abspath("./concatenated_embeddings/instrument-classification/")
    SAVED_EMBEDDINGS_PATH_EMOTION = os.path.abspath("./concatenated_embeddings/emotion-classification/")

    SAVED_EMBEDDINGS_PATHS = {
        "genre": SAVED_EMBEDDINGS_PATH_GENRE,
        "instrument": SAVED_EMBEDDINGS_PATH_INSTRUMENT,
        "emotion": SAVED_EMBEDDINGS_PATH_EMOTION
    }
    
    # print(f"EMBEDDINGS FOLDER: {SAVED_EMBEDDINGS_PATH}")
    
    # Extract the track_id from the audio file name
    track_id_str = os.path.splitext(os.path.basename(audio_file_name))[0]
    

    # Remove leading zeros from the track_id
    track_id = None
    try:
        track_id = int(track_id_str)
    except Exception:
        track_id = track_id_str

    # Construct the .npy file name without leading zeros
    npy_file_name = f"{track_id}.npy"



    list_of_embeddings = []

    # Iterate over each classification type
    for _, path in SAVED_EMBEDDINGS_PATHS.items():
        files = os.listdir(path)
        if npy_file_name in files:
            npy_file_path = os.path.join(path, npy_file_name)
            try:
                query = np.load(npy_file_path).astype('float32') 
                list_of_embeddings.append(query)
            except Exception as e:
                logger.error(f"ERROR: File: {npy_file_path} could not be loaded due to error: {e}")

    if len(list_of_embeddings) == 3:
        return np.concatenate(list_of_embeddings, axis=None)

    return None

