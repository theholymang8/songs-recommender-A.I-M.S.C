import os

import numpy as np


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
                print(f"Failed to load {file_path}: {e}")

        # Concatenate embeddings from the current subfolder, if any are available
        if list_of_embeddings:
            concatenated_embedding = np.concatenate(list_of_embeddings, axis=None)
            save_file_name = os.path.join(destination_folder, f"{subfolder}.npy")
            np.save(save_file_name, concatenated_embedding)
            print(f"Saved concatenated embeddings to {save_file_name}")
        else:
            print(f"No embeddings files found in {current_folder}")
