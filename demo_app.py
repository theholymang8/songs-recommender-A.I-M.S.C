import os
import warnings

import streamlit as st

from inference_similar_songs import (
    fetch_track_details,
    load_config,
    perform_similarity_search,
    process_audio_to_embeddings,
)

warnings.filterwarnings("ignore")


# Determine the root directory of the script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the folder where WAV files should be saved
WAV_FILES_DIR = os.path.join(PROJECT_ROOT, "test_wav_files")

# Ensure the directory exists
os.makedirs(WAV_FILES_DIR, exist_ok=True)

st.title("Audio Similarity Finder")
st.write("Upload a WAV file to find similar tracks based on audio features.")

uploaded_file = st.file_uploader("Choose a WAV file...", type=["wav"])

models_path = [
    os.path.join(PROJECT_ROOT, "models", "genre.pt"),
    os.path.join(PROJECT_ROOT, "models", "instruments.pt"),
    os.path.join(PROJECT_ROOT, "models", "mood.pt"),
]

print(f"MODEL PATH: {models_path}")

if uploaded_file is not None:
    # Save the uploaded WAV file to the test_wav_files directory
    file_path = os.path.join(WAV_FILES_DIR, uploaded_file.name)

    print(f"FILE PATH: {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display the uploaded audio file to allow playback
    st.audio(file_path, format="audio/wav", start_time=0)

    # Process the audio file to get embeddings
    embeddings = process_audio_to_embeddings(file_path, models_path)
    loaded_config = load_config()

    # Perform similarity search
    top_neighbors_indices, _ = perform_similarity_search(embeddings, loaded_config)
    vector_ids = top_neighbors_indices.flatten().tolist()

    # Fetch track details
    if vector_ids:
        track_details = fetch_track_details(vector_ids, loaded_config)
        if not track_details.empty:  # Check if the DataFrame is not empty
            st.write("Top similar tracks:")
            st.write(track_details)
        else:
            st.write("No similar tracks found.")
    else:
        st.write("No results found for the uploaded audio.")
