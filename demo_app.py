import os
from inference_similar_songs import process_audio_to_embeddings, perform_similarity_search, fetch_track_details, load_config, find_embeddings_in_local_path
import warnings
import logging

import streamlit as st

from inference_similar_songs import (
    fetch_track_details,
    load_config,
    perform_similarity_search,
    process_audio_to_embeddings,
)

warnings.filterwarnings("ignore")

# Configure logging
streamlit_logger = logging.getLogger('streamlit')
streamlit_logger.setLevel(logging.ERROR)

# Determine the root directory of the script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the folder where WAV files should be saved
WAV_FILES_DIR = os.path.join(PROJECT_ROOT, "test_wav_files")

# Path to the folder where songs are stored
SONGS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'songs'))

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

if uploaded_file is not None:
    # Save the uploaded WAV file to the test_wav_files directory
    file_path = os.path.join(WAV_FILES_DIR, uploaded_file.name)

    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Display the uploaded audio file to allow playback
    st.audio(file_path, format="audio/wav", start_time=0)

    # Process the audio file to get embeddings
    #embeddings = find_embeddings_in_local_path(file_path)
    embeddings = find_embeddings_in_local_path(file_path)

    if embeddings is None:
        embeddings = process_audio_to_embeddings(file_path, models_path)

    loaded_config = load_config()

    # Perform similarity search
    top_neighbors_indices, _ = perform_similarity_search(embeddings, loaded_config)
    vector_ids = top_neighbors_indices.flatten().tolist()

    # Fetch track details
    if vector_ids:
        track_details = fetch_track_details(vector_ids, loaded_config)

        # Extract the file name from the path
        file_name = os.path.basename(file_path)

        # Remove the file extension
        track_id_str = os.path.splitext(file_name)[0]

        # Remove leading zeros
        track_id = None
        try:
            track_id = int(track_id_str)
        except Exception:
            track_id = track_id_str

        # Remove the query id from the dataframe
        #track_details = track_details[track_details['track_id'] != track_id]

        if not track_details.empty:  # Check if the DataFrame is not empty
            st.header("Top similar tracks:")

            # Iterate through each track detail to load and play the audio file
            for index, row in track_details.iterrows():
                similar_track_id = row['track_id']
                similar_track_path = os.path.join(SONGS_DIR, f"{similar_track_id}.wav")

                # Display track details with headers
                st.subheader(f"{row['title']} - {row['artist']}")
                st.write(f"Album: {row['album']}")
                st.write(f"Genre: {row['genre']}")
                st.write(f"Released: {row['released']}")
                

                # Check if the similar track audio file exists
                if os.path.exists(similar_track_path):
                    st.audio(similar_track_path, format='audio/wav')
                else:
                    st.write(f"Audio file for Track ID {similar_track_id} not found.")
                
                # Add a separator between tracks
                st.markdown("---")
        else:
            st.write("No similar tracks found.")
    else:
        st.write("No results found for the uploaded audio.")
