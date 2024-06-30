import os
from pydub import AudioSegment
import numpy as np
from deep_audio_features.bin import basic_test as btest

def process_file(file_path: str, model_path: str):
    # Get the base path for the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Correctly build the absolute path to the model
    # Assuming your script is inside a subdirectory of the root project folder
    root_directory = os.path.dirname(script_directory)
    
      # Move one directory level up to the project root
    absolute_model_path = os.path.join(root_directory, model_path)

    # Ensure to construct the absolute path for the audio file similarly
    absolute_file_path = os.path.join(root_directory, file_path)

    # Load audio from the absolute file path
    audio = AudioSegment.from_wav(absolute_file_path)
    duration = 10000  # Duration for each segment in milliseconds
    segments = [audio[:duration], audio[duration:duration*2], audio[duration*2:]]
    embeddings = []

    # Temporary directory for storing intermediate files, relative to the root directory
    temp_dir = os.path.join(root_directory, 'temp_embs/')
    os.makedirs(temp_dir, exist_ok=True)

    for i, segment in enumerate(segments):
        temp_path = os.path.join(temp_dir, f"temp_emb_{i}.wav")
        segment.export(temp_path, format="wav")

        # Generate embedding using the model
        _, embedding = btest.test_model(absolute_model_path, temp_path, layers_dropped=1, test_segmentation=False)
        embeddings.append(embedding.reshape(-1))

        # Remove the temporary file
        os.remove(temp_path)

    concat_emb = np.concatenate(embeddings, axis=None)
    return concat_emb
