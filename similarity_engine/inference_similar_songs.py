import sys
import os

# Assuming your script is located in the "core" directory and you need to go up two levels to include the "similarity_engine"
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(os.path.dirname(script_dir))  # Navigate two levels up to the project root
sys.path.append("./")  # Add the project root to the sys.path


from pydub import AudioSegment
import os
import numpy as np
from deep_audio_features.bin import basic_test as btest
from similarity_search import SimilaritySearch, load_config

# import warnings
# warnings.filterwarnings("ignore")


def process_file(file_path: str, model_path: str):
    concatenated_embeddings = {}
    audio = AudioSegment.from_wav(file_path)
    duration = 10000  # Duration for each segment in milliseconds
    segments = [audio[:duration], audio[duration:duration*2], audio[duration*2:]]
    embeddings = []

    for i, segment in enumerate(segments):
        # Temporarily export segment to handle with the model
        temp_path = os.path.join("../temp_embs/", f"temp_emb_{i}.wav")
        segment.export(temp_path, format="wav")

        # Generate embedding
        _, embedding = btest.test_model(model_path, temp_path, layers_dropped=1, test_segmentation=False)
        embeddings.append(embedding.reshape(-1))

        # Remove the temporary file
        os.remove(temp_path)

    concat_emb = np.concatenate(embeddings, axis=None)

    return concat_emb


# Function to process and generate embeddings for segments
def process_audio_files(source_folder, model_path):
    concatenated_embeddings = {}
    
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                audio = AudioSegment.from_wav(file_path)
                duration = 10000  # Duration for each segment in milliseconds
                segments = [audio[:duration], audio[duration:duration*2], audio[duration*2:]]
                embeddings = []

                for i, segment in enumerate(segments):
                    # Temporarily export segment to handle with the model
                    temp_path = os.path.join(subdir, f"temp_seg_{i}.wav")
                    segment.export(temp_path, format="wav")

                    # Generate embedding
                    _, embedding = btest.test_model(model_path, temp_path, layers_dropped=1, test_segmentation=False)
                    embeddings.append(embedding.reshape(-1))

                    # Remove the temporary file
                    os.remove(temp_path)

                # Concatenate embeddings and store them using the original file name as the key
                concatenated_embeddings[file_path] = np.concatenate(embeddings)

    return concatenated_embeddings

def main():
    # Example usage
    # source_folder = '../test_wav_files/'  # Update this path
    model_path = "../models/genre.pt"  # Update this path
    test_query_embedding = "../test_wav_files/test_query.npy"

    # Process audio files and get concatenated embeddings
    # final_embeddings = process_file("../test_wav_files/000694.wav", model_path)


    # print(f"Final Embedding: {final_embeddings} with shape: {final_embeddings.shape}")
    #np.save("../test_wav_files/"+ "test_query.npy", final_embeddings)

    loaded_embedding = np.load(test_query_embedding).astype('float32')

    loaded_config = load_config()

    v_db = SimilaritySearch(loaded_config)

    top_neighbors = v_db.find_similar_embeddings(loaded_embedding)

    print(top_neighbors)

    

if __name__ == '__main__':
    main()
