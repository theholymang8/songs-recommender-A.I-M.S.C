import os

import numpy as np
from deep_audio_features.bin import basic_test as btest
from deep_audio_features.dataloading.dataloading import FeatureExtractorDataset
from deep_audio_features.lib.training import test
from deep_audio_features.utils.model_editing import drop_layers
from pydub import AudioSegment
from torch.utils.data import DataLoader


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
    segments = [audio[:duration], audio[duration : duration * 2], audio[duration * 2 :]]
    embeddings = []

    # Temporary directory for storing intermediate files, relative to the root directory
    temp_dir = os.path.join(root_directory, "temp_embs/")
    os.makedirs(temp_dir, exist_ok=True)

    for i, segment in enumerate(segments):
        temp_path = os.path.join(temp_dir, f"temp_emb_{i}.wav")
        segment.export(temp_path, format="wav")

        # Generate embedding using the model
        _, embedding = btest.test_model(
            absolute_model_path, temp_path, layers_dropped=1, test_segmentation=False
        )
        embeddings.append(embedding.reshape(-1))

        # Remove the temporary file
        os.remove(temp_path)

    concat_emb = np.concatenate(embeddings, axis=None)
    return concat_emb


def process_file_custom(file_path, models):
    # Get the base path for the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Correctly build the absolute path to the model
    # Assuming your script is inside a subdirectory of the root project folder
    root_directory = os.path.dirname(script_directory)

    # Ensure to construct the absolute path for the audio file similarly
    absolute_file_path = os.path.join(root_directory, file_path)

    # Load audio from the absolute file path
    audio = AudioSegment.from_wav(absolute_file_path)
    duration = 10000  # Duration for each segment in milliseconds
    segments = [audio[:duration], audio[duration : duration * 2], audio[duration * 2 :]]

    # Temporary directory for storing intermediate files, relative to the root directory
    temp_dir = os.path.join(root_directory, "temp_embs/")
    os.makedirs(temp_dir, exist_ok=True)

    embeddings_from_inference = []
    for i, segment in enumerate(segments):
        temp_path = os.path.join(temp_dir, f"temp_emb_{i}.wav")
        segment.export(temp_path, format="wav")

        for _, model_ in models.items():
            model = model_["properties"]["model"]
            model = drop_layers(model, 1)

            # Create test set
            test_set = FeatureExtractorDataset(
                X=[temp_path],
                y=[0],
                fe_method="MEL_SPECTROGRAM",
                oversampling=False,
                max_sequence_length=model_["properties"]["max_seq_length"],
                zero_pad=model_["properties"]["zero_pad"],
                forced_size=model_["properties"]["spec_size"],
                fuse=model_["properties"]["fuse"],
                show_hist=False,
                test_segmentation=False,
                hop_length=model_["properties"]["hop_length"],
                window_length=model_["properties"]["window_length"],
            )

            # Create test dataloader
            test_loader = DataLoader(
                dataset=test_set,
                batch_size=1,
                num_workers=4,
                drop_last=False,
                shuffle=False,
            )

            # Forward a sample
            posteriors, _, _ = test(
                model=model,
                dataloader=test_loader,
                cnn=True,
                task="classification",
                classifier=True if False else False,
            )

            # print(f"Embedding from segment: {posteriors}")

            posteriors = np.array(posteriors)

            embeddings_from_inference.append(posteriors.reshape(-1))

            # Remove the temporary file

        os.remove(temp_path)

    concatenated_inference_embedding = np.concatenate(
        embeddings_from_inference, axis=None
    )

    return concatenated_inference_embedding
