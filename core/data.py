import os
import shutil
import subprocess
from pathlib import Path, PurePath
from typing import Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


def unzip_data(zip_file: str, destination_path: str):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    shutil.unpack_archive(zip_file, destination_path, "zip")


def get_wav_data(
    source_folder: str,
    destination_folder: str,
    eligible_files: Union[list, None] = None,
):
    """Convert audio data of the source folder to wav format with sample rate 44,1kHz.
    Additionally convert form stereo to mono.

    Args:
        source_folder (str): Source folder of MP3 data.
        destination_folder (str): The directory we want the train and test sets to be saved.
        eligible_files (Union[list, None], optional): List with the file names that we want to convert to WAV. Defaults to None.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    folders = os.listdir(source_folder)
    for folder in folders:
        if os.path.isdir(os.path.abspath(source_folder + folder)):
            files = os.listdir(source_folder + folder)
            for file in files:
                if file[:-4] in eligible_files and not os.path.exists(
                    destination_folder + file[:-4] + ".wav"
                ):
                    source_file = list(
                        PurePath(os.path.join(source_folder, folder, file)).parts
                    )
                    destiantion_file = list(
                        PurePath(
                            os.path.join(destination_folder, file[:-4] + ".wav")
                        ).parts
                    )

                    source_file = Path(*source_file)
                    destiantion_file = Path(*destiantion_file)
                    subprocess.call(
                        [
                            "ffmpeg",
                            "-i",
                            source_file,
                            "-ac",
                            "1",
                            destiantion_file,
                            "-ar",
                            "44100",
                            "-loglevel",
                            "error",
                        ]
                    )


def segment_audio(source_folder: str):
    """Segment every audio file in the directory. Additionally the parent file is deleted along with
    segments that are less than 2 seconds long.

    Args:
        source_folder (str): Folder containing audio files.
    """
    # make segmention files and delete parent
    wav_files = os.listdir(source_folder)
    for wav in wav_files:
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                source_folder + wav,
                "-f",
                "segment",
                "-segment_time",
                "10",
                f"{source_folder}{wav[:-4]}_%0d.wav",
                "-loglevel",
                "error",
            ]
        )
        os.remove(source_folder + wav)

    # delete segments less than 2 seconds
    wav_files = os.listdir(source_folder)
    for wav in wav_files:
        duration = librosa.get_duration(path=source_folder + wav)
        if duration <= 2:
            os.remove(source_folder + wav)


def get_train_test(source_folder: str, destination_folder: str, **kwargs) -> None:
    """_summary_

    Args:
        source_folder (str): Source folder of MP3 data.
        destination_folder (str): The directory we want the train and test sets to be saved.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if not os.path.exists(destination_folder + "train_data/"):
        os.mkdir(destination_folder + "train_data/")

    if not os.path.exists(destination_folder + "test_data/"):
        os.mkdir(destination_folder + "test_data/")

    train_set, test_set = make_train_test(**kwargs)

    train_set.write_csv(destination_folder + "train_data/" + "train_set_map.csv")
    test_set.write_csv(destination_folder + "test_data/" + "test_set_map.csv")

    train_file_names = train_set["file_name"].to_list()
    test_file_names = train_set["file_name"].to_list()

    get_wav_data(source_folder, destination_folder + "train_data/", train_file_names)
    print("Training dataset generated!")

    get_wav_data(source_folder, destination_folder + "test_data/", test_file_names)
    print("Test dataset generated!")


def get_spect_data(source_folder: str, destination_folder: str):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    folders = os.listdir(source_folder)
    for folder in folders:
        if os.path.isdir(os.path.abspath(source_folder + folder)):
            # os.mkdir(destination_folder + folder)
            files = os.listdir(source_folder + folder)
            for file in files:
                get_spect(
                    file_path=source_folder + folder + "/" + file,
                    destination=destination_folder + file[:-4] + ".jpg",
                )


def get_spect(file_path: str, destination: str):
    array, _ = librosa.load(file_path)

    D = librosa.stft(array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure()
    librosa.display.specshow(S_db)
    plt.savefig(destination)
    plt.close()


def make_train_test(
    tracks_table_path: str,
    test_size: float,
    shuffle: bool = True,
    random_state: int = 0,
):
    """_summary_

    Args:
        tracks_table_path (str): Path of the csv file containing track_id and file genres.
        test_size (float): The percentage of the dataset kept as test size.
        shuffle (bool, optional): Shuffle before splitting. Defaults to True.
        random_state (int, optional): Random seed. Defaults to 0.

    Returns:
        train_set_map, test_set_map: Polars dataframes with the train and test set mapping, respectively.
    """
    # read track dataset
    metadata = pl.read_csv(tracks_table_path)
    # select necessary rows
    metadata = metadata.select(["track_id", "genre_top"])
    # make column with track filename
    metadata = metadata.with_columns(
        pl.col("track_id")
        .map_elements(fill_track_id, return_dtype=pl.String)
        .alias("file_name")
    )
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        metadata["file_name"],
        metadata["genre_top"],
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=metadata["genre_top"],
    )

    train_set_map = pl.DataFrame({"file_name": X_train, "genre": y_train})
    test_set_map = pl.DataFrame({"file_name": X_test, "genre": y_test})

    return train_set_map, test_set_map


def get_genre_data(source_folder: str, destination_folder: str, **kwargs):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    genre_dict = make_genre_data(**kwargs)

    for key in genre_dict.keys():
        if not os.path.exists(destination_folder + key + "/"):
            os.mkdir(destination_folder + key + "/")

        get_wav_data(source_folder, destination_folder + key + "/", genre_dict[key])
        segment_audio(destination_folder + key + "/")

        print(f"{key} WAV data generated!")


def make_genre_data(tracks_table_path: str):
    # read track dataset
    metadata = pl.read_csv(tracks_table_path)
    # select necessary rows
    metadata = metadata.select(["track_id", "genre_top"])
    # make column with track filename
    metadata = metadata.with_columns(
        pl.col("track_id")
        .map_elements(fill_track_id, return_dtype=pl.String)
        .alias("file_name")
    )
    genres = metadata["genre_top"].unique().to_list()

    genre_dict = {}

    for genre in genres:
        genre_dict[genre] = metadata.filter(pl.col("genre_top") == genre)[
            "file_name"
        ].to_list()

    return genre_dict


def fill_track_id(track_id: int):
    """Given a track ID, convert it to a 6 character string filled with zeros at the front
    (if the track ID contains less than 6 digits already).

    Args:
        track_id (int): The ID of the track.

    Returns:
        track_id: String converted track ID.
    """
    track_id = str(track_id)
    required_len = 6
    char_len = len(track_id)

    added_zeros = required_len - char_len
    if added_zeros > 0:
        track_id = added_zeros * "0" + track_id

    return track_id
