import os
import shutil
import subprocess
from pathlib import Path, PurePath

import librosa
import matplotlib.pyplot as plt
import numpy as np


def unzip_data(zip_file: str, destination_path: str):
    """Unzip a compressed file into the destination folder.

    Args:
        zip_file (str): Compressed file destination.
        destination_path (str): Directory into whihc the unzipped data will be stored.
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    shutil.unpack_archive(zip_file, destination_path, "zip")


def get_wav_data(source_folder: str, destination_folder: str, sample_rate: int):
    """Convert audio data of the source folder to wav format with sample rate 44,1kHz.
    Additionally convert form stereo to mono.

    Args:
        source_folder (str): Source folder of MP3 data.
        destination_folder (str): The directory we want the train and test sets to be saved.
        sample_rate (int): The sample frequency to use when converting MP3 to WAV.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    folders = os.listdir(source_folder)
    for folder in folders:
        if os.path.isdir(os.path.abspath(source_folder + folder)):
            files = os.listdir(source_folder + folder)
            for file in files:
                source_file = list(
                    PurePath(os.path.join(source_folder, folder, file)).parts
                )
                destiantion_file = list(
                    PurePath(os.path.join(destination_folder, file[:-4] + ".wav")).parts
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
                        str(sample_rate),
                        "-loglevel",
                        "error",
                    ]
                )


def segment_audio(source_folder: str, destination_folder: str):
    """Segment every audio file in the directory into 10 second segments. Additionally the parent
    file is deleted along with segments that are less than 2 seconds long.

    Args:
        source_folder (str): Folder containing audio files.
        destination_folder (str): Folder where the segmented files will be stored.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # make segmention files and delete parent
    wav_files = os.listdir(source_folder)
    for wav in wav_files:
        if wav.endswith(".wav"):
            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    source_folder + wav,
                    "-f",
                    "segment",
                    "-segment_time",
                    "10",
                    f"{destination_folder}{wav[:-4]}_%0d.wav",
                    "-loglevel",
                    "error",
                ]
            )

    # delete segments less than 2 seconds
    wav_files = os.listdir(destination_folder)
    for wav in wav_files:
        duration = librosa.get_duration(path=destination_folder + wav)
        if duration <= 2:
            os.remove(destination_folder + wav)


def get_spect(file_path: str, destination: str):
    """Create and store the spectrgram of an WAV file.

    Args:
        file_path (str): WAV file path.
        destination (str): The place where the spectrogram will be stored.
    """
    array, _ = librosa.load(file_path)

    D = librosa.stft(array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure()
    librosa.display.specshow(S_db)
    plt.savefig(destination)
    plt.close()
