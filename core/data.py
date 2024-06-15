import os
import shutil
import subprocess

import librosa
import matplotlib.pyplot as plt
import numpy as np


def unzip_data(zip_file: str, destination_path: str):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    shutil.unpack_archive(zip_file, destination_path, "zip")


def get_wav_data(source_folder: str, destination_folder: str):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    folders = os.listdir(source_folder)
    for folder in folders:
        if os.path.isdir(os.path.abspath(source_folder + folder)):
            files = os.listdir(source_folder + folder)
            for file in files:
                if not os.path.exists(destination_folder + file[:-4] + ".wav"):
                    subprocess.call(
                        [
                            "ffmpeg",
                            "-i",
                            source_folder + folder + "/" + file,
                            destination_folder + file[:-4] + ".wav",
                            "-ar",
                            "44100",
                        ]
                    )


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
