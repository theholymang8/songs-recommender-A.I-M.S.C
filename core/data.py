import os
import shutil


def unzip_data(zip_file: str, destination_path: str):
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    shutil.unpack_archive(zip_file, destination_path, "zip")
