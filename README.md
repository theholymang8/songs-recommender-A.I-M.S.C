# Introduction

This is a repo for the semester project of MSc in AI at NSCR Demokritos x University of Piraeus. The project is about building a song similarity search system utilizing DL methods.

# Table of contents
1. [First Steps](#first-steps)
    1. [Create Conda environment](#create-virtual-environment)
    2. [Download Data](#download-the-data)
    3. [Unzip Data](#unzip-the-data)
    4. [Convert MP3 to WAV](#convert-mp3-data-to-wav)
    5. [Demo Application](#demo-application)

# First steps

## Create Conda environment

The environment used during the development of this project was created with anaconda using python 3.12.3. You can choose to create you own virtual environment, although, an `environment.yml` is included in the repository to facilitate the dependency management.

To create the envrionment, make sure you have Anaconda installed in your local environment and simply run:

```bash
conda env create -f environment.yml
```

This will create the virtual envrionment named `multimodal-2024` and you can activate it afterward by running:

```bash
conda activate multimodal-2024
```

**NOTE**: Please make sure that `ffmpeg` is installed in your environment since the segmentation process can fail if this software is not installed before hand since `pydub` is not able to operate correctly without finding its installation path.

Now, you should be able to go through the rest of the project without having issues with dependencies.

## Download the data

Download the __fma_small__ dataset from [FMA Github Repo](https://github.com/mdeff/fma).

## Unzip the data

Get the data in MP3 format by unziping the file:

- zip_file: Path to zip, e.g. "path/to/zip/file.zip"
- destination_path: Destination folder of the decompressed data, e.g. "destination/folder/"

```python
from core.data import unzip_data

unzip_data(zip_file, destination_path)
```

## Convert MP3 data to WAV

In order to convert the unzipped data from MP3 to WAV run the python script below.
- __source_folder__ is where the directory where the unzipped data is located
- __destination_folder__ is where we want the WAV files to be stored
- __sample_rate__ is the sample rate used for conversion

```python
from core.data import get_wav_data

get_wav_data(source_folder, destination_folder, sample_rate)
```

## Demo application

### Requirements for the pipeline

To run the inference pipeline certain requirements have to be met since not all components can be included in the repository itself. These are the following:

- **Models**: To inference samples you need to have a folder named `models/` in your root directory containing all three downstream taks classification models, saved in the `.pt` format. Meaning, the following structure has to be created with respect to the names of the models:

    ```bash
    /root/
        ├── models/
            ├── genre.pt
            ├── instruments.pt
            └── mood.pt
    ```

- **Database Setup**: The pipeline assumes that a running MySQL server with the correct schema, tables and data is running in your environment. This is important since track details are saved as metadata in the database. module `db_setup/` includes specific information on how to setup your local database environment and insert all metadata needed for the pipeline.

To showcase the pipeline correctly, most components have to be executed first. Meaning that indexes have to be created (the most important ones are included in the repository) under the `similarity_engine/index` module folder. These are important to run the search operation.

The demo application uses `streamlit` to run so simply run the script `demo_app.py` located in the root directory of the repository using this command:

```bash
streamlit run demo_app.py
```
