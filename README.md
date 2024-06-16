# Introduction

This is a repo for the semester project of MSc in AI at NSCR Demokritos x University of Piraeus. The project is about building a song similarity search system utilizing DL methods.

# Table of contents
1. [First Steps](#first-steps)
    1. [Download Data](#download-the-data)
    2. [Unzip Data](#unzip-the-data)
    3. [Make Train & Test](#make-train--test-sets)

# First steps

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

## Make Train & Test sets

Create training and test set with __get_train_test__ function:

- __source_folder__: The folder that the data was unzipped in the previous step
- __destination_folder__: The folder you want your train & test set to be created
- __track_table_path__: The file that contains the mapping between track_id and genre class

```python
from core.data import get_train_test

get_train_test(source_folder, destination_folder,
               track_table_path,
               test_size)
```

This will create 2 extra folders inside your __destination_folder__ named "train_data" and "test_data", each containing the respective WAV files defined by the split and one csv containing the genre of each file.
