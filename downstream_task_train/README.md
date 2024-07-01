# Audio Features Classification

This repository contains Python scripts for training machine learning models on different audio classification tasks: emotion classification, genre classification, and instrument classification. Each script is designed to train a model with data specified by the user and save the model to a designated output directory.

## Directory Structure
Make sure your data for each task is organized into subfolders, each representing a class. The path you provide to the script should contain these subfolders.

### Example Directory Structure:

```bash
/path/to/data/
    ├── class1/
    ├── class2/
    └── class3/
```

## Running the Scripts
The scripts are designed to be run from the command line. Below are the commands for each task. Replace /path/to/data with the path to your specific dataset for each task.

### Emotion Classification
To train the model for emotion classification:

```bash
python emotion_classification.py /path/to/emotion/data mood.pt
```

### Genre Classification
To train the model for genre classification:

```bash
python genre_classification.py /path/to/genre/data genre.pt
```

### Instrument Classification
To train the model for instrument classification:

```bash
python instrument_classification.py /path/to/instrument/data instruments.pt
```

## Output
The trained models will then have to be saved in a directory named `models/` in the *root directory* of the repository since the scripts save the models in the directory they are being executed. Make sure this directory exists or is created by you after the execution of the scripts. That's important since the rest of the pipeline assumes that the models will be in that destination

## Notes
Ensure that each data path provided has the correct subdirectory structure as mentioned in the directory structure section.
Adjust the scripts if there are any path issues or specific configurations needed for deep_audio_features.
