import argparse
import logging
import os

from deep_audio_features.bin import basic_training as bt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_subdirectory_paths(directory):
    return [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Train a genre classification model with the instrument dataset"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the main folder containing subfolders with classes",
    )
    parser.add_argument(
        "output_filename",
        type=str,
        help="Output filename for the trained model (current implementation works with the name instruments.pt)",
    )

    # Parse arguments
    args = parser.parse_args()

    # List subdirectory paths
    subdirectory_names = list_subdirectory_paths(args.path)
    folder_paths = [os.path.join(args.path, sub_dir) for sub_dir in subdirectory_names]

    # Train model
    bt.train_model(folder_paths, args.output_filename)
    logger.info(
        "model trainig completed succesfully, make sure to place the genarated model in a folder named models in the root directory of the project."
    )


if __name__ == "__main__":
    main()
