# Vector Database and Similarity Search Module

This project provides tools for managing a vector database and performing similarity searches on embeddings. The module includes classes and scripts to handle vector data, metadata, and index files efficiently.

## Requirements

Before running the scripts, ensure that the following requirements are met:

- Python 3.6 or higher
- NumPy
- Pandas
- SQLAlchemy
- FAISS
- MySQL server (or compatible SQL server)

Install the required Python packages using:

```bash
pip install numpy pandas sqlalchemy faiss-cpu mysqlclient
```

## Configuration

All configuration parameters such as database credentials, index information, and paths for embeddings are managed in the `config.json` file. Update this configuration file with the appropriate settings before proceeding with database and index setup:

```json
{
    "database": {
        "user": "your_username",
        "password": "your_password",
        "host": "localhost",
        "port": "3306",
        "database": "your_database_name"
    },
    "paths": {
        "vector_file": "path/to/your/vectors.npy",
        "index_path": "path/to/save/index"
    },
    "faiss": {
        "dimension": 128,
        "index_type": "FlatL2"  // Use IVFFlat or other types for larger datasets
    }
}

```

## Modules

### Vector Database Setup

This module includes a class `VectorDatabase` responsible for:

-   Loading vectors from a specified file.
-   Creating and saving a FAISS index in the `index` folder within the module directory.
-   Inserting metadata about the vectors into the configured SQL database.

The index stores the vectors themselves and is loaded from the specified path on instantiation.

### Similarity Search

Once the vector database is set up and indexed, the `similarity_search.py` script can be used to perform similarity searches. This script uses the saved FAISS index to find the most similar embeddings based on a given query vector.

### Creating the Database

To set up the database from scratch and insert vector metadata into MySQL, run the `create_vector_database.py` script. Ensure that the configuration file has the correct parameters.

## FAISS Indexes

For the use case of 8000 embeddings, recommended FAISS indexes include:

-   `FlatL2`: Ideal for smaller datasets with exact search requirements.
-   `IVFFlat`: Suitable for larger datasets where a balance between speed and accuracy is needed.

## Metadata and Track IDs

Each embedding corresponds to one track ID, derived from the original WAV files. The metadata stored in the database links these track IDs with their respective embeddings, facilitating efficient retrieval and management.

## Usage

To use this module, configure your settings in `config.json`, set up the database with `create_vector_database.py`, and use `similarity_search.py` to perform queries.
