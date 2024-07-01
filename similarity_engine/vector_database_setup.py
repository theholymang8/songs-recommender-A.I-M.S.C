import logging
import os

import faiss
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self, config, create_index=False, load_vectors=False):
        self.config = config
        self.combinator = config["combinator"]
        self.vector_files = config["paths"]["embeddings_folder"]
        self.dimension = config["faiss"]["dimension"]
        self.index_type = config["faiss"]["index_type"]
        self.engine = self.create_db_engine() if config else None
        self.index = None
        self.vectors = []
        self.track_ids = []
        self.embedding_map = {}
        if load_vectors:
            self.load_vectors()
        if create_index:
            self.create_index()

    def get_full_path(self, relative_path):
        """
        Convert a relative path to an absolute path based on the script's location.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)

    def create_db_engine(self):
        db_config = self.config["database"]
        return create_engine(
            f"mysql+mysqldb://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

    def load_vectors(self):
        if not os.path.exists(self.vector_files["genre_classification_embeddings"]):
            logger.error(f"Error: Folder {path} not found.")
            return
        try:
            # default to genre-classification embeddings in the index
            map_of_embeddings = {"genre": [], "instrument": [], "emotion": []}
            for key, _ in map_of_embeddings.items():
                if self.combinator[key]:
                    full_folder_path = self.get_full_path(
                        self.vector_files[f"{key}_classification_embeddings"]
                    )
                    file_contents = os.listdir(full_folder_path)
                    for file_ in tqdm(file_contents):
                        full_path = os.path.join(full_folder_path, file_)
                        loaded_embedding = np.load(full_path).astype("float32")
                        if loaded_embedding.shape[0] == self.dimension:
                            # logger.debug(f"Embedding shape: {loaded_embedding.shape}")
                            track_id = int(file_[:-4])

                            if track_id not in self.embedding_map:
                                self.embedding_map[track_id] = {}

                            self.embedding_map[track_id][key] = loaded_embedding

                            if track_id not in self.track_ids:
                                self.track_ids.append(track_id)
                            map_of_embeddings[key].append(loaded_embedding)

            # Concatenate embeddings by track_id
            # final_embeddings = []
            for track_id in self.track_ids:
                embeddings_to_concat = [
                    self.embedding_map[track_id][key]
                    for key in map_of_embeddings
                    if self.combinator[key] and key in self.embedding_map[track_id]
                ]
                # logger.debug(f"Embeddings to concat so far: {embeddings_to_concat}")
                if embeddings_to_concat:
                    # Ensure all embeddings to concatenate are arrays and concatenation is along axis 1
                    concatenated_embedding = np.concatenate(
                        embeddings_to_concat, axis=None
                    )
                    self.vectors.append(concatenated_embedding)

            logger.info(
                f"Embeddings successfully concatenated for multiple genres, number of vectors loaded: {len(self.vectors)}"
            )
            # return np.array(final_embeddings)
        except Exception as e:
            logger.error(f"An error occurred while loading vectors: {e}")

    def create_index(self):
        # Calculate the effective dimension
        effective_dimension = self.dimension * self.dimensionality_calculation()

        # Normalize the vectors to unit length for cosine similarity
        embeddings_for_index = np.array(self.vectors).astype("float32")

        if self.index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(effective_dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(effective_dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, effective_dimension, 100, faiss.METRIC_L2
            )
            self.index.train(embeddings_for_index)
        elif self.index_type == "Cosine":
            # Use IndexFlatIP for cosine similarity, which uses normalized vectors
            norm = np.linalg.norm(embeddings_for_index, axis=1, keepdims=True)
            index = faiss.index_factory(
                effective_dimension, "Flat", faiss.METRIC_INNER_PRODUCT
            )
            faiss.normalize_L2(embeddings_for_index)
            # normalized_embeddings = embeddings_for_index / np.maximum(norm, 1e-6)
            self.index = index

        logger.info(f"Index type set to: {self.index_type}")
        logger.info(f"Dimensions of the vectors: {effective_dimension}")
        logger.info(f"Length of Vectors so far: {len(self.vectors)}")

        # Add vectors to the index
        self.index.add(
            embeddings_for_index
        )  # Add normalized vectors for cosine similarity

        logger.info(
            f"Index created and embeddings added. Total embeddings: {len(self.vectors)}"
        )

    def insert_metadata(self):
        dimensions = self.dimension * self.dimensionality_calculation()
        data = {
            "vector_id": range(len(self.vectors)),
            "track_id": self.track_ids,
            "vector_dimensions": [dimensions] * len(self.vectors),
            "faiss_index": [self.index_type] * len(self.vectors),
        }
        df = pd.DataFrame(data)
        df.to_sql("vector_metadata", con=self.engine, if_exists="replace", index=False)
        logger.info("Metadata inserted into the database successfully.")

    def save_index(self):
        index_path = self.get_full_path(self.config["paths"]["index_path"])
        faiss.write_index(self.index, index_path)
        logger.info(f"Index saved to {index_path}")

    def load_index(self):
        index_path = self.get_full_path(self.config["paths"]["index_path"])
        if not os.path.exists(index_path):
            logger.error(f"Error: Index file {index_path} not found.")
            return
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"Index: {index_path} successfully loaded.")
        except Exception as e:
            logger.error(f"An error occurred while loading the index: {e}")

    def dimensionality_calculation(self):
        if (
            self.combinator["genre"]
            and self.combinator["instrument"]
            and self.combinator["emotion"]
        ):
            return 3
        elif (
            self.combinator["genre"]
            and self.combinator["instrument"]
            and not self.combinator["emotion"]
        ):
            return 2
        elif (
            self.combinator["genre"]
            and not self.combinator["instrument"]
            and not self.combinator["emotion"]
        ):
            return 1
