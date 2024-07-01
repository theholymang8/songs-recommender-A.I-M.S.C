import json
import logging
import os

from similarity_engine.vector_database_setup import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path=None):
    if config_path is None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dir_path, "config.json")
    with open(config_path, "r") as file:
        return json.load(file)


class SimilaritySearch:
    def __init__(self, config):
        """Initialize by loading the FAISS index from the index path in config."""
        self.vector_db = VectorDatabase(config, create_index=False, load_vectors=False)

    def find_similar_embeddings(self, query_vector, top_k=5):
        """
        Find the top_k most similar embeddings to the given query_vector.

        Args:
        query_vector (numpy.array): The query vector to compare against the database.
        top_k (int): The number of nearest neighbors to return.

        Returns:
        numpy.array: Indices of the top_k nearest vectors in the database.
        numpy.array: Distances to the top_k nearest vectors.
        """

        self.vector_db.load_index()

        if self.vector_db.index is None:
            raise ValueError(
                "FAISS index is not loaded. Please ensure the index is properly loaded."
            )

        # Guys this is also important, ensure the query vector is in the correct shape (1, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Perform the search
        distances, indices = self.vector_db.index.search(query_vector, top_k)
        return indices, distances
