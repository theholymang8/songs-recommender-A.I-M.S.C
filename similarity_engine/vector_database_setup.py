import os
import pandas as pd
import faiss
import numpy as np
import logging
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, config, create_index=False, load_vectors=False):
        self.config = config
        self.vector_files = self.get_full_path(config['paths']['vector_file'])
        self.dimension = config['faiss']['dimension']
        self.index_type = config['faiss']['index_type']
        self.engine = self.create_db_engine() if config else None
        self.index = None
        self.vectors = []
        self.track_ids = []
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
        db_config = self.config['database']
        return create_engine(f"mysql+mysqldb://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

    def load_vectors(self):
        if not os.path.exists(self.vector_files):
            logger.error(f"Error: Folder {self.vector_files} not found.")
            return
        try:
            file_contents = os.listdir(self.vector_files)
            for file_ in file_contents:
                full_path = os.path.join(self.vector_files, file_)
                loaded_embedding = np.load(full_path).astype('float32')
                if loaded_embedding.shape[0] == self.dimension:
                    self.track_ids.append(int(file_[:-4]))
                    self.vectors.append(loaded_embedding)
            logger.info(f"Vectors successfully loaded. Number of vectors loaded: {len(self.vectors)}")
        except Exception as e:
            logger.error(f"An error occurred while loading vectors: {e}")

    def create_index(self):
        if self.index_type == 'FlatL2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_L2)
        embeddings_for_index = np.array(self.vectors).astype('float32')
        self.index.add(embeddings_for_index)
        logger.info(f"Index created and embeddings added. Total embeddings: {len(self.vectors)}")

    def insert_metadata(self):
        data = {
            'vector_id': range(len(self.vectors)),
            'track_id': self.track_ids,
            'vector_dimensions': [self.dimension] * len(self.vectors),
            'faiss_index': [self.index_type] * len(self.vectors)
        }
        df = pd.DataFrame(data)
        df.to_sql('vector_metadata', con=self.engine, if_exists='replace', index=False)
        logger.info("Metadata inserted into the database successfully.")

    def save_index(self):
        index_path = self.get_full_path(self.config['paths']['index_path'])
        faiss.write_index(self.index, index_path)
        logger.info(f"Index saved to {index_path}")

    def load_index(self):
        index_path = self.get_full_path(self.config['paths']['index_path'])
        if not os.path.exists(index_path):
            logger.error(f"Error: Index file {index_path} not found.")
            return
        try:
            self.index = faiss.read_index(index_path)
            logger.info("Index successfully loaded.")
        except Exception as e:
            logger.error(f"An error occurred while loading the index: {e}")
