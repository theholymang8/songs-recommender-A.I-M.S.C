import os
import pandas as pd
import faiss
import numpy as np
from sqlalchemy import create_engine

class VectorDatabase:
    def __init__(self, config):
        self.vector_file = config['paths']['vector_file']
        self.dimension = config['faiss']['dimension']
        self.index_type = config['faiss']['index_type']
        self.engine = self.create_db_engine() if config else None
        self.index = None
        self.vectors = None
        self.load_vectors()
        if self.vectors is not None:
            self.create_index()

    def create_db_engine(self):
        db_config = self.config['database']
        return create_engine(f"mysql+mysqldb://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

    def load_vectors(self):
        if not os.path.exists(self.vector_file):
            print(f"Error: File {self.vector_file} not found.")
            return
        try:
            self.vectors = np.load(self.vector_file).astype('float32')
            print("Vectors successfully loaded.")
        except Exception as e:
            print(f"An error occurred while loading vectors: {e}")

    def create_index(self):
        if self.index_type == 'FlatL2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_L2)
        self.index.add(self.vectors)

    def insert_metadata(self, track_ids):
        # Create a DataFrame to hold the metadata
        data = {
            'vector_id': range(len(self.vectors)),
            'track_id': track_ids,
            'vector_dimensions': [self.dimension] * len(self.vectors),
            'faiss_index': [self.index_type] * len(self.vectors)
        }
        df = pd.DataFrame(data)
        # Insert data into the database
        df.to_sql('vector_metadata', con=self.engine, if_exists='replace', index=False)

    def save_index(self):
        index_path = self.config['paths']['index_path']
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")

    def load_index(self):
        index_path = self.config['paths']['index_path']
        if not os.path.exists(index_path):
            print(f"Error: Index file {index_path} not found.")
            return
        try:
            self.index = faiss.read_index(index_path)
            print("Index successfully loaded.")
        except Exception as e:
            print(f"An error occurred while loading the index: {e}")
