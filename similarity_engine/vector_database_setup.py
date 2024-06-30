import os
import pandas as pd
import faiss
import numpy as np
from sqlalchemy import create_engine

class VectorDatabase:
    def __init__(self, config, create_index=True):
        self.config = config
        self.vector_files = config['paths']['vector_file']
        self.dimension = config['faiss']['dimension']
        self.index_type = config['faiss']['index_type']
        self.engine = self.create_db_engine() if config else None
        self.index = None
        self.vectors = []
        self.track_ids = []
        self.load_vectors()
        if len(self.vectors) != 0 or create_index!=False:
            self.create_index()

    def create_db_engine(self):
        db_config = self.config['database']
        return create_engine(f"mysql+mysqldb://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

    def load_vectors(self):
        print(f"Vectors Folder Path: {self.vector_files}")
        if not os.path.exists(self.vector_files):
            print(f"Error: Folder {self.vector_files} not found.")
            return
        try:
            file_contents = os.listdir(self.vector_files)
            for file_ in file_contents:
                loaded_embedding = np.load(self.vector_files + "/" + file_).astype('float32')
                if loaded_embedding.shape[0] == 768:
                    self.track_ids.append(int(file_[:-4]))
                    self.vectors.append(loaded_embedding)
            print(f"Vectors successfully loaded. Number of vectors loaded from the embeddings file: {len(self.vectors)}")
        except Exception as e:
            print(f"An error occurred while loading vectors: {e}")

    def create_index(self):
        if self.index_type == 'FlatL2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_L2)
        
        embeddings_for_index = np.array(self.vectors).astype('float32')
        print(embeddings_for_index.shape)
        self.index.add(embeddings_for_index)

    def insert_metadata(self):
        # Create a DataFrame to hold the metadata
        data = {
            'vector_id': range(len(self.vectors)),
            'track_id': self.track_ids,
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
