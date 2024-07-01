import os

import numpy as np

embeddings_folder = "./embeddings/genre-classification/genre-classification/"

embeddings_subfolders = os.listdir(embeddings_folder)

# print(embeddings_folder)

destination_folder = "./concatenated_embeddings/genre-classification/"

for subfolder in embeddings_subfolders:
    embeddings_files = os.listdir(embeddings_folder + subfolder)

    current_folder = embeddings_folder + subfolder + "/"

    file_name = subfolder

    list_ofembeddings = []
    for file_ in embeddings_files:
        embedding = np.load(current_folder + file_).astype("float32")
        list_ofembeddings.append(embedding)
    concatenated_embedding = np.concatenate(list_ofembeddings, axis=None)
    # print(concatenated_embedding, concatenated_embedding.shape)
    save_file_name = destination_folder + file_name + ".npy"
    np.save(save_file_name, concatenated_embedding)


# self.vectors = np.load(self.vector_file).astype('float32')
