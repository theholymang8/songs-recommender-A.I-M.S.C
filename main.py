from core.data import get_genre_data, get_train_test

# unzip_data("data/fma_small.zip", "data/")

get_train_test(
    "data/fma_small/",
    "data/fma_wav/",
    tracks_table_path="metadata/tracks_parsed.csv",
    test_size=0.25,
)

get_genre_data(
    "data/fma_small/",
    "data/fma_wav/train_data/",
    tracks_table_path="metadata/tracks_parsed.csv",
)

# if __name__ == "__main__":

#     bt.train_model(["data/fma_wav/Electronic",
#                     "data/fma_wav/Folk",
#                     "data/fma_wav/Rock",
#                     "data/fma_wav/Hip-Hop",
#                     "data/fma_wav/Instrumental",
#                     "data/fma_wav/International",
#                     "data/fma_wav/Pop",
#                     "data/fma_wav/Experimental"], "genre_clf")
