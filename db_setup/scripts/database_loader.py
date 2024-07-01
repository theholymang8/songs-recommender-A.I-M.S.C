import pandas as pd
from sqlalchemy import create_engine

# Replace 'your_username', 'your_password', and 'your_database' with your MySQL user, password, and database
username = "your_username"
password = "your_password"
host = "localhost"  # Assuming your db is hosted locally
port = "3306"  # Default MySQL port

# Considering you have named the schema as the name bellow
database = "multimodal_msc_ai_songs"

# Create the connection engine
engine = create_engine(
    f"mysql+mysqldb://{username}:{password}@{host}:{port}/{database}"
)

# Load the tracks CSV with UTF-8 encoding
tracks = pd.read_csv("./tracks_parsed.csv", encoding="utf-8")

# Load the CSV with UTF-8 encoding
albums = pd.read_csv("./albums_parsed.csv", encoding="utf-8")

# Load the CSV with UTF-8 encoding
artists = pd.read_csv("./artists_parsed.csv", encoding="utf-8")

# Write the dataframes to SQL
tracks.to_sql("tracks", con=engine, if_exists="replace", index=False)
albums.to_sql("albums", con=engine, if_exists="replace", index=False)
artists.to_sql("artists", con=engine, if_exists="replace", index=False)

print("Data has been successfully loaded into the database.")
