import os
import zipfile
import urllib.request
import ssl
import pandas as pd

# Paths
DATA_DIR = os.path.join("artifacts", "data")
ARTIFACTS_DIR = "artifacts"
os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")

print("1. Downloading MovieLens 100K...")
# Bypass local SSL certificate verification for dataset download
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

with urllib.request.urlopen(URL, context=ctx) as response, open(ZIP_PATH, 'wb') as out_file:
    out_file.write(response.read())

print("2. Extracting files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

ratings_path = os.path.join(DATA_DIR, "ml-100k", "u.data")
items_path = os.path.join(DATA_DIR, "ml-100k", "u.item")

print("3. Processing data...")
# Load Ratings
ratings = pd.read_csv(
    ratings_path, 
    sep='\t', 
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)

# Load Movie Metadata
genres = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genres
movies = pd.read_csv(items_path, sep='|', names=item_cols, encoding='latin-1')

# Build text metadata combining title and genres for embedding extraction
def extract_genres(row):
    active = [g for g in genres if row[g] == 1]
    return ", ".join(active) if active else "General"

movies['genre_str'] = movies.apply(extract_genres, axis=1)
movies['metadata_text'] = movies['title'] + " | Genres: " + movies['genre_str']

# Temporal Train-Test Split (80/20 based strictly on timestamp order)
ratings = ratings.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(ratings) * 0.8)

train_ratings = ratings.iloc[:split_idx]
test_ratings = ratings.iloc[split_idx:]

# Save artifacts in parquet format for fast I/O
ratings.to_parquet(os.path.join(ARTIFACTS_DIR, "ratings.parquet"), index=False)
train_ratings.to_parquet(os.path.join(ARTIFACTS_DIR, "train_ratings.parquet"), index=False)
test_ratings.to_parquet(os.path.join(ARTIFACTS_DIR, "test_ratings.parquet"), index=False)
movies[['movie_id', 'title', 'genre_str', 'metadata_text']].to_parquet(
    os.path.join(ARTIFACTS_DIR, "movies.parquet"), index=False
)

print(f"Done! Train size: {len(train_ratings)}, Test size: {len(test_ratings)}")