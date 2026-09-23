from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def rebuild_all_embeddings():
    print("🧠 Initializing SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    movies_df = pd.read_parquet(ARTIFACTS_DIR / "movies.parquet")
    print(f"📦 Loaded {len(movies_df)} MovieLens movies.")

    # Standardize genre column
    if "genres" in movies_df.columns:
        genre_col = movies_df["genres"].astype(str).str.replace("|", " ", regex=False)
    else:
        genre_col = ""

    texts = [
        f"Movie Title: {title}. Genres: {genre}"
        for title, genre in zip(movies_df["title"], genre_col)
    ]

    print("⚡ Encoding MovieLens films into 384-dim space...")
    base_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.save(ARTIFACTS_DIR / "movie_embeddings.npy", base_embeddings.astype("float32"))
    print(f"💾 Saved base embeddings to {ARTIFACTS_DIR / 'movie_embeddings.npy'}")

if __name__ == "__main__":
    rebuild_all_embeddings()