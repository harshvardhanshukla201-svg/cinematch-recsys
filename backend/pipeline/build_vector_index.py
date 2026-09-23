import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = "artifacts"
movies_path = os.path.join(ARTIFACTS_DIR, "movies.parquet")

print("1. Loading movie metadata...")
movies_df = pd.read_parquet(movies_path)

print("2. Generating sentence embeddings via all-MiniLM-L6-v2...")
# Downloads lightweight, high-performance transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

movie_texts = movies_df['metadata_text'].tolist()
# Normalize embeddings so dot product equals cosine similarity
embeddings = embedder.encode(
    movie_texts, 
    convert_to_numpy=True, 
    normalize_embeddings=True,
    show_progress_bar=True
)

print("3. Building FAISS index for vector search...")
dimension = embeddings.shape[1]  # 384 dimensions for all-MiniLM-L6-v2
# IndexFlatIP uses Inner Product (exact cosine similarity on normalized vectors)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))

# Save FAISS index and raw embeddings array
faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "faiss_index.index"))
np.save(os.path.join(ARTIFACTS_DIR, "movie_embeddings.npy"), embeddings)

# Create and save bidrectional ID-to-index mappings
movie_id_to_idx = {int(mid): i for i, mid in enumerate(movies_df['movie_id'])}
idx_to_movie_id = {i: int(mid) for i, mid in enumerate(movies_df['movie_id'])}

with open(os.path.join(ARTIFACTS_DIR, "mappings.pkl"), "wb") as f:
    pickle.dump({
        "movie_id_to_idx": movie_id_to_idx, 
        "idx_to_movie_id": idx_to_movie_id
    }, f)

print(f"Done! Indexed {index.ntotal} movies with vector dimension {dimension}.")