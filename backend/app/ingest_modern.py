import os
import asyncio
from pathlib import Path
import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTENDED_CATALOG_PATH = DATA_DIR / "modern_movies.parquet"

# TMDB Genre ID mapping to standard genre names
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Children's",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Musical",
    9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 53: "Thriller",
    10752: "War", 37: "Western"
}

async def fetch_modern_movies(pages: int = 5):
    if not TMDB_API_KEY:
        raise ValueError(f"TMDB_API_KEY is not set.")

    movies = []
    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        for page in range(1, pages + 1):
            url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "sort_by": "vote_count.desc",
                "include_adult": "false",
                "page": page,
                "primary_release_date.gte": "2015-01-01",
                "vote_count.gte": 500,
            }
            try:
                res = await client.get(url, params=params, timeout=20.0)
                if res.status_code == 200:
                    for item in res.json().get("results", []):
                        title = item.get("title", "")
                        year = item.get("release_date", "")[:4]
                        formatted_title = f"{title} ({year})" if year else title
                        
                        g_ids = item.get("genre_ids", [])
                        genre_names = [TMDB_GENRES[g] for g in g_ids if g in TMDB_GENRES]
                        genre_str = "|".join(genre_names) if genre_names else "Action|Sci-Fi"

                        overview = item.get("overview", "")
                        poster_path = item.get("poster_path")
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

                        movies.append({
                            "tmdb_id": item.get("id"),
                            "title": formatted_title,
                            "genres": genre_str,
                            "genre_str": genre_str,
                            "overview": overview,
                            "vote_average": float(item.get("vote_average", 0.0)),
                            "vote_count": int(item.get("vote_count", 0)),
                            "poster_url": poster_url,
                            "is_modern": 1
                        })
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
            await asyncio.sleep(0.2)
    return movies

def build_modern_catalog():
    print("🎬 Ingesting modern movies with full genre mappings...")
    modern_items = asyncio.run(fetch_modern_movies(pages=5))
    df_modern = pd.DataFrame(modern_items)
    
    # Drop duplicates by title
    df_modern = df_modern.drop_duplicates(subset=["title"]).reset_index(drop=True)

    print(f"✅ Ingested {len(df_modern)} modern movies.")
    print("🧠 Generating rich semantic vectors with genre weight...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Format: heavily weight the movie genres and title so thematic similarity aligns cleanly
    texts = [
        f"Movie Title: {row['title']}. Genres: {row['genres'].replace('|', ' ')}. Overview: {row['overview']}"
        for _, row in df_modern.iterrows()
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    df_modern["embedding"] = [emb.tolist() for emb in embeddings]
    
    # MovieLens 100k IDs stop at 1682 -> start at 5000
    df_modern["movie_id"] = range(5000, 5000 + len(df_modern))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_modern.to_parquet(EXTENDED_CATALOG_PATH, index=False)
    print(f"💾 Saved to {EXTENDED_CATALOG_PATH}")

if __name__ == "__main__":
    build_modern_catalog()