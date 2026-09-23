import os
import re
import json
import asyncio
from pathlib import Path
import httpx
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CACHE_FILE = DATA_DIR / "poster_cache.json"

SEMAPHORE = asyncio.Semaphore(10)


def clean_title_and_year(raw_title: str):
    # Extract 4-digit year
    year_match = re.search(r"\((\d{4})\)", str(raw_title))
    year = year_match.group(1) if year_match else None

    # Remove year and trailing ellipses/whitespace
    title = re.sub(r"\s*\(\d{4}\)", "", str(raw_title))
    title = title.replace("...", "").strip()

    # Invert trailing articles
    for art in ["The", "A", "An"]:
        if title.endswith(f", {art}"):
            title = f"{art} " + title[:-len(f", {art}")].strip()

    title = title.replace('"', '').replace("'", "").strip()
    return title, year


async def fetch_poster_url(client: httpx.AsyncClient, raw_title: str):
    clean_title, year = clean_title_and_year(raw_title)
    if not TMDB_API_KEY:
        return raw_title, ""

    async with SEMAPHORE:
        # Pass 1: Strict search with title + primary_release_year
        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "include_adult": "false"
        }
        if year:
            params["primary_release_year"] = year

        try:
            r = await client.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=8.0)
            if r.status_code == 200:
                results = r.json().get("results", [])
                for item in results:
                    if item.get("poster_path"):
                        return raw_title, f"https://image.tmdb.org/t/p/w500{item['poster_path']}"

            # Pass 2: Fallback without strict year
            if year:
                params.pop("primary_release_year", None)
                r_fallback = await client.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=8.0)
                if r_fallback.status_code == 200:
                    results = r_fallback.json().get("results", [])
                    for item in results:
                        if item.get("poster_path"):
                            return raw_title, f"https://image.tmdb.org/t/p/w500{item['poster_path']}"
        except Exception:
            pass

    return raw_title, ""


async def build_cache():
    print("🎬 Starting offline poster pre-cache...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Gather all unique titles across base and modern datasets
    movies_df = pd.read_parquet(ARTIFACTS_DIR / "movies.parquet")
    titles = [t for t in movies_df["title"].unique() if t not in cache or not cache[t]]

    print(f"📦 Titles needing posters: {len(titles)}")
    if not titles:
        print("✅ All posters are already cached!")
        return

    async with httpx.AsyncClient(headers={"User-Agent": "RecSysLab/1.0"}) as client:
        # Run in chunks of 50 to track progress
        chunk_size = 50
        for i in range(0, len(titles), chunk_size):
            chunk = titles[i : i + chunk_size]
            tasks = [fetch_poster_url(client, t) for t in chunk]
            results = await asyncio.gather(*tasks)
            for raw_title, url in results:
                if url:
                    cache[raw_title] = url
            print(f"  Processed {min(i + chunk_size, len(titles))}/{len(titles)} titles...")
            await asyncio.sleep(0.2)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"💾 Successfully saved {len(cache)} posters to {CACHE_FILE}")


if __name__ == "__main__":
    asyncio.run(build_cache())