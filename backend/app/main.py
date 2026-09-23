import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple, Any
import httpx
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.engine import engine

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "poster_cache.json"

app = FastAPI(title="Hybrid RecSys API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_valid_text(val: Any) -> bool:
    """Returns True if the value is non-empty and not a stringified null/nan artifact."""
    if val is None or pd.isna(val):
        return False
    cleaned = str(val).strip()
    return bool(cleaned) and cleaned.lower() not in ("nan", "none", "null", "undefined")


METADATA_CACHE = {}
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
            for k, v in raw_cache.items():
                if isinstance(v, str):
                    METADATA_CACHE[k] = {"poster_url": v, "overview": ""}
                elif isinstance(v, dict):
                    overview_val = v.get("overview", "")
                    METADATA_CACHE[k] = {
                        "poster_url": v.get("poster_url", ""),
                        "overview": overview_val if is_valid_text(overview_val) else "",
                    }
        print(f"🖼️ Loaded {len(METADATA_CACHE)} cached titles from disk.")
    except Exception as e:
        print(f"⚠️ Could not load metadata cache: {e}")

SEMAPHORE = asyncio.Semaphore(5)


def clean_title_and_year(raw_title: str) -> Tuple[str, Optional[str]]:
    year_match = re.search(r"\((\d{4})\)", str(raw_title))
    year = year_match.group(1) if year_match else None

    title = re.sub(r"\s*\(\d{4}\)", "", str(raw_title))
    title = title.replace("...", "").strip()

    for art in ["The", "A", "An"]:
        if title.endswith(f", {art}"):
            title = f"{art} " + title[:-len(f", {art}")].strip()

    title = title.replace('"', "").replace("'", "").strip()
    return title, year


async def fetch_tmdb_metadata(client: httpx.AsyncClient, raw_title: str) -> Tuple[str, str]:
    clean_title, year = clean_title_and_year(raw_title)
    fallback_poster = f"https://placehold.co/500x750/0f172a/94a3b8?text={clean_title.replace(' ', '+')}"

    if raw_title in METADATA_CACHE:
        entry = METADATA_CACHE[raw_title]
        cached_poster = entry.get("poster_url", "")
        cached_overview = entry.get("overview", "")
        if cached_poster:
            return cached_poster, cached_overview

    if not TMDB_API_KEY:
        return fallback_poster, ""

    async with SEMAPHORE:
        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "include_adult": "false",
        }
        if year:
            params["primary_release_year"] = year

        try:
            resp = await client.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
                timeout=10.0,
                headers={"Accept": "application/json"}
            )
            if resp.status_code == 429:
                await asyncio.sleep(0.8)
                return fallback_poster, ""

            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    best = results[0]
                    poster = f"https://image.tmdb.org/t/p/w500{best['poster_path']}" if best.get("poster_path") else fallback_poster
                    overview = best.get("overview", "") if is_valid_text(best.get("overview")) else ""
                    METADATA_CACHE[raw_title] = {"poster_url": poster, "overview": overview}
                    return poster, overview

            if year:
                params.pop("primary_release_year", None)
                resp_relaxed = await client.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params=params,
                    timeout=10.0,
                    headers={"Accept": "application/json"}
                )
                if resp_relaxed.status_code == 200:
                    results = resp_relaxed.json().get("results", [])
                    if results:
                        best = results[0]
                        poster = f"https://image.tmdb.org/t/p/w500{best['poster_path']}" if best.get("poster_path") else fallback_poster
                        overview = best.get("overview", "") if is_valid_text(best.get("overview")) else ""
                        METADATA_CACHE[raw_title] = {"poster_url": poster, "overview": overview}
                        return poster, overview
        except Exception as e:
            print(f"❌ TMDB Error for '{clean_title}': {repr(e)}")

    return fallback_poster, ""


async def enrich_movie_items(items: list) -> list:
    async with httpx.AsyncClient(
        headers={"User-Agent": "CineMatchLab/1.0"},
        timeout=10.0,
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
    ) as client:
        tasks = []
        for item in items:
            raw_title = item["title"]
            has_poster = (
                item.get("poster_url")
                and isinstance(item["poster_url"], str)
                and item["poster_url"].startswith("http")
                and "placehold.co" not in item["poster_url"]
            )
            has_overview = is_valid_text(item.get("overview"))

            if has_poster and has_overview:
                tasks.append(asyncio.sleep(0, result=(item["poster_url"], str(item["overview"]).strip())))
            elif raw_title in METADATA_CACHE and METADATA_CACHE[raw_title].get("poster_url"):
                cached = METADATA_CACHE[raw_title]
                tasks.append(asyncio.sleep(0, result=(cached["poster_url"], cached.get("overview", ""))))
            elif has_poster:
                tasks.append(asyncio.sleep(0, result=(item["poster_url"], item.get("overview", ""))))
            else:
                tasks.append(fetch_tmdb_metadata(client, raw_title))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item, res in zip(items, results):
            if isinstance(res, tuple):
                poster, overview = res
                item["poster_url"] = poster or item.get("poster_url") or ""
                item["overview"] = overview if is_valid_text(overview) else (item.get("overview", "") if is_valid_text(item.get("overview")) else "")
            else:
                clean_title, _ = clean_title_and_year(item["title"])
                if not item.get("poster_url"):
                    item["poster_url"] = f"https://placehold.co/500x750/0f172a/94a3b8?text={clean_title.replace(' ', '+')}"
                if not is_valid_text(item.get("overview")):
                    item["overview"] = ""

            raw_score = item.get("score")
            item["score"] = 3.5 if (raw_score is None or pd.isna(raw_score)) else float(raw_score)

    return items


@app.get("/")
def health_check():
    return {"status": "online", "catalog_size": len(engine.movies_df)}


@app.get("/recommend/{user_id}")
async def recommend(user_id: int, mode: str = "hybrid", limit: int = 10):
    try:
        recs = engine.recommend(user_id=user_id, mode=mode, top_k=limit)
        return await enrich_movie_items(recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
async def compare_modes(user_id: int = Query(1, alias="user_id")):
    try:
        collab = engine.recommend_collaborative(user_id, k=5)
        content = engine.recommend_content(user_id, k=5)
        hybrid = engine.recommend_hybrid(user_id, k=5)

        collab = await enrich_movie_items(collab)
        content = await enrich_movie_items(content)
        hybrid = await enrich_movie_items(hybrid)

        return {
            "collaborative": collab,
            "content": content,
            "hybrid": hybrid,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/movie/{movie_id}/similar")
async def similar_movies(
    movie_id: int, 
    limit: int = 10,
    diversity: float = Query(0.65, ge=0.0, le=1.0, description="MMR Lambda: 1.0 = Max Similarity, 0.0 = Max Diversity")
):
    try:
        recs = engine.get_similar_movies(movie_id=movie_id, top_k=limit, lambda_mult=diversity)
        return await enrich_movie_items(recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_movies(query: str = Query(..., min_length=1), limit: int = 8):
    q = query.strip()
    if not q:
        return []

    tokens = [re.escape(t) for t in q.split() if t]
    mask = engine.movies_df["title"].str.contains(tokens[0], case=False, na=False)
    for token in tokens[1:]:
        mask &= engine.movies_df["title"].str.contains(token, case=False, na=False)

    matches = engine.movies_df[mask].head(limit)

    results = []
    for _, row in matches.iterrows():
        raw_overview = row.get("overview", "")
        benchmark_score = engine.get_film_benchmark_rating(row)
        clean_score = float(benchmark_score) if pd.notna(benchmark_score) else 3.5

        results.append({
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "genres": str(row.get("genre_str", "Film")),
            "poster_url": str(row.get("poster_url", "")),
            "overview": raw_overview if is_valid_text(raw_overview) else "",
            "score": clean_score,
        })
    return await enrich_movie_items(results)