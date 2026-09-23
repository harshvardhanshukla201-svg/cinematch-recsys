import os
import re
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx

# Explicit path to .env
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

_poster_cache = {}

# Well-known MovieLens 100k alias mappings where canonical TMDB titles diverge
TITLE_ALIASES = {
    "Star Wars": "Star Wars: Episode IV - A New Hope",
    "Return of the Jedi": "Star Wars: Episode VI - Return of the Jedi",
    "Empire Strikes Back, The": "Star Wars: Episode V - The Empire Strikes Back",
    "The Empire Strikes Back": "Star Wars: Episode V - The Empire Strikes Back",
    "Raiders of the Lost Ark": "Indiana Jones and the Raiders of the Lost Ark",
    "Star Trek: The Wrath of Khan": "Star Trek II: The Wrath of Khan",
}

def clean_movie_title(raw_title: str):
    # Extract release year
    year_match = re.search(r'\((\d{4})\)', raw_title)
    year = int(year_match.group(1)) if year_match else None
    
    # Strip (Year), extra spaces, trailing punctuation
    title = re.sub(r'\s*\(\d{4}\)', '', raw_title).strip()
    title = re.sub(r'\.{2,}', '', title).strip()
    
    # Inverted articles: "Boot, Das" -> "Das Boot"
    article_match = re.match(r'^(.*?),\s*(The|A|An|Das|Der|Die|Le|La|Les|Il|L\')\b', title, flags=re.IGNORECASE)
    if article_match:
        title = f"{article_match.group(2)} {article_match.group(1)}".strip()
        
    title = title.replace("’", "'").replace("`", "'")
    return title, year

async def fetch_poster(client: httpx.AsyncClient, raw_title: str) -> str:
    if raw_title in _poster_cache:
        return _poster_cache[raw_title]

    clean_title, year = clean_movie_title(raw_title)

    # Check hardcoded alias dictionary first
    aliased_title = TITLE_ALIASES.get(clean_title, clean_title)

    if TMDB_API_KEY:
        # Build search fallback strategies
        queries = [aliased_title]
        if ":" in clean_title:
            queries.append(clean_title.split(":")[0].strip()) # e.g. "Star Trek"
        if aliased_title != clean_title:
            queries.append(clean_title)

        attempts = []
        for q in queries:
            if year:
                attempts.append({"query": q, "primary_release_year": year})
                attempts.append({"query": q, "year": year})
            attempts.append({"query": q})
        
        # Add alphanumeric-only fallback
        attempts.append({"query": re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_title).strip()})

        for params in attempts:
            params["api_key"] = TMDB_API_KEY
            params["include_adult"] = "false"
            
            for retry in range(2):
                try:
                    res = await client.get(
                        "https://api.themoviedb.org/3/search/movie", 
                        params=params, 
                        timeout=6.0
                    )
                    if res.status_code == 200:
                        data = res.json()
                        results = data.get("results", [])
                        for item in results:
                            if item.get("poster_path"):
                                poster_url = f"https://image.tmdb.org/t/p/w500{item['poster_path']}"
                                _poster_cache[raw_title] = poster_url
                                return poster_url
                    elif res.status_code == 429:
                        await asyncio.sleep(0.3)
                        continue
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

    # Final visual fallback
    fallback = f"https://placehold.co/500x750/0f172a/94a3b8?text={clean_title.replace(' ', '+')}"
    _poster_cache[raw_title] = fallback
    return fallback