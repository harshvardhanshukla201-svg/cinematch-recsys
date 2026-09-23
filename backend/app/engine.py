import os
import pickle
from pathlib import Path
from typing import Any, List, Dict
import pandas as pd
import numpy as np
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
MODERN_CATALOG_PATH = DATA_DIR / "modern_movies.parquet"


def clean_overview(val: Any) -> str:
    """Sanitizes Pandas NaN or stringified null variations into clean empty strings."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null", "undefined") else s


class HybridRecommendationEngine:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.svd_model = None
        self.lgbm_ranker = None
        self.faiss_index = None
        self.content_embeddings = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.mean_ratings = {}
        self.load_artifacts()

    def load_artifacts(self):
        print("⚡ Loading MovieLens core artifacts from artifacts directory...")

        # 1. Datasets
        self.movies_df = pd.read_parquet(ARTIFACTS_DIR / "movies.parquet")
        self.ratings_df = pd.read_parquet(ARTIFACTS_DIR / "ratings.parquet")

        mean_series = self.ratings_df.groupby("movie_id")["rating"].mean()
        self.mean_ratings = mean_series.to_dict()

        # 2. Collaborative & GBDT Models
        with open(ARTIFACTS_DIR / "svd_model.pkl", "rb") as f:
            self.svd_model = pickle.load(f)

        with open(ARTIFACTS_DIR / "hybrid_lgb.pkl", "rb") as f:
            self.lgbm_ranker = pickle.load(f)

        # 3. Base Embeddings
        base_embeddings = np.load(ARTIFACTS_DIR / "movie_embeddings.npy").astype("float32")

        if "genre_str" not in self.movies_df.columns:
            if "genres" in self.movies_df.columns:
                self.movies_df["genre_str"] = self.movies_df["genres"].astype(str)
            else:
                self.movies_df["genre_str"] = "Film"

        self.movies_df["is_modern"] = 0

        # 4. Modern Catalog Ingestion
        if MODERN_CATALOG_PATH.exists():
            print("🚀 Merging modern cinema catalog into FAISS vector space...")
            modern_df = pd.read_parquet(MODERN_CATALOG_PATH)

            if "genres" not in modern_df.columns:
                modern_df["genres"] = "Action, Adventure, Sci-Fi"
            if "genre_str" not in modern_df.columns:
                modern_df["genre_str"] = modern_df["genres"].astype(str)

            modern_df["is_modern"] = 1
            modern_embeddings = np.vstack(modern_df["embedding"].values).astype("float32")

            self.content_embeddings = np.vstack([base_embeddings, modern_embeddings])
            self.movies_df = pd.concat([self.movies_df, modern_df], ignore_index=True)
        else:
            self.content_embeddings = base_embeddings

        faiss.normalize_L2(self.content_embeddings)

        # 5. Build FAISS Index
        dim = self.content_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.content_embeddings)

        # 6. Mappings
        self.movie_id_to_idx = {int(row["movie_id"]): i for i, row in self.movies_df.iterrows()}
        self.idx_to_movie_id = {i: int(row["movie_id"]) for i, row in self.movies_df.iterrows()}
        print(f"✅ Catalog ready with {len(self.movies_df)} titles in {dim}-dim semantic space.")

    def get_film_benchmark_rating(self, row: pd.Series) -> float:
        """Derives a realistic 5-star baseline rating from TMDB or MovieLens ground truth."""
        if "vote_average" in row and pd.notna(row["vote_average"]):
            val = float(row["vote_average"])
            if val > 0:
                return round(val / 2.0, 1)

        m_id = int(row["movie_id"])
        if m_id in self.mean_ratings:
            return round(float(self.mean_ratings[m_id]), 1)

        return 3.6

    def predict_svd_rating(self, user_id: int, movie_id: int) -> float:
        """Collaborative predicted score for specific user-movie pair."""
        try:
            _ = self.svd_model.trainset.to_inner_iid(movie_id)
            pred = float(self.svd_model.predict(uid=user_id, iid=movie_id).est)
            return round(min(5.0, max(1.0, pred)), 2)
        except Exception:
            row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if not row.empty:
                return round(self.get_film_benchmark_rating(row.iloc[0]), 2)
            return 3.5

    def get_similar_movies(self, movie_id: int, top_k: int = 10, lambda_mult: float = 0.65) -> List[Dict]:
        """
        L3 Diversity: Maximal Marginal Relevance (MMR) over candidate embeddings.
        - lambda_mult = 1.0: Pure Cosine Relevance (dense cluster / filter bubble)
        - lambda_mult = 0.5 - 0.7: Balanced relevance with high genre & plot diversity
        """
        movie_id = int(movie_id)
        if movie_id not in self.movie_id_to_idx:
            return []

        seed_idx = self.movie_id_to_idx[movie_id]
        query_vec = self.content_embeddings[seed_idx : seed_idx + 1]

        # Retrieve an expanded candidate pool (L1)
        candidate_pool_size = min(top_k * 4, len(self.movies_df) - 1)
        sim_scores, candidate_indices = self.faiss_index.search(query_vec, candidate_pool_size + 1)

        # Exclude the seed movie itself
        filtered_indices = [idx for idx in candidate_indices[0] if idx != seed_idx][:candidate_pool_size]
        if not filtered_indices:
            return []

        candidate_embeddings = self.content_embeddings[filtered_indices]
        query_similarities = np.dot(candidate_embeddings, query_vec.T).squeeze()

        # MMR Iterative Selection
        selected_indices = []
        selected_candidate_pos = []
        unselected_candidate_pos = list(range(len(filtered_indices)))

        for _ in range(min(top_k, len(filtered_indices))):
            if not selected_candidate_pos:
                # First element: Highest similarity to query
                best_pos = int(np.argmax(query_similarities))
            else:
                # Subsequent elements: Maximize MMR score
                selected_embeds = candidate_embeddings[selected_candidate_pos]
                
                # Pairwise cosine between unselected and already chosen items
                inter_sims = np.dot(candidate_embeddings[unselected_candidate_pos], selected_embeds.T)
                max_inter_sims = np.max(inter_sims, axis=1)

                mmr_scores = (
                    lambda_mult * query_similarities[unselected_candidate_pos]
                    - (1.0 - lambda_mult) * max_inter_sims
                )
                best_sub_idx = int(np.argmax(mmr_scores))
                best_pos = unselected_candidate_pos[best_sub_idx]

            selected_candidate_pos.append(best_pos)
            unselected_candidate_pos.remove(best_pos)
            selected_indices.append(filtered_indices[best_pos])

        results = []
        for orig_idx in selected_indices:
            m_id = self.idx_to_movie_id[orig_idx]
            row = self.movies_df[self.movies_df["movie_id"] == m_id].iloc[0]

            true_rating = self.get_film_benchmark_rating(row)
            cosine_score = float(np.dot(self.content_embeddings[orig_idx], query_vec.squeeze()))
            pct_match = int(round(cosine_score * 100))

            results.append({
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "genres": str(row.get("genre_str", "Film")),
                "score": true_rating,
                "reason": f"Semantic Match ({pct_match}% affinity)",
                "poster_url": str(row.get("poster_url", "")),
                "overview": clean_overview(row.get("overview", "")),
            })

        return results

    def recommend_collaborative(self, user_id: int, k: int = 10) -> List[Dict]:
        scores = []
        candidate_ids = self.movies_df["movie_id"].values[:350]
        for m_id in candidate_ids:
            pred = self.predict_svd_rating(user_id, int(m_id))
            scores.append((int(m_id), pred))

        scores.sort(key=lambda x: x[1], reverse=True)

        recs = []
        for m_id, score in scores[:k]:
            row = self.movies_df[self.movies_df["movie_id"] == m_id].iloc[0]
            recs.append({
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "genres": str(row.get("genre_str", "Film")),
                "score": round(score, 1),
                "reason": f"Collaborative Filtering ({score:.1f}★)",
                "poster_url": str(row.get("poster_url", "")),
                "overview": clean_overview(row.get("overview", "")),
            })
        return recs

    def recommend_hybrid(self, user_id: int, k: int = 10) -> List[Dict]:
        sample_size = min(250, len(self.movies_df))
        candidates = self.movies_df.sample(sample_size, random_state=42).copy()

        features = []
        for _, row in candidates.iterrows():
            svd_pred = self.predict_svd_rating(user_id, row["movie_id"])
            benchmark = self.get_film_benchmark_rating(row)
            is_modern = int(row.get("is_modern", 0))

            blended_score = (svd_pred * 0.7) + (benchmark * 0.3)
            if is_modern:
                blended_score += 0.1

            final_rating = round(min(5.0, max(1.0, blended_score)), 1)

            features.append({
                "movie_id": int(row["movie_id"]),
                "score": final_rating,
                "is_modern": is_modern,
            })

        feat_df = pd.DataFrame(features).sort_values("score", ascending=False).head(k)

        recs = []
        for _, r in feat_df.iterrows():
            row = self.movies_df[self.movies_df["movie_id"] == r["movie_id"]].iloc[0]
            reason = (
                "Hybrid: High predicted affinity + contemporary release"
                if r["is_modern"]
                else "Hybrid: Blends personalized taste with critical consensus"
            )
            recs.append({
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "genres": str(row.get("genre_str", "Film")),
                "score": float(r["score"]),
                "reason": reason,
                "poster_url": str(row.get("poster_url", "")),
                "overview": clean_overview(row.get("overview", "")),
            })
        return recs

    def recommend_content(self, user_id: int, k: int = 10) -> List[Dict]:
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if not user_ratings.empty:
            top_movie_id = int(user_ratings.sort_values("rating", ascending=False).iloc[0]["movie_id"])
            return self.get_similar_movies(top_movie_id, k)
        return self.get_similar_movies(1, k)

    def recommend(self, user_id: int, mode: str = "hybrid", top_k: int = 10) -> List[Dict]:
        if mode == "content":
            return self.recommend_content(user_id, k=top_k)
        elif mode == "collaborative":
            return self.recommend_collaborative(user_id, k=top_k)
        return self.recommend_hybrid(user_id, k=top_k)


engine = HybridRecommendationEngine()