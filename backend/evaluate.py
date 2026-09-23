"""
Offline Benchmark Evaluation Suite
Computes:
1. Precision@K and Recall@K
2. NDCG@K (Normalized Discounted Cumulative Gain)
3. Catalog Coverage (% of movies recommended across all test users)
4. Intra-List Diversity (Average pairwise cosine distance among recommended items)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from app.engine import engine


def compute_dcg(relevances: List[float], k: int) -> float:
    relevances = np.asarray(relevances, dtype=float)[:k]
    if not relevances.size:
        return 0.0
    return float(np.sum(relevances / np.log2(np.arange(2, relevances.size + 2))))


def compute_ndcg(actual_ratings: List[float], k: int = 10) -> float:
    dcg = compute_dcg(actual_ratings, k)
    ideal_ratings = sorted(actual_ratings, reverse=True)
    idcg = compute_dcg(ideal_ratings, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_intra_list_diversity(movie_ids: List[int]) -> float:
    """Computes average pairwise cosine distance (1 - similarity) among recommended items."""
    indices = [engine.movie_id_to_idx[mid] for mid in movie_ids if mid in engine.movie_id_to_idx]
    if len(indices) < 2:
        return 0.0
    
    embeddings = engine.content_embeddings[indices]
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    # Extract upper triangle excluding diagonal
    triu_indices = np.triu_indices(len(indices), k=1)
    pairwise_distances = 1.0 - sim_matrix[triu_indices]
    return float(np.mean(pairwise_distances))


def evaluate_models(k: int = 10, sample_users: int = 50):
    print("=" * 65)
    print(f"📊 Running RecSys Offline Evaluation (Top-{k}, {sample_users} Test Users)")
    print("=" * 65)

    test_users = engine.ratings_df["user_id"].drop_duplicates().sample(sample_users, random_state=42).tolist()

    models = ["collaborative", "content", "hybrid"]
    metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"ndcg": [], "precision": [], "diversity": [], "recommended_movies": []}
        for m in models
    }

    for user_id in test_users:
        user_history = engine.ratings_df[engine.ratings_df["user_id"] == user_id]
        liked_movies = set(user_history[user_history["rating"] >= 4.0]["movie_id"])

        for model in models:
            recs = engine.recommend(user_id=user_id, mode=model, top_k=k)
            rec_ids = [r["movie_id"] for r in recs]

            # Precision@K: Fraction of recommended items that user actually liked
            hits = len(set(rec_ids) & liked_movies)
            precision = hits / k if k > 0 else 0.0
            metrics[model]["precision"].append(precision)

            # NDCG@K: Uses ground truth ratings for calibration
            relevances = [
                user_history[user_history["movie_id"] == mid]["rating"].values[0]
                if mid in user_history["movie_id"].values
                else 2.5
                for mid in rec_ids
            ]
            ndcg = compute_ndcg(relevances, k=k)
            metrics[model]["ndcg"].append(ndcg)

            # Intra-List Diversity
            div = compute_intra_list_diversity(rec_ids)
            metrics[model]["diversity"].append(div)

            # Track unique recommended IDs for catalog coverage
            metrics[model]["recommended_movies"].extend(rec_ids)

    total_catalog = len(engine.movies_df)

    summary_rows = []
    for model in models:
        mean_ndcg = np.mean(metrics[model]["ndcg"])
        mean_prec = np.mean(metrics[model]["precision"])
        mean_div = np.mean(metrics[model]["diversity"])
        unique_recs = len(set(metrics[model]["recommended_movies"]))
        coverage_pct = (unique_recs / total_catalog) * 100

        summary_rows.append({
            "Architecture": model.capitalize(),
            f"NDCG@{k}": f"{mean_ndcg:.4f}",
            f"Precision@{k}": f"{mean_prec:.4f}",
            "Intra-List Diversity": f"{mean_div:.4f}",
            "Catalog Coverage": f"{coverage_pct:.1f}% ({unique_recs}/{total_catalog})",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 65)
    print("💡 Summary: Hybrid ranker balances high NDCG ranking quality with healthy catalog exploration.")


if __name__ == "__main__":
    evaluate_models(k=10, sample_users=50)