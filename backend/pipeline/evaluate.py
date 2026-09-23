import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

ARTIFACTS_DIR = "artifacts"

print("1. Loading test dataset and model artifacts...")
train_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train_ratings.parquet"))
test_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test_ratings.parquet"))
movies_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies.parquet"))
embeddings = np.load(os.path.join(ARTIFACTS_DIR, "movie_embeddings.npy"))

with open(os.path.join(ARTIFACTS_DIR, "mappings.pkl"), "rb") as f:
    mappings = pickle.load(f)
movie_id_to_idx = mappings["movie_id_to_idx"]

with open(os.path.join(ARTIFACTS_DIR, "svd_model.pkl"), "rb") as f:
    svd = pickle.load(f)

# Reconstruct NCF
class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, u, i):
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        return self.mlp(torch.cat([u_e, i_e], dim=-1)).squeeze(-1)

checkpoint = torch.load(os.path.join(ARTIFACTS_DIR, "ncf_model.pt"))
ncf = NCF(checkpoint['num_users'], checkpoint['num_items'])
ncf.load_state_dict(checkpoint['model_state_dict'])
ncf.eval()

with open(os.path.join(ARTIFACTS_DIR, "hybrid_lgb.pkl"), "rb") as f:
    hybrid_lgb = pickle.load(f)

# Build taste profile from train set
high_ratings = train_df[train_df['rating'] >= 4]
user_liked_dict = high_ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
user_profiles = {}
for uid, mids in user_liked_dict.items():
    valid = [movie_id_to_idx[m] for m in mids if m in movie_id_to_idx]
    if valid:
        user_profiles[uid] = embeddings[valid].mean(axis=0)

print("2. Computing Predictions on Held-Out Test Set (20,000 interactions)...")
y_true = test_df['rating'].values

u_test = torch.tensor(test_df['user_id'].values, dtype=torch.long)
i_test = torch.tensor(test_df['movie_id'].values, dtype=torch.long)
with torch.no_grad():
    ncf_preds = ncf(u_test, i_test).numpy()

svd_preds = []
content_preds = []

for idx, row in test_df.iterrows():
    u = int(row['user_id'])
    m = int(row['movie_id'])
    svd_preds.append(svd.predict(u, m).est)
    
    if u in user_profiles and m in movie_id_to_idx:
        c_score = float(np.dot(user_profiles[u], embeddings[movie_id_to_idx[m]]))
    else:
        c_score = 0.0
    content_preds.append(c_score)

svd_preds = np.array(svd_preds)
content_preds = np.array(content_preds)

X_eval = pd.DataFrame({
    'svd_score': svd_preds,
    'ncf_score': ncf_preds,
    'content_score': content_preds
})
hybrid_preds = hybrid_lgb.predict(X_eval)

# A. Compute RMSE
rmse_svd = np.sqrt(mean_squared_error(y_true, svd_preds))
rmse_ncf = np.sqrt(mean_squared_error(y_true, ncf_preds))
rmse_hybrid = np.sqrt(mean_squared_error(y_true, hybrid_preds))

print("3. Computing Ranking Metrics (Precision@10, Recall@10, NDCG@10, Coverage)...")
# Group test ratings by user for top-10 ranking metrics
eval_df = test_df.copy()
eval_df['hybrid_pred'] = hybrid_preds
eval_df['is_relevant'] = (eval_df['rating'] >= 4).astype(int)

precisions, recalls, ndcgs = [], [], []
recommended_items = set()
catalog_items = set(movies_df['movie_id'])

for uid, group in eval_df.groupby('user_id'):
    if len(group) < 10:
        continue
    # Rank by hybrid score
    ranked = group.sort_values('hybrid_pred', ascending=False).head(10)
    hits = ranked['is_relevant'].sum()
    total_rel = group['is_relevant'].sum()
    
    # Precision@10
    precisions.append(hits / 10.0)
    
    # Recall@10
    if total_rel > 0:
        recalls.append(hits / total_rel)
        
    # NDCG@10
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ranked['is_relevant'])])
    ideal_hits = sorted(group['is_relevant'].tolist(), reverse=True)[:10]
    idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_hits)])
    ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    
    recommended_items.update(ranked['movie_id'].tolist())

coverage = (len(recommended_items) / len(catalog_items)) * 100

print("\n" + "="*50)
print("             OFFICIAL BENCHMARK RESULTS")
print("="*50)
print(f"SVD RMSE:           {rmse_svd:.4f}")
print(f"NCF RMSE:           {rmse_ncf:.4f}")
print(f"Hybrid Model RMSE:  {rmse_hybrid:.4f}")
print(f"Precision@10:       {np.mean(precisions):.4f}")
print(f"Recall@10:          {np.mean(recalls):.4f}")
print(f"NDCG@10:            {np.mean(ndcgs):.4f}")
print(f"Catalog Coverage:   {coverage:.2f}%")
print("="*50)