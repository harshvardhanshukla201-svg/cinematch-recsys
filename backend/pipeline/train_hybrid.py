import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

ARTIFACTS_DIR = "artifacts"

print("1. Loading artifacts and training data...")
train_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train_ratings.parquet"))
movies_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies.parquet"))
embeddings = np.load(os.path.join(ARTIFACTS_DIR, "movie_embeddings.npy"))

with open(os.path.join(ARTIFACTS_DIR, "mappings.pkl"), "rb") as f:
    mappings = pickle.load(f)
movie_id_to_idx = mappings["movie_id_to_idx"]

with open(os.path.join(ARTIFACTS_DIR, "svd_model.pkl"), "rb") as f:
    svd = pickle.load(f)

# Recreate NCF architecture to load weights
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

print("2. Building user content preference profiles...")
# Compute each user's taste profile as the mean vector of movies they rated 4 or 5
high_ratings = train_df[train_df['rating'] >= 4]
user_liked_dict = high_ratings.groupby('user_id')['movie_id'].apply(list).to_dict()

user_profiles = {}
for uid, mids in user_liked_dict.items():
    valid_indices = [movie_id_to_idx[m] for m in mids if m in movie_id_to_idx]
    if valid_indices:
        user_profiles[uid] = embeddings[valid_indices].mean(axis=0)

print("3. Extracting hybrid features (SVD, NCF, and Content similarity)...")
svd_scores = []
ncf_scores = []
content_scores = []

# Prepare batch tensors for faster NCF inference
u_tensor = torch.tensor(train_df['user_id'].values, dtype=torch.long)
i_tensor = torch.tensor(train_df['movie_id'].values, dtype=torch.long)

with torch.no_grad():
    ncf_preds = ncf(u_tensor, i_tensor).numpy()

for idx, row in train_df.iterrows():
    u = int(row['user_id'])
    m = int(row['movie_id'])
    
    # 1. SVD score
    svd_scores.append(svd.predict(u, m).est)
    
    # 2. NCF score
    ncf_scores.append(float(ncf_preds[idx]))
    
    # 3. Content score (cosine similarity with user's profile)
    if u in user_profiles and m in movie_id_to_idx:
        c_score = float(np.dot(user_profiles[u], embeddings[movie_id_to_idx[m]]))
    else:
        c_score = 0.0
    content_scores.append(c_score)

X_train = pd.DataFrame({
    'svd_score': svd_scores,
    'ncf_score': ncf_scores,
    'content_score': content_scores
})
y_train = train_df['rating'].values

print("4. Training LightGBM stacking regressor...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

# Save the trained hybrid stacking model
with open(os.path.join(ARTIFACTS_DIR, "hybrid_lgb.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)

# Feature importance readout
feature_imp = dict(zip(X_train.columns, lgb_model.feature_importances_))
print(f"Done! Feature Importances: {feature_imp}")