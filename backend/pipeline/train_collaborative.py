import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from surprise import SVD, Dataset as SurpriseDataset, Reader

ARTIFACTS_DIR = "artifacts"
train_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train_ratings.parquet"))
movies_df = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "movies.parquet"))

# 1. Baseline Model: Classical SVD via Surprise
print("1. Training Classical Matrix Factorization (SVD)...")
reader = Reader(rating_scale=(1, 5))
surprise_data = SurpriseDataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)
trainset = surprise_data.build_full_trainset()

svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

with open(os.path.join(ARTIFACTS_DIR, "svd_model.pkl"), "wb") as f:
    pickle.dump(svd, f)
print("   SVD baseline trained and saved.")

# 2. Main Model: Neural Collaborative Filtering (NCF) in PyTorch
print("2. Setting up PyTorch Neural Collaborative Filtering (NCF)...")

num_users = int(train_df['user_id'].max()) + 1
num_items = int(movies_df['movie_id'].max()) + 1

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
        x = torch.cat([u_e, i_e], dim=-1)
        return self.mlp(x).squeeze(-1)

class RecDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.i = torch.tensor(df['movie_id'].values, dtype=torch.long)
        self.r = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Training NCF on device: {device}")

model = NCF(num_users, num_items).to(device)
train_loader = DataLoader(RecDataset(train_df), batch_size=256, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

print("3. Training NCF (5 Epochs)...")
model.train()
for epoch in range(5):
    total_loss = 0.0
    for u_b, i_b, r_b in train_loader:
        u_b, i_b, r_b = u_b.to(device), i_b.to(device), r_b.to(device)
        optimizer.zero_grad()
        preds = model(u_b, i_b)
        loss = criterion(preds, r_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"   Epoch {epoch + 1}/5 | Train MSE Loss: {avg_loss:.4f}")

# Save PyTorch model state and dimensions configuration
torch.save({
    'model_state_dict': model.state_dict(),
    'num_users': num_users,
    'num_items': num_items
}, os.path.join(ARTIFACTS_DIR, "ncf_model.pt"))

print("Done! Both SVD and NCF models trained and saved to artifacts.")