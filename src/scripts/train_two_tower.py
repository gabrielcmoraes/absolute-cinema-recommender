# src/scripts/train.py

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.model.recommender import TwoTowerModel

# 1) Configuration & paths
print("--- [Step 1/8] Configuration & Setup ---")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MOVIES_CSV = os.path.join(DATA_DIR, "movie.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "rating.csv")
GENOME_SCORES_CSV = os.path.join(DATA_DIR, "genome_scores.csv")

print("Configuration complete.")
print("-" * 40)

# 2) Data loading
print("--- [Step 2/8] Loading Data ---")
try:
    movies_df = pd.read_csv(MOVIES_CSV)
    ratings_df = pd.read_csv(RATINGS_CSV)
    genome_scores_df = pd.read_csv(GENOME_SCORES_CSV)
    print("Datasets loaded successfully.")
    print(f"Movies: {len(movies_df)} rows, Ratings: {len(ratings_df)} rows")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    raise SystemExit(1)
print("-" * 40)

# Global ID pools for consistent encoding
all_user_ids = ratings_df["userId"].unique()
all_movie_ids = movies_df["movieId"].unique()

# 3) Feature engineering (item tower)
print("--- [Step 3/8] Feature Engineering ---")

# Genres → multi-hot
print("Processing genres...")
movies_df["genres"] = movies_df["genres"].apply(
    lambda x: [] if x == "(no genres listed)" else str(x).split("|")
)
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(movies_df["genres"])
genre_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=movies_df["movieId"])

# Tag Genome → dense features
print("Processing Tag Genome...")
genome_vectors_df = genome_scores_df.pivot_table(
    index="movieId",
    columns="tagId",
    values="relevance"
).fillna(0)
genome_vectors_df.columns = [f"genome_tag_{col}" for col in genome_vectors_df.columns]

# Merge content features
movies_with_features = movies_df[["movieId", "title", "genres"]].set_index("movieId")
content_features_df = pd.merge(movies_with_features, genre_df, left_index=True, right_index=True, how="left")
content_features_df = pd.merge(content_features_df, genome_vectors_df, left_index=True, right_index=True, how="outer")

# Fill NaNs only in feature columns (not title/genres)
feature_cols = list(genre_df.columns) + list(genome_vectors_df.columns)
content_features_df[feature_cols] = content_features_df[feature_cols].fillna(0)
print(f"Content features shape: {content_features_df.shape}")

valid_movie_ids = set(content_features_df.index)
print("Feature engineering complete.")
print("-" * 40)

# 4) Build triplets
print("--- [Step 4/8] Preparing Training Data (Triplets) ---")

positive_df = ratings_df[ratings_df["rating"] >= 4.0].copy()
print(f"Total positive examples (rating >= 4): {len(positive_df)}")

negative_df = ratings_df[ratings_df["rating"] <= 2.0].copy()
print(f"Total hard-negative examples (rating <= 2): {len(negative_df)}")

triplet_df = pd.merge(
    positive_df[["userId", "movieId"]],
    negative_df[["userId", "movieId"]],
    on="userId",
    suffixes=("_pos", "_neg"),
)
triplet_df = triplet_df[triplet_df["movieId_pos"] != triplet_df["movieId_neg"]]
triplet_df = triplet_df.drop_duplicates()

SAMPLE_SIZE = 5_000_000
if len(triplet_df) > SAMPLE_SIZE:
    print(f"Subsampling {SAMPLE_SIZE} triplets from {len(triplet_df)} total...")
    training_data_df = triplet_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
else:
    print("Using full triplet pool (< 5M).")
    training_data_df = triplet_df.reset_index(drop=True)

print(f"Total triplets for training: {len(training_data_df)}")
print("-" * 40)

# 5) Filtering & encoding
print("--- [Step 5/8] Filtering & Encoding Triplet Data ---")

print(f"Original training data size: {len(training_data_df)}")
training_data_df = training_data_df[
    training_data_df["movieId_pos"].isin(valid_movie_ids) &
    training_data_df["movieId_neg"].isin(valid_movie_ids)
]
print(f"Filtered training data size (both movies have features): {len(training_data_df)}")

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

user_encoder.fit(all_user_ids)
movie_encoder.fit(all_movie_ids)

training_data_df["user_id_encoded"] = user_encoder.transform(training_data_df["userId"])
training_data_df["movie_id_pos_encoded"] = movie_encoder.transform(training_data_df["movieId_pos"])
training_data_df["movie_id_neg_encoded"] = movie_encoder.transform(training_data_df["movieId_neg"])

n_users = len(user_encoder.classes_)
n_movies = len(movie_encoder.classes_)
print(f"Unique Users: {n_users}, Unique Movies: {n_movies}")
print("-" * 40)

# 6) Save API artifacts
print("--- [Step 6/8] Saving API Artifacts ---")

movie_features_for_api = content_features_df.reindex(movie_encoder.classes_, fill_value=0)
content_feature_names = list(genre_df.columns) + list(genome_vectors_df.columns)

artifacts_to_save = {
    "user_encoder.pkl": user_encoder,
    "movie_encoder.pkl": movie_encoder,
    "movie_features_df.pkl": movie_features_for_api,
    "content_feature_names.pkl": content_feature_names,
}

saved_artifacts = {}
for filename, obj in artifacts_to_save.items():
    with open(os.path.join(ARTIFACTS_DIR, filename), "wb") as f:
        pickle.dump(obj, f)
    saved_artifacts[filename] = obj
    print(f"Saved artifact: {filename}")
print("-" * 40)

# 6.5) Train SVD (re-ranker)
print("--- [Step 6.5/8] Training SVD (Re-ranker) ---")
try:
    print("Preparing SVD data...")
    svd_data_df = ratings_df[ratings_df["rating"] >= 4.0].copy()

    print("Encoding IDs for SVD...")
    svd_data_df["user_idx"] = user_encoder.transform(svd_data_df["userId"])
    svd_data_df["movie_idx"] = movie_encoder.transform(svd_data_df["movieId"])

    print(f"Building sparse matrix with {len(svd_data_df)} positives...")
    sparse_matrix = csr_matrix(
        (svd_data_df["rating"], (svd_data_df["user_idx"], svd_data_df["movie_idx"])),
        shape=(n_users, n_movies)
    )

    print("Fitting TruncatedSVD...")
    svd_model = TruncatedSVD(n_components=100, random_state=42)
    svd_model.fit(sparse_matrix)

    Q = svd_model.components_.T  # (n_movies, k)
    movie_index = {int(mid): int(idx) for idx, mid in enumerate(movie_encoder.classes_)}

    np.savez_compressed(
        os.path.join(ARTIFACTS_DIR, "svd_runtime_data.npz"),
        Q=Q,
        movie_index=movie_index
    )
    print("Exported SVD factors to svd_runtime_data.npz")

    SVD_MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "svd_model.pkl")
    with open(SVD_MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(svd_model, f)
    print(f"SVD model saved to {SVD_MODEL_SAVE_PATH}")

except Exception as e:
    print(f"[WARN] SVD training failed: {e}. Continuing with Two-Tower training.")

print("-" * 40)

# 7) Dataset & DataLoader (Triplet)
print("--- [Step 7/8] Creating Dataset and DataLoaders ---")

class MovieTripletDataset(Dataset):
    """
    Returns (user, pos_item, pos_content, neg_item, neg_content)
    """
    def __init__(self, user_ids, pos_movie_ids, neg_movie_ids, content_lookup_array):
        self.user_ids = user_ids.values
        self.pos_movie_ids = pos_movie_ids.values
        self.neg_movie_ids = neg_movie_ids.values
        self.content_lookup = content_lookup_array

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id_enc = self.user_ids[idx]
        pos_movie_id_enc = self.pos_movie_ids[idx]
        neg_movie_id_enc = self.neg_movie_ids[idx]
        pos_content_vec = self.content_lookup[pos_movie_id_enc]
        neg_content_vec = self.content_lookup[neg_movie_id_enc]
        return user_id_enc, pos_movie_id_enc, pos_content_vec, neg_movie_id_enc, neg_content_vec

print("Creating content feature lookup array...")
content_df_lookup = saved_artifacts["movie_features_df.pkl"]
encoder = saved_artifacts["movie_encoder.pkl"]
content_cols = saved_artifacts["content_feature_names.pkl"]

encoded_indices = encoder.transform(content_df_lookup.index)
content_df_lookup = content_df_lookup.set_index(encoded_indices).sort_index()
assert n_movies == len(content_df_lookup), "Encoder size and feature frame size mismatch."
content_features_array = content_df_lookup[content_cols].values.astype(np.float32)
print(f"Content lookup array shape: {content_features_array.shape}")

df_train, df_val = train_test_split(
    training_data_df,
    test_size=0.1,
    random_state=42
)

train_dataset = MovieTripletDataset(
    user_ids=df_train["user_id_encoded"],
    pos_movie_ids=df_train["movie_id_pos_encoded"],
    neg_movie_ids=df_train["movie_id_neg_encoded"],
    content_lookup_array=content_features_array
)

val_dataset = MovieTripletDataset(
    user_ids=df_val["user_id_encoded"],
    pos_movie_ids=df_val["movie_id_pos_encoded"],
    neg_movie_ids=df_val["movie_id_neg_encoded"],
    content_lookup_array=content_features_array
)

BATCH_SIZE = 1024
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

print("DataLoaders ready.")
print("-" * 40)

# 8) Train Two-Tower with TripletLoss
print("--- [Step 8/8] Training Two-Tower (Triplet Loss) ---")

N_CONTENT_FEATURES = content_features_array.shape[1]
EMBED_DIM = 32
OUTPUT_DIM = 64
N_EPOCHS = 5
LR = 1e-3
MARGIN = 1.0  # enforced distance between positives and negatives

model = TwoTowerModel(
    n_users=n_users,
    n_movies=n_movies,
    n_content_features=N_CONTENT_FEATURES,
    embed_dim=EMBED_DIM,
    output_dim=OUTPUT_DIM
)

criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
optimizer = optim.Adam(model.parameters(), lr=LR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

for epoch in range(N_EPOCHS):
    model.train()
    total_train_loss = 0.0

    for user_b, pos_movie_b, pos_content_b, neg_movie_b, neg_content_b in train_loader:
        user_b = user_b.to(device)
        pos_movie_b = pos_movie_b.to(device)
        pos_content_b = pos_content_b.to(device)
        neg_movie_b = neg_movie_b.to(device)
        neg_content_b = neg_content_b.to(device)

        optimizer.zero_grad()

        anchor_vec, positive_vec, negative_vec = model(
            user_b,
            pos_movie_b, pos_content_b,
            neg_movie_b, neg_content_b
        )

        loss = criterion(anchor_vec, positive_vec, negative_vec)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / max(1, len(train_loader))

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for user_b, pos_movie_b, pos_content_b, neg_movie_b, neg_content_b in val_loader:
            user_b = user_b.to(device)
            pos_movie_b = pos_movie_b.to(device)
            pos_content_b = pos_content_b.to(device)
            neg_movie_b = neg_movie_b.to(device)
            neg_content_b = neg_content_b.to(device)

            anchor_vec, positive_vec, negative_vec = model(
                user_b,
                pos_movie_b, pos_content_b,
                neg_movie_b, neg_content_b
            )
            loss = criterion(anchor_vec, positive_vec, negative_vec)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    print(f"Epoch {epoch+1}/{N_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print("\nTraining finished.")

# Save final model (CPU)
model.to("cpu")
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "model.pth")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print("-" * 40)
print("SCRIPT FINISHED SUCCESSFULLY!")
