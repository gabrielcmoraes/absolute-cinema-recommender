# src/models/collaborative/train_svd_grid.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

from src.utils.paths import ROOT, data_raw_dir, pick_existing
from src.utils.io import load_csv, save_npz


def build_matrix(center_user: bool = True, binarize: bool = False):
    """
    Build a sparse (users x items) matrix from MovieLens-style ratings.

    Args:
        center_user: If True, subtract user mean rating (user-centered).
        binarize: If True, convert ratings to implicit feedback (>=4 → 1.0, else 0.0).

    Returns:
        X: csr_matrix of shape (n_users, n_items)
        items: np.ndarray of sorted MovieLens movieIds (maps columns → movieId)
    """
    raw = data_raw_dir()
    ratings_csv = pick_existing(raw / "ratings.csv", raw / "rating.csv")

    # Load minimal columns
    df = load_csv(ratings_csv, usecols=["userId", "movieId", "rating"])

    # Optional binarization (implicit feedback)
    if binarize:
        df["rating"] = (df["rating"] >= 4.0).astype("float32")

    # Optional user-mean centering
    if center_user:
        df["rating"] = df["rating"] - df.groupby("userId")["rating"].transform("mean")

    # Build indexers
    users = np.sort(df["userId"].unique())
    items = np.sort(df["movieId"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    m2i = {m: i for i, m in enumerate(items)}

    rows = df["userId"].map(u2i).to_numpy()
    cols = df["movieId"].map(m2i).to_numpy()
    vals = df["rating"].astype("float32").to_numpy()

    # Sparse matrix
    X = coo_matrix((vals, (rows, cols)), shape=(len(users), len(items))).tocsr()
    return X, items


def train_and_save(X, items, k: int, tag: str):
    """
    Train TruncatedSVD for a given k and save L2-normalized item factors Q.

    Artifacts saved to: artifacts/factors/svd_{tag}_qk{k}.npz
    Contents:
      - Q (n_items, k) float32 L2-normalized
      - movie_ids (n_items,) int32
      - movie_index: dict{movieId -> row index in Q} as np.array(dtype=object)
    """
    print(f"[INFO] Training TruncatedSVD (k={k}) for tag={tag} ...")
    svd = TruncatedSVD(n_components=k, random_state=42)
    Vt = svd.fit(X).components_          # (k, n_items)
    Q = Vt.T                              # (n_items, k)

    # L2-normalize so dot ≈ cosine
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)

    out = ROOT / "artifacts" / "factors" / f"svd_{tag}_qk{k}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    movie_index = {int(m): int(i) for i, m in enumerate(items)}
    save_npz(
        out,
        Q=Q.astype(np.float32),
        movie_ids=items.astype(np.int32),
        movie_index=np.array(movie_index, dtype=object),
    )
    print(f"[OK] Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a grid of TruncatedSVD models over rating matrix variants.")
    ap.add_argument(
        "--k_list",
        type=str,
        default="64,100,128,160",
        help="Comma-separated latent dimensions to train, e.g. '64,100,128,160'.",
    )
    ap.add_argument(
        "--center_user",
        action="store_true",
        help="Apply user-mean centering before SVD.",
    )
    ap.add_argument(
        "--binarize",
        action="store_true",
        help="Binarize ratings as implicit feedback (>=4 → 1.0, else 0.0).",
    )
    args = ap.parse_args()

    X, items = build_matrix(center_user=args.center_user, binarize=args.binarize)
    tag = f"{'ctr' if args.center_user else 'noc'}_{'bin' if args.binarize else 'raw'}"

    ks = [int(x) for x in args.k_list.split(",")]
    print(f"[INFO] Matrix shape={X.shape}, nnz={X.nnz:,}; Training ks={ks} with tag={tag}")
    for k in ks:
        train_and_save(X, items, k, tag)
