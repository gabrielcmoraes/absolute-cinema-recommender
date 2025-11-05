# src/models/collaborative/train_svd_centered.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix

from src.utils.paths import ROOT, data_raw_dir, pick_existing
from src.utils.io import load_csv, save_npz


def main(args):
    """
    Train TruncatedSVD on a user-centered rating matrix (MovieLens style),
    export item factors Q (L2-normalized), and stash a MovieLens link.csv
    into the Lambda layer folder if present.
    """
    raw = data_raw_dir()
    ratings_csv = pick_existing(raw / "ratings.csv", raw / "rating.csv")

    print("[INFO] Loading ratings...")
    df = load_csv(
        ratings_csv,
        usecols=["userId", "movieId", "rating"],
        dtypes={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )

    # User centering: rating - user mean
    df["rating_centered"] = df["rating"] - df.groupby("userId")["rating"].transform("mean")

    # Build indexers
    users = np.sort(df["userId"].unique())
    items = np.sort(df["movieId"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    m2i = {m: i for i, m in enumerate(items)}

    rows = df["userId"].map(u2i).to_numpy()
    cols = df["movieId"].map(m2i).to_numpy()
    vals = df["rating_centered"].to_numpy(dtype=np.float32)

    # Sparse matrix (users x items) with centered ratings
    X = coo_matrix((vals, (rows, cols)), shape=(len(users), len(items))).tocsr()
    print(f"[INFO] Matrix: {X.shape}, nnz={X.nnz:,}")

    # Train SVD
    print(f"[INFO] Training TruncatedSVD (k={args.k})...")
    svd = TruncatedSVD(n_components=args.k, random_state=42)
    _Z = svd.fit_transform(X)          # (n_users, k) — not used further here
    Vt = svd.components_               # (k, n_items)
    Q = Vt.T                           # (n_items, k)

    # L2-normalize item factors (so dot = cosine)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)

    # Save artifacts
    out_npz = ROOT / "artifacts" / "factors" / f"svd_centered_qk{args.k}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # movie_index as a Python dict stored in npz (dtype=object)
    movie_index = {int(m): int(i) for i, m in enumerate(items)}
    save_npz(
        out_npz,
        Q=Q.astype(np.float32),
        movie_ids=items.astype(np.int32),
        movie_index=np.array(movie_index, dtype=object),
    )
    print(f"[INFO] Item factors saved to: {out_npz}")

    # If a MovieLens links file exists, copy it into the Lambda layer payload
    link_csv_src = None
    for cand in [raw / "links.csv", raw / "link.csv"]:
        if cand.exists():
            link_csv_src = cand
            break

    if link_csv_src:
        # Match your repo layout for the Lambda layer
        dest = ROOT / "deployment" / "lambda" / "layers" / "recommender-data" / "python" / "link.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(link_csv_src.read_bytes())
        print(f"[INFO] link.csv copied to: {dest}")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100, help="Number of latent factors")
    args = ap.parse_args()
    main(args)
