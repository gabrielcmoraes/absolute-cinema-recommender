# code/inference.py

import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Import the model class (this file is copied into the SageMaker code/ folder)
from recommender import TwoTowerModel

# ===================================================================
# 1) STARTUP FUNCTION (SageMaker-compatible model_fn)
# ===================================================================
def model_fn(model_dir):
    """
    Load the model and all artifacts from the 'model/' directory.
    Pre-compute all item (movie) vectors for fast retrieval.
    """
    print("--- [DEBUG] model_fn start ---")
    print(f"--- [DEBUG] Searching files in: {model_dir}")
    print(f"--- [DEBUG] Directory contents: {os.listdir(model_dir)}")

    try:
        # Load preprocessing artifacts
        print("--- [DEBUG] Loading 'movie_encoder.pkl' ---")
        with open(os.path.join(model_dir, 'movie_encoder.pkl'), 'rb') as f:
            movie_encoder = pickle.load(f)

        print("--- [DEBUG] Loading 'movie_features_df.pkl' ---")
        with open(os.path.join(model_dir, 'movie_features_df.pkl'), 'rb') as f:
            movie_features_df = pickle.load(f)

        print("--- [DEBUG] Loading 'content_feature_names.pkl' ---")
        with open(os.path.join(model_dir, 'content_feature_names.pkl'), 'rb') as f:
            content_feature_names = pickle.load(f)
        print("--- [DEBUG] Pickle artifacts loaded ---")

        # Instantiate the model with training-time dimensions
        n_movies = len(movie_encoder.classes_)
        n_content_features = len(content_feature_names)
        n_users = 138493  # same value used in the training script

        print("--- [DEBUG] Instantiating TwoTowerModel ---")
        model_instance = TwoTowerModel(
            n_users=n_users,
            n_movies=n_movies,
            n_content_features=n_content_features,
            embed_dim=32,
            output_dim=64
        )
        print("--- [DEBUG] Architecture instantiated ---")

        # Load trained weights
        model_path = os.path.join(model_dir, 'model.pth')
        print(f"--- [DEBUG] Loading model weights from: {model_path} ---")
        model_instance.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model_instance.eval()
        print("--- [DEBUG] Model weights loaded ---")

        # --- PRECOMPUTE ALL ITEM VECTORS ---
        print("--- [DEBUG] Precomputing item vectors ---")
        item_tower = model_instance.item_tower.to('cpu')

        all_movie_ids_encoded = torch.tensor(range(n_movies), dtype=torch.long)
        all_content_features = torch.tensor(
            movie_features_df[content_feature_names].values.astype(np.float32),
            dtype=torch.float32
        )

        with torch.no_grad():
            # Ensure tensor is on CPU before converting to numpy
            all_movie_vectors = item_tower(all_movie_ids_encoded, all_content_features).cpu().numpy()

        # L2-normalize item vectors
        norms = np.linalg.norm(all_movie_vectors, axis=1, keepdims=True)
        all_movie_vectors_normalized = all_movie_vectors / norms

        print(f"--- [DEBUG] Precomputed {len(all_movie_vectors_normalized)} item vectors ---")
        print("--- [DEBUG] model_fn COMPLETED SUCCESSFULLY ---")

        # Return everything needed at inference time
        model_artifacts = {
            "model": model_instance,
            "movie_encoder": movie_encoder,
            "movie_features_df": movie_features_df,
            "all_movie_vectors": all_movie_vectors_normalized
        }
        return model_artifacts

    except Exception as e:
        # Surface startup errors in logs so the container fails fast
        print("!!!!!!!!!!!!!! FATAL ERROR IN model_fn !!!!!!!!!!!!!!")
        print(e)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise e  # crash the worker to expose the error

# ===================================================================
# 2) INFERENCE FUNCTION (SageMaker-compatible transform_fn)
# ===================================================================
def transform_fn(model_artifacts, request_body, content_type, accept_type):
    """
    Parse the request, run inference using loaded artifacts, and return a prediction.
    """
    # 1) Parse request
    if content_type == 'application/json':
        data = json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    movie_ids = data['movie_ids']
    top_n = data.get('top_n', 10)

    # 2) Extract artifacts
    movie_encoder = model_artifacts['movie_encoder']
    all_movie_vectors = model_artifacts['all_movie_vectors']
    movie_features_df = model_artifacts['movie_features_df']

    # 3) Inference logic (same as your API)
    try:
        valid_ids_encoded = movie_encoder.transform(movie_ids)
    except ValueError:
        # Any invalid ID → return a clear error
        error_response = json.dumps({"error": "One or more movie IDs are invalid."})
        return error_response, 'application/json'

    # Build a user profile by averaging liked item vectors
    liked_vectors = all_movie_vectors[valid_ids_encoded]
    user_profile_vector = np.mean(liked_vectors, axis=0).reshape(1, -1)

    # Compute affinity (cosine similarity)
    affinity_scores = cosine_similarity(user_profile_vector, all_movie_vectors)[0]

    # Rank and format
    results = pd.DataFrame({
        'movie_id_encoded': range(len(all_movie_vectors)),
        'affinity_score': affinity_scores
    })

    # Exclude the seed items
    results = results[~results['movie_id_encoded'].isin(valid_ids_encoded)]
    top_recommendations = results.nlargest(top_n, 'affinity_score')

    # Map back to original IDs
    top_recommendations['movie_id'] = movie_encoder.inverse_transform(top_recommendations['movie_id_encoded'])

    final_results_df = pd.merge(
        top_recommendations,
        movie_features_df[['title', 'genres']],
        left_on='movie_id',
        right_index=True
    )

    # 4) Build JSON response
    recommendations = []
    for _, row in final_results_df.iterrows():
        recommendations.append({
            "movie_id": int(row['movie_id']),
            "title": row['title'],
            "genres": '|'.join(row['genres']) if isinstance(row['genres'], list) else "",
            "affinity_score": float(row['affinity_score'])
        })

    response_dict = {"recommendations": recommendations}

    return json.dumps(response_dict), 'application/json'
