# src/model/recommender.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    """
    Item Tower (Movie).
    Learns a representation of a movie by combining:
      - a collaborative ID embedding, and
      - content features (genres, tag genome, etc.) passed through an MLP.

    Inputs:
        movie_ids: LongTensor of shape (B,) with encoded movie IDs
        content_features: FloatTensor of shape (B, n_content_features)

    Output:
        item_vector: FloatTensor of shape (B, output_dim)
    """
    def __init__(self, n_movies: int, n_content_features: int,
                 embed_dim: int = 32, output_dim: int = 64) -> None:
        super().__init__()

        # Embedding for movie ID (collaborative component)
        self.movie_embed = nn.Embedding(n_movies, embed_dim)

        # MLP for content features (content-based component)
        self.content_mlp = nn.Sequential(
            nn.Linear(n_content_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Final projection to the shared embedding space
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, movie_ids: torch.LongTensor,
                content_features: torch.FloatTensor) -> torch.FloatTensor:
        # ID path
        movie_vector = self.movie_embed(movie_ids)

        # Content path
        content_vector = self.content_mlp(content_features)

        # Combine and project
        combined = torch.cat([movie_vector, content_vector], dim=1)
        output_vector = self.final_mlp(combined)
        return output_vector


class UserTower(nn.Module):
    """
    User Tower.
    Learns a user preference vector from a user ID embedding + MLP.

    Input:
        user_ids: LongTensor of shape (B,) with encoded user IDs
    Output:
        user_vector: FloatTensor of shape (B, output_dim)
    """
    def __init__(self, n_users: int,
                 embed_dim: int = 32, output_dim: int = 64) -> None:
        super().__init__()

        # Embedding for user ID
        self.user_embed = nn.Embedding(n_users, embed_dim)

        # MLP to transform the raw ID embedding into a preference vector
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, user_ids: torch.LongTensor) -> torch.FloatTensor:
        user_vector = self.user_embed(user_ids)
        output_vector = self.user_mlp(user_vector)
        return output_vector


class TwoTowerModel(nn.Module):
    """
    Two-Tower recommender: User tower + Item tower.
    During training you can:
      - use a pointwise/pairwise objective on the dot product between user and item vectors, or
      - use triplet loss by calling the towers appropriately in your training loop.

    The forward below returns normalized user and item vectors suitable for
    cosine similarity / dot product scoring.
    """
    def __init__(self, n_users: int, n_movies: int, n_content_features: int,
                 embed_dim: int = 32, output_dim: int = 64) -> None:
        super().__init__()
        self.user_tower = UserTower(n_users, embed_dim, output_dim)
        self.item_tower = ItemTower(n_movies, n_content_features, embed_dim, output_dim)

    def forward(self,
                user_ids: torch.LongTensor,
                movie_ids: torch.LongTensor,
                content_features: torch.FloatTensor):
        """
        Compute L2-normalized user and item vectors.

        Returns:
            user_vector: (B, output_dim), L2-normalized
            item_vector: (B, output_dim), L2-normalized
        """
        user_vector = self.user_tower(user_ids)
        item_vector = self.item_tower(movie_ids, content_features)

        # L2-normalize (helps stability and makes dot product = cosine similarity)
        user_vector = F.normalize(user_vector, p=2, dim=1)
        item_vector = F.normalize(item_vector, p=2, dim=1)

        return user_vector, item_vector
