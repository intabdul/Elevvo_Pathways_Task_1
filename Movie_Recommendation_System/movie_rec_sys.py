import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score

# Load dataset

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# Build user-item matrix

user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
user_item_matrix.fillna(0, inplace=True)

# User-based collaborative filtering

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_movies_userCF(user_id, top_n=10):
    # Find similar users
    sim_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10).index
    sim_users_ratings = user_item_matrix.loc[sim_users]

    # Weighted average of ratings by similarity
    weighted_ratings = sim_users_ratings.T.dot(user_similarity_df.loc[user_id, sim_users])
    norm_factor = user_similarity_df.loc[user_id, sim_users].sum()

    recommendations = (weighted_ratings / norm_factor).sort_values(ascending=False)

    # Exclude already watched
    watched = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = recommendations.drop(watched, errors="ignore")

    top_movies = recommendations.head(top_n).index
    return movies[movies["movieId"].isin(top_movies)][["movieId", "title"]]

# Precision@K evaluation

def precision_at_k(y_true, y_pred, k=10, threshold=4.0):
    """Evaluate precision at K (relevant if rating >= threshold)."""
    y_true = (y_true >= threshold).astype(int)
    y_pred = y_pred[:k]
    if len(y_pred) == 0:
        return 0.0
    return precision_score(y_true[:k], y_pred, zero_division=0)

# Example usage

user_id = 1  # change this to test for different users
print(f"\nTop recommendations for User {user_id}:\n")
print(recommend_movies_userCF(user_id, top_n=10))
