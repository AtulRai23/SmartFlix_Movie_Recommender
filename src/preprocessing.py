import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_ratings(ratings):
    result = ratings.copy()
    result = result.dropna(subset=["userId", "movieId", "rating"])
    result = result.drop_duplicates(subset=["userId", "movieId"], keep="last")
    result["rating"] = result["rating"].clip(0.5, 5.0)
    return result


def split_by_user(ratings, test_size=0.2, random_state=42):
    train_parts = []
    test_parts = []

    for user_id, group in ratings.groupby("userId", sort=False):
        if len(group) < 2:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(len(group) * test_size)))
        n_test = min(n_test, len(group) - 1)

        train, test = train_test_split(
            group,
            test_size=n_test,
            random_state=random_state,
        )
        train_parts.append(train)
        test_parts.append(test)

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def build_matrix(ratings):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
    )


def matrix_statistics(matrix):
    total = matrix.shape[0] * matrix.shape[1]
    observed = int(matrix.notna().sum().sum())
    sparsity = 1 - observed / total if total else 0

    return {
        "users": int(matrix.shape[0]),
        "movies": int(matrix.shape[1]),
        "observed_ratings": observed,
        "sparsity": float(sparsity),
    }


def explode_genres(movies):
    rows = []
    for row in movies.itertuples(index=False):
        for genre in str(row.genres).split("|"):
            rows.append({"movieId": row.movieId, "genre": genre})
    return pd.DataFrame(rows)


def user_profiles(ratings):
    return ratings.groupby("userId").agg(
        rating_count=("rating", "count"),
        mean_rating=("rating", "mean"),
        rating_std=("rating", "std"),
    ).fillna(0)
