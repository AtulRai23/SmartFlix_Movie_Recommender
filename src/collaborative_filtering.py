import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    def __init__(self, mode="item", k=30, min_similarity=0.05):
        if mode not in {"item", "user"}:
            raise ValueError("mode must be 'item' or 'user'")
        self.mode = mode
        self.k = k
        self.min_similarity = min_similarity
        self.matrix = None
        self.similarity = None
        self.movies = None
        self.user_means = None
        self.movie_means = None

    def fit(self, ratings, movies):
        self.movies = movies
        self.matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
        )

        self.user_means = self.matrix.mean(axis=1)
        self.movie_means = self.matrix.mean(axis=0)

        centered = self.matrix.sub(self.user_means, axis=0)
        centered = centered.fillna(0.0)

        if self.mode == "item":
            self.similarity = cosine_similarity(centered.T)
        else:
            self.similarity = cosine_similarity(centered)

        return self

    def _item_prediction(self, user_id, movie_id):
        if user_id not in self.matrix.index or movie_id not in self.matrix.columns:
            return np.nan

        item_index = self.matrix.columns.get_loc(movie_id)
        ratings = self.matrix.loc[user_id].dropna()
        neighbors = []

        for other_movie, rating in ratings.items():
            if other_movie == movie_id:
                continue

            other_index = self.matrix.columns.get_loc(other_movie)
            similarity = self.similarity[item_index, other_index]

            if similarity >= self.min_similarity:
                neighbors.append((similarity, rating))

        neighbors.sort(key=lambda x: x[0], reverse=True)
        neighbors = neighbors[:self.k]

        if not neighbors:
            return float(self.user_means.loc[user_id])

        weights = np.array([x[0] for x in neighbors])
        values = np.array([x[1] for x in neighbors])
        return float(np.dot(weights, values) / weights.sum())

    def _user_prediction(self, user_id, movie_id):
        if user_id not in self.matrix.index or movie_id not in self.matrix.columns:
            return np.nan

        user_index = self.matrix.index.get_loc(user_id)
        movie_ratings = self.matrix[movie_id].dropna()
        neighbors = []

        for other_user, rating in movie_ratings.items():
            if other_user == user_id:
                continue

            other_index = self.matrix.index.get_loc(other_user)
            similarity = self.similarity[user_index, other_index]

            if similarity >= self.min_similarity:
                neighbors.append((similarity, rating))

        neighbors.sort(key=lambda x: x[0], reverse=True)
        neighbors = neighbors[:self.k]

        if not neighbors:
            return float(self.movie_means.loc[movie_id])

        weights = np.array([x[0] for x in neighbors])
        values = np.array([x[1] for x in neighbors])
        return float(np.dot(weights, values) / weights.sum())

    def predict(self, user_id, movie_id):
        if self.mode == "item":
            prediction = self._item_prediction(user_id, movie_id)
        else:
            prediction = self._user_prediction(user_id, movie_id)

        if np.isfinite(prediction):
            return float(np.clip(prediction, 0.5, 5.0))
        return prediction

    def similar_items(self, movie_id, n=10):
        if self.mode != "item":
            raise ValueError("similar_items is available for item mode only")

        if movie_id not in self.matrix.columns:
            return pd.DataFrame()

        index = self.matrix.columns.get_loc(movie_id)
        scores = self.similarity[index]
        pairs = [
            (other, scores[i])
            for i, other in enumerate(self.matrix.columns)
            if other != movie_id
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)

        result = pd.DataFrame(pairs[:n], columns=["movieId", "similarity"])
        return result.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )

    def recommend(self, user_id, n=10, exclude=None):
        if user_id not in self.matrix.index:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        watched = set(exclude or self.matrix.loc[user_id].dropna().index)
        candidates = []

        for movie_id in self.matrix.columns:
            if movie_id in watched:
                continue
            score = self.predict(user_id, movie_id)
            if np.isfinite(score):
                candidates.append((movie_id, score))

        result = pd.DataFrame(candidates, columns=["movieId", "score"])
        result = result.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )
        return result.sort_values("score", ascending=False).head(n)
