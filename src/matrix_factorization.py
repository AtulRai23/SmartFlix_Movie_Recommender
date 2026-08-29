import numpy as np
import pandas as pd


class MatrixFactorization:
    def __init__(
        self,
        factors=50,
        epochs=40,
        learning_rate=0.005,
        regularization=0.04,
        random_state=42,
    ):
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state
        self.global_mean = 0.0
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.user_to_index = {}
        self.item_to_index = {}
        self.movies = None
        self.history = []

    def fit(self, ratings, movies):
        self.movies = movies

        users = np.sort(ratings["userId"].unique())
        items = np.sort(ratings["movieId"].unique())

        self.user_to_index = {u: i for i, u in enumerate(users)}
        self.item_to_index = {m: i for i, m in enumerate(items)}

        rng = np.random.default_rng(self.random_state)

        self.global_mean = float(ratings["rating"].mean())
        self.P = rng.normal(0, 0.08, (len(users), self.factors))
        self.Q = rng.normal(0, 0.08, (len(items), self.factors))
        self.bu = np.zeros(len(users))
        self.bi = np.zeros(len(items))

        records = list(
            ratings[["userId", "movieId", "rating"]].itertuples(index=False)
        )

        for epoch in range(self.epochs):
            rng.shuffle(records)
            squared_error = 0.0

            for user_id, movie_id, rating in records:
                u = self.user_to_index[user_id]
                i = self.item_to_index[movie_id]

                pu = self.P[u].copy()
                qi = self.Q[i].copy()

                prediction = (
                    self.global_mean
                    + self.bu[u]
                    + self.bi[i]
                    + np.dot(pu, qi)
                )

                error = float(rating - prediction)
                squared_error += error * error

                lr = self.learning_rate
                reg = self.regularization

                self.bu[u] += lr * (error - reg * self.bu[u])
                self.bi[i] += lr * (error - reg * self.bi[i])
                self.P[u] += lr * (error * qi - reg * pu)
                self.Q[i] += lr * (error * pu - reg * qi)

            mse = squared_error / max(len(records), 1)
            self.history.append(float(np.sqrt(mse)))

        return self

    def predict(self, user_id, movie_id):
        if user_id not in self.user_to_index or movie_id not in self.item_to_index:
            return np.nan

        u = self.user_to_index[user_id]
        i = self.item_to_index[movie_id]

        prediction = (
            self.global_mean
            + self.bu[u]
            + self.bi[i]
            + np.dot(self.P[u], self.Q[i])
        )

        return float(np.clip(prediction, 0.5, 5.0))

    def recommend(self, user_id, n=10, exclude=None):
        if user_id not in self.user_to_index:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        excluded = set(exclude or [])
        scored = []

        for movie_id in self.item_to_index:
            if movie_id in excluded:
                continue
            prediction = self.predict(user_id, movie_id)
            if np.isfinite(prediction):
                scored.append((movie_id, prediction))

        result = pd.DataFrame(scored, columns=["movieId", "score"])
        result = result.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )
        return result.sort_values("score", ascending=False).head(n)

    def latent_similarity(self, movie_id, n=10):
        if movie_id not in self.item_to_index:
            return pd.DataFrame()

        index = self.item_to_index[movie_id]
        vector = self.Q[index]
        norms = np.linalg.norm(self.Q, axis=1) * np.linalg.norm(vector)
        similarities = (self.Q @ vector) / np.maximum(norms, 1e-12)

        pairs = [
            (item, float(similarities[i]))
            for item, i in self.item_to_index.items()
            if item != movie_id
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)

        result = pd.DataFrame(pairs[:n], columns=["movieId", "similarity"])
        return result.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )
