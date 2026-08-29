import pandas as pd


class PopularityRecommender:
    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranked = None

    def fit(self, ratings, movies):
        stats = ratings.groupby("movieId")["rating"].agg(
            count="count",
            mean="mean",
        )

        eligible = stats[stats["count"] >= self.min_ratings].copy()
        if eligible.empty:
            eligible = stats.copy()

        global_mean = ratings["rating"].mean()
        m = eligible["count"].quantile(0.60)
        m = max(float(m), 1.0)

        eligible["score"] = (
            eligible["count"] / (eligible["count"] + m) * eligible["mean"]
            + m / (eligible["count"] + m) * global_mean
        )

        self.ranked = (
            eligible.reset_index()
            .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
            .sort_values(["score", "count"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return self

    def recommend(self, n=10, exclude=None):
        if self.ranked is None:
            raise RuntimeError("Call fit before recommend.")

        excluded = set(exclude or [])
        result = self.ranked[~self.ranked["movieId"].isin(excluded)]
        return result.head(n).copy()

    def score(self, movie_id):
        row = self.ranked[self.ranked["movieId"] == movie_id]
        return float(row.iloc[0]["score"]) if not row.empty else None
