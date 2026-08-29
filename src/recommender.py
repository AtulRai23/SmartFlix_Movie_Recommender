import pandas as pd

from .popularity import PopularityRecommender


class SmartRecommender:
    def __init__(self, model, popularity=None):
        self.model = model
        self.popularity = popularity

    def recommend(self, user_id, ratings, n=10):
        watched = set(
            ratings.loc[ratings["userId"] == user_id, "movieId"].tolist()
        )

        if user_id not in getattr(self.model, "matrix", getattr(self.model, "user_to_index", {})):
            return self.popularity.recommend(n=n, exclude=watched)

        result = self.model.recommend(user_id, n=n, exclude=watched)

        if len(result) < n and self.popularity is not None:
            existing = set(result["movieId"])
            extra = self.popularity.recommend(
                n=n * 2,
                exclude=watched | existing,
            )
            result = pd.concat([result, extra], ignore_index=True).head(n)

        return result


def build_popularity(ratings, movies, min_ratings=20):
    return PopularityRecommender(min_ratings=min_ratings).fit(ratings, movies)
