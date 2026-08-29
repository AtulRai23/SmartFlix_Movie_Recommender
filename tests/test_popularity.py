import pandas as pd

from src.popularity import PopularityRecommender


def test_popularity_ranking():
    movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["A", "B"],
        "genres": ["Drama", "Comedy"],
    })
    ratings = pd.DataFrame({
        "userId": [1, 2, 3, 4],
        "movieId": [1, 1, 2, 2],
        "rating": [5.0, 5.0, 2.0, 2.0],
    })

    model = PopularityRecommender(min_ratings=1).fit(ratings, movies)
    result = model.recommend(2)

    assert result.iloc[0]["movieId"] == 1
