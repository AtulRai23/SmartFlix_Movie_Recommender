import pandas as pd

from src.preprocessing import clean_ratings, build_matrix, matrix_statistics


def test_clean_ratings_removes_duplicate_pairs():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 10, 20],
        "rating": [5.0, 4.0, 3.0],
    })
    result = clean_ratings(ratings)
    assert len(result) == 2


def test_matrix_statistics():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 10],
        "rating": [5.0, 4.0, 3.0],
    })
    stats = matrix_statistics(build_matrix(ratings))
    assert stats["users"] == 2
    assert stats["movies"] == 2
    assert stats["observed_ratings"] == 3
