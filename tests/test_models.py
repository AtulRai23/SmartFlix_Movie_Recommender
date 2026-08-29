import pandas as pd

from src.collaborative_filtering import CollaborativeFiltering
from src.matrix_factorization import MatrixFactorization


def dataset():
    movies = pd.DataFrame({
        "movieId": [10, 20, 30, 40, 50],
        "title": ["A", "B", "C", "D", "E"],
        "genres": ["Drama", "Comedy", "Action", "Drama", "Comedy"],
    })

    ratings = pd.DataFrame({
        "userId": [1,1,1,2,2,2,3,3,3,4,4,4],
        "movieId": [10,20,30,10,20,40,10,30,50,20,40,50],
        "rating": [5,4,3,4,5,2,5,4,4,3,5,4],
    })
    return movies, ratings


def test_item_knn_prediction():
    movies, ratings = dataset()
    model = CollaborativeFiltering("item", k=2).fit(ratings, movies)
    prediction = model.predict(1, 40)
    assert 0.5 <= prediction <= 5.0


def test_user_knn_prediction():
    movies, ratings = dataset()
    model = CollaborativeFiltering("user", k=2).fit(ratings, movies)
    prediction = model.predict(1, 40)
    assert 0.5 <= prediction <= 5.0


def test_svd_prediction():
    movies, ratings = dataset()
    model = MatrixFactorization(
        factors=5,
        epochs=8,
        learning_rate=0.01,
        random_state=7,
    ).fit(ratings, movies)

    prediction = model.predict(1, 40)
    assert 0.5 <= prediction <= 5.0
    assert len(model.history) == 8


def test_svd_recommendations():
    movies, ratings = dataset()
    model = MatrixFactorization(
        factors=5,
        epochs=5,
        learning_rate=0.01,
        random_state=7,
    ).fit(ratings, movies)

    result = model.recommend(1, n=3, exclude={10, 20, 30})
    assert len(result) <= 3
    assert "title" in result.columns
