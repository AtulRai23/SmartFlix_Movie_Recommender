import pandas as pd

from .metrics import evaluate
from .preprocessing import split_by_user
from .popularity import PopularityRecommender
from .collaborative_filtering import CollaborativeFiltering
from .matrix_factorization import MatrixFactorization


def predict_test(model, test):
    predictions = []
    actual = []

    for row in test.itertuples(index=False):
        predictions.append(model.predict(row.userId, row.movieId))
        actual.append(row.rating)

    return evaluate(actual, predictions)


def compare_models(ratings, movies, test_size=0.2, random_state=42):
    train, test = split_by_user(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )

    models = {
        "popularity": PopularityRecommender(min_ratings=20).fit(train, movies),
        "item_knn": CollaborativeFiltering(mode="item", k=30).fit(train, movies),
        "user_knn": CollaborativeFiltering(mode="user", k=30).fit(train, movies),
        "svd": MatrixFactorization(
            factors=50,
            epochs=30,
            learning_rate=0.005,
            regularization=0.04,
            random_state=random_state,
        ).fit(train, movies),
    }

    rows = []
    for name, model in models.items():
        metrics = predict_test(model, test)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows).sort_values("rmse")
