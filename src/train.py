import argparse
import json

from .data_loader import MovieLensData
from .preprocessing import clean_ratings, split_by_user, matrix_statistics
from .matrix_factorization import MatrixFactorization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--factors", type=int, default=50)
    args = parser.parse_args()

    data = MovieLensData(args.data_dir)
    movies, ratings = data.load()
    ratings = clean_ratings(ratings)

    train, test = split_by_user(ratings)
    stats = matrix_statistics(
        train.pivot_table(index="userId", columns="movieId", values="rating")
    )

    print("Training statistics")
    print(json.dumps(stats, indent=2))
    print(f"Train ratings: {len(train)}")
    print(f"Test ratings:  {len(test)}")

    model = MatrixFactorization(
        factors=args.factors,
        epochs=args.epochs,
    ).fit(train, movies)

    print(f"Final training RMSE: {model.history[-1]:.4f}")


if __name__ == "__main__":
    main()
