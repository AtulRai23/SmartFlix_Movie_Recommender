import argparse

from .data_loader import MovieLensData
from .experiments import compare_models
from .preprocessing import clean_ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--model",
        choices=["all", "popularity", "item_knn", "user_knn", "svd"],
        default="all",
    )
    args = parser.parse_args()

    data = MovieLensData(args.data_dir)
    movies, ratings = data.load()
    ratings = clean_ratings(ratings)

    result = compare_models(ratings, movies)

    if args.model != "all":
        result = result[result["model"] == args.model]

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
