import argparse

from .data_loader import MovieLensData
from .preprocessing import clean_ratings, split_by_user
from .popularity import PopularityRecommender
from .collaborative_filtering import CollaborativeFiltering
from .matrix_factorization import MatrixFactorization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument(
        "--model",
        choices=["popularity", "item_knn", "user_knn", "svd"],
        default="svd",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data = MovieLensData(args.data_dir)
    movies, ratings = data.load()
    ratings = clean_ratings(ratings)
    train, _ = split_by_user(ratings)

    watched = set(train.loc[train.userId == args.user_id, "movieId"])

    if args.model == "popularity":
        model = PopularityRecommender(20).fit(train, movies)
        result = model.recommend(args.top_k, watched)
    elif args.model == "item_knn":
        model = CollaborativeFiltering("item", 30).fit(train, movies)
        result = model.recommend(args.user_id, args.top_k, watched)
    elif args.model == "user_knn":
        model = CollaborativeFiltering("user", 30).fit(train, movies)
        result = model.recommend(args.user_id, args.top_k, watched)
    else:
        model = MatrixFactorization(50, 40).fit(train, movies)
        result = model.recommend(args.user_id, args.top_k, watched)

    columns = [c for c in ["movieId", "title", "genres", "score"] if c in result]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
