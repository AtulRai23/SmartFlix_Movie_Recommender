import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .data_loader import MovieLensData
from .preprocessing import explode_genres


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = MovieLensData(args.data_dir)
    movies, ratings = data.load()

    plt.figure(figsize=(8, 5))
    ratings["rating"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Rating distribution")
    plt.tight_layout()
    plt.savefig(output / "rating_distribution.png", dpi=150)
    plt.close()

    genre_counts = explode_genres(movies)["genre"].value_counts().head(15)

    plt.figure(figsize=(9, 6))
    genre_counts.sort_values().plot(kind="barh")
    plt.xlabel("Number of movies")
    plt.ylabel("Genre")
    plt.title("Most common genres")
    plt.tight_layout()
    plt.savefig(output / "genre_distribution.png", dpi=150)
    plt.close()

    activity = ratings.groupby("userId").size()

    plt.figure(figsize=(8, 5))
    activity.plot(kind="hist", bins=30)
    plt.xlabel("Ratings per user")
    plt.ylabel("Users")
    plt.title("User activity distribution")
    plt.tight_layout()
    plt.savefig(output / "user_activity.png", dpi=150)
    plt.close()

    print(f"Saved analysis plots to {output.resolve()}")


if __name__ == "__main__":
    main()
