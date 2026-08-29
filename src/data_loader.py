from pathlib import Path
import pandas as pd


class MovieLensData:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def _read(self, filename):
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")
        return pd.read_csv(path)

    def load_movies(self):
        movies = self._read("movies.csv")
        movies["genres"] = movies["genres"].fillna("(no genres listed)")
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
        movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int64")
        return movies

    def load_ratings(self):
        ratings = self._read("ratings.csv")
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        if "timestamp" in ratings.columns:
            ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
        return ratings

    def load_tags(self):
        path = self.data_dir / "tags.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def load(self):
        return self.load_movies(), self.load_ratings()
