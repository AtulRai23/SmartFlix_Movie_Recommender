# Exploratory Data Analysis


from src.data_loader import MovieLensData
from src.preprocessing import explode_genres

data = MovieLensData("data/ml-latest-small")
movies, ratings = data.load()

print(ratings.describe())
print(ratings.groupby("userId").size().describe())
print(ratings.groupby("movieId").size().describe())

genres = explode_genres(movies)
print(genres["genre"].value_counts())
