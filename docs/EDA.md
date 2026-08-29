# Exploratory Data Analysis

The MovieLens ratings data can be inspected with a short pandas workflow.

```python
from src.data_loader import MovieLensData
from src.preprocessing import explode_genres

data = MovieLensData("data/ml-latest-small")
movies, ratings = data.load()

print(ratings.describe())
print(ratings.groupby("userId").size().describe())
print(ratings.groupby("movieId").size().describe())

genres = explode_genres(movies)
print(genres["genre"].value_counts())
```

Useful questions to investigate:

- How are ratings distributed?
- Which movies receive the most ratings?
- How sparse is the user-item matrix?
- Which genres are most common?
- How active are users?
- Does rating behavior vary between users?
