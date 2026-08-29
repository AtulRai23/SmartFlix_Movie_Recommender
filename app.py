import streamlit as st

from src.data_loader import MovieLensData
from src.preprocessing import clean_ratings, split_by_user
from src.popularity import PopularityRecommender
from src.collaborative_filtering import CollaborativeFiltering
from src.matrix_factorization import MatrixFactorization


st.set_page_config(page_title="SmartFlix", page_icon="🎬", layout="wide")

st.title("🎬 SmartFlix")
st.write("Explore movie recommendations using collaborative filtering and matrix factorization.")

DATA_DIR = "data/ml-latest-small"

try:
    data = MovieLensData(DATA_DIR)
    movies, ratings = data.load()
    ratings = clean_ratings(ratings)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

train, _ = split_by_user(ratings)

@st.cache_resource
def train_models(train_df, movies_df):
    popularity = PopularityRecommender(20).fit(train_df, movies_df)
    item_knn = CollaborativeFiltering("item", 30).fit(train_df, movies_df)
    svd = MatrixFactorization(40, 30, 0.005, 0.04).fit(train_df, movies_df)
    return popularity, item_knn, svd

popularity, item_knn, svd = train_models(train, movies)

st.sidebar.header("Recommendation settings")

user_id = st.sidebar.selectbox(
    "User",
    sorted(train["userId"].unique()),
)

model_name = st.sidebar.selectbox(
    "Model",
    ["SVD", "Item KNN", "Popularity"],
)

top_k = st.sidebar.slider("Recommendations", 5, 20, 10)

watched = set(train.loc[train["userId"] == user_id, "movieId"])

if model_name == "SVD":
    result = svd.recommend(user_id, top_k, watched)
elif model_name == "Item KNN":
    result = item_knn.recommend(user_id, top_k, watched)
else:
    result = popularity.recommend(top_k, watched)

st.subheader(f"Recommendations for User {user_id}")

if result.empty:
    st.warning("No recommendations available for this user.")
else:
    display = result[["title", "genres", "score"]].copy()
    display["score"] = display["score"].round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("User activity")
user_ratings = train[train["userId"] == user_id]
c1, c2, c3 = st.columns(3)
c1.metric("Ratings", len(user_ratings))
c2.metric("Average rating", f"{user_ratings.rating.mean():.2f}")
c3.metric("Movies available", len(movies))
