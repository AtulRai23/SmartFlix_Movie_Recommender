# SmartFlix

SmartFlix is a movie recommendation system built with Python and the MovieLens dataset. It implements several recommendation approaches and provides a Streamlit interface for exploring personalized recommendations.

## Features

- MovieLens data loading and preprocessing
- Exploratory statistics and genre analysis
- Popularity-based recommendations
- User-based collaborative filtering
- Item-based collaborative filtering
- Cosine-similarity based neighborhood prediction
- SVD-style matrix factorization trained with SGD
- User and item bias terms
- RMSE and MAE evaluation
- Top-K personalized recommendations
- Cold-start fallback
- Model comparison utilities
- Streamlit application
- Unit tests

## Dataset

Download the MovieLens dataset from GroupLens:

https://grouplens.org/datasets/movielens/

For development, use the `ml-latest-small` dataset and place it under:

```text
data/ml-latest-small/
    movies.csv
    ratings.csv
    tags.csv
    links.csv
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train --data-dir data/ml-latest-small
```

Evaluate individual models:

```bash
python -m src.evaluate --data-dir data/ml-latest-small --model popularity
python -m src.evaluate --data-dir data/ml-latest-small --model item_knn
python -m src.evaluate --data-dir data/ml-latest-small --model user_knn
python -m src.evaluate --data-dir data/ml-latest-small --model svd
```

Generate recommendations:

```bash
python -m src.recommend --data-dir data/ml-latest-small --user-id 1 --model svd --top-k 10
```

Run the web application:

```bash
streamlit run app.py
```

## Architecture

```text
                    MovieLens
                       |
                       v
                 Data Loader
                       |
                       v
                 Preprocessing
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Popularity      KNN           SVD
          |         CF Models    Matrix Factorization
          +------------+-------------+
                       |
                       v
                  Evaluation
                       |
                       v
                 Recommendation
                       |
                       v
                   Streamlit
```

## Models

### Popularity

A weighted rating baseline is used to avoid ranking a movie highly only because it has very few ratings.

### Collaborative filtering

The KNN models build a user-item rating matrix and use cosine similarity. Item-based filtering finds movies similar to movies a user has already rated. User-based filtering finds users with similar rating behavior.

### Matrix factorization

The SVD-style model represents users and movies using lower-dimensional latent vectors. The prediction combines global mean, user bias, item bias, and the dot product of the latent vectors.

## Evaluation

The project reports:

- RMSE
- MAE
- number of evaluated ratings

The split is performed independently for each user so that users with sufficient history have ratings in both training and test sets.

## Project layout

```text
SmartFlix/
├── app.py
├── requirements.txt
├── config.yaml
├── data/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── popularity.py
│   ├── collaborative_filtering.py
│   ├── matrix_factorization.py
│   ├── recommender.py
│   ├── experiments.py
│   ├── train.py
│   ├── evaluate.py
│   └── recommend.py
└── tests/
    ├── test_data.py
    ├── test_metrics.py
    ├── test_popularity.py
    └── test_models.py
```
