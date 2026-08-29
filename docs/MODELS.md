# Recommendation Models

## Popularity baseline

The popularity model uses:

```text
weighted_score =
    count / (count + m) * movie_mean
    + m / (count + m) * global_mean
```

This reduces the influence of movies with very few ratings.

## User-based collaborative filtering

Users are represented as rating vectors. Similar users are found using cosine similarity, and ratings from the nearest users are combined to predict an unseen rating.

## Item-based collaborative filtering

Movies are represented by the users who rated them. Similarity is calculated between movie vectors. A user's existing ratings are then used to estimate the score of unseen movies.

## Matrix factorization

The factorization model learns:

```text
r_hat(u,i) =
    mu + b_u + b_i + P_u · Q_i
```

where:

- `mu` is the global mean
- `b_u` is the user bias
- `b_i` is the item bias
- `P_u` is the user's latent vector
- `Q_i` is the movie's latent vector

The parameters are optimized with stochastic gradient descent and L2 regularization.
