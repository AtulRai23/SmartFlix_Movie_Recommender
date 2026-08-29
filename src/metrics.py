import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred)

    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "count": 0}

    return {
        "rmse": rmse(y_true[mask], y_pred[mask]),
        "mae": mae(y_true[mask], y_pred[mask]),
        "count": int(mask.sum()),
    }


def precision_at_k(recommended, relevant, k=10):
    recommended = list(recommended)[:k]
    if not recommended:
        return 0.0
    relevant = set(relevant)
    return sum(item in relevant for item in recommended) / len(recommended)


def recall_at_k(recommended, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    recommended = list(recommended)[:k]
    return sum(item in relevant for item in recommended) / len(relevant)
