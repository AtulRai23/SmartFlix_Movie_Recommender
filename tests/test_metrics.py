from src.metrics import rmse, mae, precision_at_k, recall_at_k


def test_zero_error():
    y = [1, 2, 3]
    assert rmse(y, y) == 0
    assert mae(y, y) == 0


def test_precision():
    assert precision_at_k([1, 2, 3], [2, 4], 3) == 1 / 3


def test_recall():
    assert recall_at_k([1, 2, 3], [2, 4], 3) == 0.5
