import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_rmse(y_pred, y_true):
    """
    Вычисляет RMSE (Root Mean Square Error) для модели.

    Args:
        y_pred: Предсказанные значения.
        y_true: Истинные значения.

    Returns:
        float: RMSE в тех же единицах, что и y.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_pred, y_true):
    """
    Вычисляет R² (Coefficient of Determination).

    Args:
        y_pred: Предсказанные значения.
        y_true: Истинные значения.

    Returns:
        float: R² (1.0 — идеальное предсказание, 0.0 — нет корреляции).
    """
    return r2_score(y_true, y_pred)


def calculate_mae(y_pred, y_true):
    """
    Вычисляет MAE (Mean Absolute Error).

    Args:
        y_pred: Предсказанные значения.
        y_true: Истинные значения.

    Returns:
        float: MAE в тех же единицах, что и y.
    """
    return np.mean(np.abs(y_true - y_pred))
