import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_rmse(y_pred, y_true, debug=False):
    """
    Вычисляет RMSE (Root Mean Square Error) для модели.

    Args:
        y_pred: Предсказанные значения.
        y_true: Истинные значения.
        debug: Если True — печатает диагностику и проверяет единицы.

    Returns:
        float: RMSE в тех же единицах, что и y.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if debug:
        print("y_true min/max:", y_true.min(), y_true.max())
        print("y_pred min/max:", y_pred.min(), y_pred.max())
        print("RMSE:", rmse)
        y_range = y_true.max() - y_true.min()
        if y_range > 0 and rmse > y_range:
            print(
                f"⚠️ RMSE ({rmse:.2f}) превышает размах y ({y_range:.2f}) — "
                f"возможна ошибка в единицах измерения"
            )

    return rmse


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
