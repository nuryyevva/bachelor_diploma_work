import numpy as np


class LowerConfidenceBoundStrategy:
    """Стратегия выбора точки на основе Lower Confidence Bound (LCB)."""

    def __init__(self, kappa=3.0):
        """
        Инициализация стратегии LCB.

        Args:
            kappa (float): Параметр баланса между исследованием и использованием.
                           Большие значения способствуют исследованию.
        """
        self.kappa = kappa

    def select_next_point(self, model, X_candidates):
        """
        Выбирает точку с минимальным Lower Confidence Bound.

        LCB = mu - kappa * std

        Args:
            model: Обученная модель (должна поддерживать predict(return_std=True))
            X_candidates: Кандидаты для выбора (np.array)

        Returns:
            int: Индекс выбранной точки в массиве X_candidates
        """
        # Получаем предсказания и стандартное отклонение
        mu, std = model.predict(X_candidates, return_std=True)

        # Вычисляем LCB (для минимизации выбираем точку с наименьшим LCB)
        lcb = mu - self.kappa * std

        next_idx = np.argmin(lcb)
        return int(next_idx)
