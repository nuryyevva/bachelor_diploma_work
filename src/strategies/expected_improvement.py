import numpy as np
from scipy.special import erf


class ExpectedImprovementStrategy:
    """Стратегия выбора точки на основе Expected Improvement (EI)."""

    def __init__(self, xi=0.01):
        """
        Инициализация стратегии Expected Improvement.

        Args:
            xi (float): Параметр исследования (exploration parameter).
                        Большие значения способствуют исследованию,
                        малые — использованию (exploitation).
        """
        self.xi = xi

    def select_next_point(self, model, X_candidates):
        """
        Выбирает точку с максимальным Expected Improvement.

        Args:
            model: Обученная модель (должна поддерживать predict(return_std=True))
            X_candidates: Кандидаты для выбора (np.array)

        Returns:
            int: Индекс выбранной точки в массиве X_candidates
        """
        # Получаем предсказания и стандартное отклонение
        mu, std = model.predict(X_candidates, return_std=True)

        # Находим лучшее текущее значение (минимум, т.к. задача минимизации)
        if hasattr(model, 'y_train_max_'):
            y_best = model.y_train_max_
        else:
            y_best = np.max(model.predict(model.X_train_))

        # Вычисляем Expected Improvement
        with np.errstate(divide='ignore'):
            imp = y_best - mu - self.xi
            Z = imp / std
            ei = imp * self._norm_cdf(Z) + std * self._norm_pdf(Z)
            ei[std == 0] = 0.0

        next_idx = np.argmax(ei)
        return int(next_idx)

    @staticmethod
    def _norm_pdf(x):
        """Плотность вероятности стандартного нормального распределения."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    @staticmethod
    def _norm_cdf(x):
        """Функция распределения стандартного нормального распределения."""
        return 0.5 * (1 + erf(x / np.sqrt(2)))
