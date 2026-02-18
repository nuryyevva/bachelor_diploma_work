import numpy as np


class MaxVarianceStrategy:
    """Стратегия выбора точки с максимальной дисперсией предсказания."""

    def select_next_point(self, model, X_candidates):
        """
        Выбирает точку с максимальной дисперсией из X_candidates.

        Args:
            model: Обученная модель (должна поддерживать predict(return_std=True))
            X_candidates: Кандидаты для выбора (np.array)

        Returns:
            int: Индекс выбранной точки в массиве X_candidates
        """
        # Получаем стандартное отклонение (меру неопределенности)
        _, std = model.predict(X_candidates, return_std=True)

        # Находим индекс точки с максимальной неопределенностью
        next_idx = np.argmax(std)

        # ВАЖНО: Возвращаем чистый индекс (int), а не саму точку!
        return int(next_idx)
