import numpy as np


class RandomStrategy:
    """Стратегия случайного выбора точки из кандидатов."""

    def __init__(self, seed=None):
        """
        Инициализация случайной стратегии.

        Args:
            seed (int, optional): Seed для воспроизводимости.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def select_next_point(self, model, X_candidates):
        """
        Случайно выбирает точку из кандидатов.

        Args:
            model: Обученная модель (не используется, но требуется для интерфейса)
            X_candidates: Кандидаты для выбора (np.array)

        Returns:
            int: Индекс выбранной точки в массиве X_candidates
        """
        next_idx = np.random.randint(0, len(X_candidates))
        return int(next_idx)
