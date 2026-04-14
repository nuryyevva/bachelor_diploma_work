import numpy as np
from scipy.spatial.distance import cdist


class FarthestPointStrategy:
    """Стратегия выбора точки, наиболее удалённой от уже обученных точек."""

    def __init__(self):
        """Инициализация стратегии Farthest Point."""
        pass

    def select_next_point(self, model, X_candidates):
        """
        Выбирает точку, наиболее удалённую от всех уже использованных точек.

        Args:
            model: Обученная модель (используется для получения X_train)
            X_candidates: Кандидаты для выбора (np.array)

        Returns:
            int: Индекс выбранной точки в массиве X_candidates
        """
        # Получаем уже использованные точки обучения
        if hasattr(model, 'X_train_'):
            X_train = model.X_train_
        else:
            raise ValueError("Модель не имеет атрибута X_train_")

        # Вычисляем расстояния от каждого кандидата до всех точек обучения
        distances = cdist(X_candidates, X_train, metric='euclidean')

        # Для каждого кандидата находим минимальное расстояние до обучающих точек
        min_distances = np.min(distances, axis=1)

        # Выбираем точку с максимальным минимальным расстоянием
        next_idx = np.argmax(min_distances)
        return int(next_idx)
