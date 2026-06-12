"""Обёртки для вычисления целевой переменной через физическую модель."""

from typing import Callable, Optional

import numpy as np

from src.physics.buckling import evaluate_point, get_feature_bounds


def make_evaluate_fn(
    add_noise: bool,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> Callable[[np.ndarray], float]:
    """
    Создаёт функцию evaluate_fn(X_raw) -> y в кН/м для адаптивного цикла.

    Args:
        add_noise: Добавлять ли шум к физической модели.
        noise_level: Уровень шума (доля от std чистых N_cr).
        seed: Seed для воспроизводимости шума при оценке новых точек.
    """
    rng = np.random.default_rng(seed)

    def evaluate_fn(X_raw: np.ndarray) -> float:
        return evaluate_point(
            float(X_raw[0]),
            float(X_raw[1]),
            float(X_raw[2]),
            add_noise=add_noise,
            noise_level=noise_level,
            rng=rng,
        )

    return evaluate_fn


def get_experiment_bounds(use_all_features: bool = True):
    """Возвращает bounds для эксперимента."""
    if use_all_features:
        return get_feature_bounds()
    return get_feature_bounds()[:2]
