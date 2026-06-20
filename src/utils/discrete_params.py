"""Дискретные значения входных параметров (технологические ограничения)."""

import itertools
from typing import List, Tuple

import numpy as np

from src.physics.buckling import FEATURE_BOUNDS


def get_discrete_theta(step_deg: float = 5) -> np.ndarray:
    """Углы укладки: шаг 5° в диапазоне 0–45°."""
    theta_min, theta_max = FEATURE_BOUNDS[0]
    return np.arange(theta_min, theta_max + step_deg / 2, step_deg)


def get_discrete_thickness(
    ply_thickness_mm: float = 0.2,
    min_mm: float = 2.0,
    max_mm: float = 20.0,
) -> np.ndarray:
    """
    Толщина монослоя T300/934 ≈ 0.2 мм; общая толщина = N × 0.2 мм, N — чётное.
    Шаг 0.4 мм (2 монослоя) в диапазоне 2–20 мм. Возвращает значения в метрах.
    """
    step_mm = 2 * ply_thickness_mm
    thicknesses_mm = np.arange(min_mm, max_mm + step_mm / 2, step_mm)
    return thicknesses_mm * 1e-3


def get_discrete_aspect_ratio(step: float = 0.1) -> np.ndarray:
    """Соотношение сторон a/b: шаг 0.1 в диапазоне 1.0–4.0."""
    ar_min, ar_max = FEATURE_BOUNDS[2]
    return np.arange(ar_min, ar_max + step / 2, step)


def get_all_discrete_combinations() -> np.ndarray:
    """Все комбинации дискретных параметров (полная сетка)."""
    thetas = get_discrete_theta()
    thicknesses = get_discrete_thickness()
    aspect_ratios = get_discrete_aspect_ratio()

    combos: List[Tuple[float, float, float]] = list(
        itertools.product(thetas, thicknesses, aspect_ratios)
    )
    return np.array(combos, dtype=float)
