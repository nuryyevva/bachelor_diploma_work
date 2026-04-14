"""Модуль стратегий выбора точек для адаптивных моделей."""

from src.strategies.max_variance import MaxVarianceStrategy
from src.strategies.expected_improvement import ExpectedImprovementStrategy
from src.strategies.lower_confidence_bound import LowerConfidenceBoundStrategy
from src.strategies.random_strategy import RandomStrategy
from src.strategies.farthest_point import FarthestPointStrategy

__all__ = [
    'MaxVarianceStrategy',
    'ExpectedImprovementStrategy',
    'LowerConfidenceBoundStrategy',
    'RandomStrategy',
    'FarthestPointStrategy',
]


def get_strategy(name: str, **kwargs):
    """
    Фабричная функция для получения стратегии по имени.

    Args:
        name (str): Название стратегии ('max_variance', 'expected_improvement',
                    'lower_confidence_bound', 'random', 'farthest_point')
        **kwargs: Дополнительные параметры для стратегии

    Returns:
        Объект стратегии

    Raises:
        ValueError: Если название стратегии неизвестно
    """
    strategies = {
        'max_variance': MaxVarianceStrategy,
        'expected_improvement': ExpectedImprovementStrategy,
        'lower_confidence_bound': LowerConfidenceBoundStrategy,
        'random': RandomStrategy,
        'farthest_point': FarthestPointStrategy,
    }

    if name not in strategies:
        raise ValueError(
            f"Неизвестная стратегия: {name}. "
            f"Доступные: {list(strategies.keys())}"
        )

    return strategies[name](**kwargs)
