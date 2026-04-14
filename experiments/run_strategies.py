"""
Скрипт для удобного запуска и сравнения различных стратегий адаптивных моделей.

Примеры использования:
    # Запустить все стратегии
    python experiments/run_strategies.py --all

    # Запустить конкретные стратегии
    python experiments/run_strategies.py --strategies max_variance expected_improvement random

    # Запустить с кастомными параметрами
    python experiments/run_strategies.py --strategies max_variance lcb --noise_type no_noise --max_iter 30

    # Запустить одну стратегию
    python experiments/run_strategies.py --strategy farthest_point
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data import load_and_split_data, find_latest_data_file
from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae
from src.utils.plots import plot_convergence
from src.utils.preprocessor import DataPreprocessor
from src.models.non_adaptive import NonAdaptiveModel
from src.models.adaptive import AdaptiveModel
from src.strategies import (
    get_strategy,
    MaxVarianceStrategy,
    ExpectedImprovementStrategy,
    LowerConfidenceBoundStrategy,
    RandomStrategy,
    FarthestPointStrategy,
)


# Доступные стратегии
AVAILABLE_STRATEGIES = {
    'max_variance': MaxVarianceStrategy,
    'expected_improvement': ExpectedImprovementStrategy,
    # 'lcb': LowerConfidenceBoundStrategy,
    'lower_confidence_bound': LowerConfidenceBoundStrategy,
    'random': RandomStrategy,
    'farthest_point': FarthestPointStrategy,
}


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Запуск экспериментов с различными стратегиями адаптивных моделей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --all                              Запустить все стратегии
  %(prog)s --strategies max_variance ei       Запустить конкретные стратегии
  %(prog)s --strategy random                  Запустить одну стратегию
  %(prog)s --all --noise_type no_noise        Все стратегии на чистых данных
  %(prog)s --all --max_iter 50                Увеличить макс. количество итераций
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--all',
        action='store_true',
        help='Запустить все доступные стратегии'
    )
    group.add_argument(
        '--strategies',
        nargs='+',
        type=str,
        choices=list(AVAILABLE_STRATEGIES.keys()),
        help='Список стратегий для запуска'
    )
    group.add_argument(
        '--strategy',
        type=str,
        choices=list(AVAILABLE_STRATEGIES.keys()),
        help='Запустить одну конкретную стратегию'
    )

    parser.add_argument(
        '--noise_type',
        type=str,
        default='with_noise',
        choices=['with_noise', 'no_noise'],
        help='Тип данных: с шумом или без (по умолчанию: with_noise)'
    )
    parser.add_argument(
        '--n_initial',
        type=int,
        default=35,
        help='Количество начальных точек обучения (по умолчанию: 35)'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=20,
        help='Количество тестовых точек (по умолчанию: 20)'
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=50,
        help='Максимальное количество итераций (по умолчанию: 50)'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='RBF',
        choices=['RBF', 'Matern', 'RationalQuadratic'],
        help='Тип ядра GP (по умолчанию: RBF)'
    )
    parser.add_argument(
        '--target_improvement',
        type=float,
        default=0.30,
        help='Целевое улучшение RMSE относительно базы (по умолчанию: 0.30)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='results',
        help='Папка для сохранения результатов (по умолчанию: results)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Папка с данными (по умолчанию: data)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод в консоль'
    )

    return parser.parse_args()


def run_single_strategy(
    strategy_name: str,
    data: Dict,
    preprocessor: DataPreprocessor,
    non_adaptive_rmse: float,
    config: argparse.Namespace
) -> Dict:
    """Запуск одной стратегии и возврат результатов."""

    # Создаём стратегию
    strategy = get_strategy(strategy_name)

    # Целевой RMSE
    target_rmse = non_adaptive_rmse * (1 - config.target_improvement)

    # Создаём адаптивную модель
    model = AdaptiveModel(
        strategy=strategy,
        kernel=config.kernel,
        max_iterations=config.max_iter,
        target_rmse=target_rmse,
        scaler_y=preprocessor.scaler_y,
        convergence_patience=2
    )

    # Обучаем
    if config.verbose:
        print(f"\n  📍 Стратегия: {strategy_name}")

    results = model.train(
        X_initial=data['X_initial'],
        y_initial=data['y_initial'],
        X_candidates=data['X_candidates'],
        y_candidates=data['y_candidates'],
        X_test=data['X_test'],
        y_test=data['y_test_original']
    )

    return {
        'strategy_name': strategy_name,
        'final_rmse': results['final_rmse'],
        'final_r2': results['final_r2'],
        'final_mae': results['final_mae'],
        'n_iterations': results['n_iterations'],
        'n_final_points': results['n_final_points'],
        'rmse_history': results['rmse_history'],
        'r2_history': results['r2_history'],
        'n_points_history': results['n_points_history'],
    }


def compare_strategies(all_results: List[Dict], non_adaptive_rmse: float, save_dir: str):
    """Сравнение результатов всех стратегий и сохранение."""

    print("\n" + "=" * 80)
    print("📊 СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 80)

    # Таблица результатов
    print(f"\n{'Стратегия':<25} | {'RMSE':<12} | {'R²':<10} | {'Итераций':<10} | {'Улучшение':<10}")
    print("-" * 80)

    sorted_results = sorted(all_results, key=lambda x: x['final_rmse'])

    for res in sorted_results:
        improvement = (non_adaptive_rmse - res['final_rmse']) / non_adaptive_rmse * 100
        print(f"{res['strategy_name']:<25} | {res['final_rmse']:>8.2f} кН/м | {res['final_r2']:>8.4f} | {res['n_iterations']:>10} | {improvement:>6.2f}%")

    print("-" * 80)
    print(f"{'Неадаптивная (LHS)':<25} | {non_adaptive_rmse:>8.2f} кН/м | {'-':>8} | {'-':>10} | {'0.00':>6}%")
    print("=" * 80)

    # Лучшая стратегия
    best = sorted_results[0]
    print(f"\n🏆 Лучшая стратегия: {best['strategy_name']}")
    print(f"   Финальный RMSE: {best['final_rmse']:.2f} кН/м")
    print(f"   Итераций: {best['n_iterations']}")

    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    # Текстовый отчёт
    report_path = os.path.join(save_dir, f"strategies_comparison_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("СРАВНЕНИЕ СТРАТЕГИЙ АДАПТИВНЫХ МОДЕЛЕЙ\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Дата: {timestamp}\n")
        f.write(f"Неадаптивная модель (LHS) RMSE: {non_adaptive_rmse:.2f} кН/м\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Стратегия':<25} | {'RMSE':<12} | {'R²':<10} | {'Итераций':<10}\n")
        f.write("-" * 80 + "\n")

        for res in sorted_results:
            f.write(f"{res['strategy_name']:<25} | {res['final_rmse']:>8.2f} кН/м | {res['final_r2']:>8.4f} | {res['n_iterations']:>10}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Неадаптивная (LHS)':<25} | {non_adaptive_rmse:>8.2f} кН/м | {'-':>8} | {'-':>10}\n")
        f.write("=" * 80 + "\n")

        f.write(f"\n🏆 Лучшая стратегия: {best['strategy_name']}\n")
        f.write(f"   Финальный RMSE: {best['final_rmse']:.2f} кН/м\n")
        f.write(f"   Улучшение относительно LHS: {(non_adaptive_rmse - best['final_rmse']) / non_adaptive_rmse * 100:.2f}%\n")

    print(f"\n✅ Отчёт сохранён: {report_path}")

    # График сравнения
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_results)))

        for i, res in enumerate(sorted_results):
            ax.plot(
                res['n_points_history'],
                res['rmse_history'],
                marker='o',
                label=f"{res['strategy_name']} (RMSE={res['final_rmse']:.2f})",
                color=colors[i],
                linewidth=2,
                markersize=4
            )

        # Линия неадаптивной модели
        ax.axhline(y=non_adaptive_rmse, color='red', linestyle='--',
                   label=f'Неадаптивная (LHS): {non_adaptive_rmse:.2f}', linewidth=2)

        ax.set_xlabel('Количество точек обучения', fontsize=12)
        ax.set_ylabel('RMSE (кН/м)', fontsize=12)
        ax.set_title('Сравнение стратегий адаптивного обучения', fontsize=14)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(save_dir, f"strategies_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ График сохранён: {plot_path}")

    except Exception as e:
        print(f"⚠️ Не удалось создать график: {e}")

    return sorted_results


def main():
    args = parse_args()

    # Определяем список стратегий для запуска
    if args.all:
        strategies_to_run = list(AVAILABLE_STRATEGIES.keys())
    elif args.strategy:
        strategies_to_run = [args.strategy]
    else:
        strategies_to_run = args.strategies

    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА СО СТРАТЕГИЯМИ")
    print("=" * 80)
    print(f"Стратегии: {', '.join(strategies_to_run)}")
    print(f"Тип данных: {args.noise_type}")
    print(f"Начальных точек: {args.n_initial}")
    print(f"Макс. итераций: {args.max_iter}")
    print(f"Ядро: {args.kernel}")
    print("=" * 80)

    # Загрузка данных
    print(f"\n[1/4] Загрузка данных...")
    data_path = find_latest_data_file(data_dir=args.data_dir, noise_type=args.noise_type)
    print(f"  Файл: {os.path.basename(data_path)}")

    X_initial, y_initial, X_candidates, y_candidates, X_test, y_test = load_and_split_data(
        data_path=data_path,
        n_initial=args.n_initial,
        test_size=args.test_size,
        use_all_features=True
    )

    data = {
        'X_initial': X_initial,
        'y_initial': y_initial,
        'X_candidates': X_candidates,
        'y_candidates': y_candidates,
        'X_test': X_test,
        'y_test': y_test,
    }

    # Предобработка
    print(f"\n[2/4] Предобработка данных...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(data['X_initial'], data['y_initial'])

    X_initial, y_initial = preprocessor.transform(data['X_initial'], data['y_initial'])
    X_candidates, y_candidates = preprocessor.transform(data['X_candidates'], data['y_candidates'])
    X_test, y_test = preprocessor.transform(data['X_test'], data['y_test'])

    data['X_initial'] = X_initial
    data['y_initial'] = y_initial
    data['X_candidates'] = X_candidates
    data['y_candidates'] = y_candidates
    data['X_test'] = X_test
    data['y_test_original'] = preprocessor.get_original_y_test(y_test)

    # Неадаптивная модель (база для сравнения)
    print(f"\n[3/4] Обучение неадаптивной модели (LHS)...")
    non_adaptive_model = NonAdaptiveModel(kernel=args.kernel, scaler_y=preprocessor.scaler_y)
    non_adaptive_model.train(data['X_initial'], data['y_initial'])

    y_pred_na = non_adaptive_model.predict(data['X_test'])
    non_adaptive_rmse = calculate_rmse(y_pred_na, data['y_test_original'])
    non_adaptive_r2 = calculate_r2(y_pred_na, data['y_test_original'])

    print(f"  ✅ RMSE: {non_adaptive_rmse:.2f} кН/м")
    print(f"  ✅ R²:  {non_adaptive_r2:.4f}")

    # Запуск стратегий
    print(f"\n[4/4] Запуск адаптивных стратегий...")
    all_results = []

    for strategy_name in strategies_to_run:
        result = run_single_strategy(
            strategy_name=strategy_name,
            data=data,
            preprocessor=preprocessor,
            non_adaptive_rmse=non_adaptive_rmse,
            config=args
        )
        all_results.append(result)

    # Сравнение и визуализация
    compare_strategies(all_results, non_adaptive_rmse, args.save_dir)

    print(f"\n🎉 Эксперимент завершён!")
    print(f"📁 Результаты: {os.path.abspath(args.save_dir)}/")

    return all_results


if __name__ == "__main__":
    main()
