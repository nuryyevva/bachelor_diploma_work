"""
Скрипт для удобного запуска и сравнения различных стратегий адаптивных моделей.

Примеры использования:
    # Запустить все стратегии
    python experiments/run_strategies.py --all

    # Запустить конкретные стратегии
    python experiments/run_strategies.py --strategies max_variance random

    # Запустить с кастомными параметрами
    python experiments/run_strategies.py --strategies max_variance --noise_type no_noise --max_iter 30

    # Запустить одну стратегию
    python experiments/run_strategies.py --strategy farthest_point

    # Сравнение с контролем (случайные выборки 55/75/95 точек)
    python experiments/run_strategies.py --all --random_baseline

    # Статистически значимое сравнение по 10 seed
    python experiments/run_strategies.py --all --n_seeds 10

    # Сравнение ядер для farthest_point (10 seed)
    python experiments/run_strategies.py --strategy farthest_point \
        --kernels RBF Matern RationalQuadratic --n_seeds 10
"""

import os
import sys
import random
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data import load_and_split_data, find_latest_data_file
from src.utils.evaluator import make_evaluate_fn, get_experiment_bounds
from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae
from src.utils.plot_style import apply_plot_style
from src.utils.preprocessor import DataPreprocessor
from src.models.non_adaptive import NonAdaptiveModel
from src.models.adaptive import AdaptiveModel
from src.strategies import (
    get_strategy,
    MaxVarianceStrategy,
    RandomStrategy,
    FarthestPointStrategy,
    ExpectedImprovementStrategy,
    LowerConfidenceBoundStrategy,
)


# Доступные стратегии адаптивного обучения
AVAILABLE_STRATEGIES = {
    'max_variance': MaxVarianceStrategy,
    'random': RandomStrategy,
    'farthest_point': FarthestPointStrategy,
    'expected_improvement': ExpectedImprovementStrategy,
    'lower_confidence_bound': LowerConfidenceBoundStrategy,
}


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Запуск экспериментов с различными стратегиями адаптивных моделей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --all                              Запустить все стратегии
  %(prog)s --strategies max_variance random      Запустить конкретные стратегии
  %(prog)s --strategy random                  Запустить одну стратегию
  %(prog)s --all --noise_type no_noise        Все стратегии на чистых данных
  %(prog)s --all --max_iter 50                Увеличить макс. количество итераций
  %(prog)s --all --n_seeds 10                 Запуск с 10 разными seed
  %(prog)s --strategy farthest_point --kernels RBF Matern RationalQuadratic --n_seeds 10
        """
    )

    KERNEL_CHOICES = ['RBF', 'Matern', 'RationalQuadratic']

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
        choices=KERNEL_CHOICES,
        help='Тип ядра GP, если --kernels не задан (по умолчанию: RBF)'
    )
    parser.add_argument(
        '--kernels',
        nargs='+',
        type=str,
        choices=KERNEL_CHOICES,
        default=None,
        help='Список ядер для сравнения (например: RBF Matern RationalQuadratic)'
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
    parser.add_argument(
        '--random_baseline',
        action='store_true',
        help='Запустить контрольный эксперимент со случайными выборками фиксированного размера'
    )
    parser.add_argument(
        '--baseline_sizes',
        nargs='+',
        type=int,
        default=[55, 75, 95],
        help='Размеры случайных выборок для контроля (по умолчанию: 55 75 95)'
    )
    parser.add_argument(
        '--baseline_repeats',
        type=int,
        default=5,
        help='Число повторов случайной выборки для каждого размера (по умолчанию: 5)'
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=1,
        help='Число независимых запусков с разными seed (по умолчанию: 1)'
    )
    parser.add_argument(
        '--seed_start',
        type=int,
        default=42,
        help='Начальный seed; используются seed_start, seed_start+1, ... (по умолчанию: 42)'
    )
    parser.add_argument(
        '--n_candidates',
        type=int,
        default=1000,
        help='Число генерируемых кандидатов на итерацию адаптивного обучения (по умолчанию: 1000)'
    )
    parser.add_argument(
        '--noise_levels',
        nargs='+',
        type=float,
        default=[0.01, 0.03, 0.05, 0.10],
        help='Уровни шума для эксперимента чувствительности (доли, по умолчанию: 0.01 0.03 0.05 0.10)'
    )
    parser.add_argument(
        '--include_boundaries',
        action='store_true',
        help='Добавлять граничные и угловые точки в пул кандидатов (по умолчанию: False)'
    )
    parser.add_argument(
        '--discrete',
        action='store_true',
        help='Дискретный режим параметров (по умолчанию: False)'
    )
    parser.add_argument(
        '--font_scale',
        type=float,
        default=1.0,
        help='Множитель размера шрифта на графиках (по умолчанию: 1.0)'
    )

    return parser.parse_args()


def get_seeds(n_seeds: int, seed_start: int) -> List[int]:
    """Возвращает список seed для эксперимента."""
    return list(range(seed_start, seed_start + n_seeds))


def set_global_seed(seed: int):
    """Устанавливает глобальные генераторы случайных чисел."""
    np.random.seed(seed)
    random.seed(seed)


def prepare_data(
    args: argparse.Namespace,
    data_path: str,
    seed: int,
    verbose: bool = False,
    noise_type: Optional[str] = None,
    noise_level: float = 0.05,
    force_add_noise: Optional[bool] = None,
) -> Dict:
    """Загрузка, разбиение и предобработка данных для одного seed."""
    set_global_seed(seed)

    X_initial, y_initial, X_candidates, y_candidates, X_test, y_test = load_and_split_data(
        data_path=data_path,
        n_initial=args.n_initial,
        test_size=args.test_size,
        use_all_features=True,
        random_state=seed,
        verbose=verbose,
    )

    data = {
        'X_initial': X_initial,
        'y_initial': y_initial,
        'X_candidates': X_candidates,
        'y_candidates': y_candidates,
        'X_test': X_test,
        'y_test': y_test,
    }

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
    data['bounds'] = get_experiment_bounds(use_all_features=True)
    effective_noise_type = noise_type if noise_type is not None else args.noise_type
    if force_add_noise is not None:
        add_noise = force_add_noise
    else:
        add_noise = effective_noise_type == 'with_noise'
    data['evaluate_fn'] = make_evaluate_fn(
        add_noise=add_noise,
        noise_level=noise_level,
        seed=seed,
    )

    return data, preprocessor


def get_kernels_list(args: argparse.Namespace) -> List[str]:
    """Возвращает список ядер для эксперимента."""
    return args.kernels if args.kernels else [args.kernel]


def run_single_strategy(
    strategy_name: str,
    data: Dict,
    preprocessor: DataPreprocessor,
    non_adaptive_rmse: float,
    config: argparse.Namespace,
    seed: Optional[int] = None,
    kernel: Optional[str] = None,
) -> Dict:
    """Запуск одной стратегии и возврат результатов."""

    strategy_kwargs = {}
    if strategy_name == 'random' and seed is not None:
        strategy_kwargs['seed'] = seed

    strategy = get_strategy(strategy_name, **strategy_kwargs)
    kernel_name = kernel or config.kernel

    # Целевой RMSE
    target_rmse = non_adaptive_rmse * (1 - config.target_improvement)

    # Создаём адаптивную модель
    model = AdaptiveModel(
        strategy=strategy,
        kernel=kernel_name,
        max_iterations=config.max_iter,
        target_rmse=target_rmse,
        scaler_y=preprocessor.scaler_y,
        convergence_patience=2,
        n_candidates=config.n_candidates,
        include_boundaries=config.include_boundaries,
        discrete=config.discrete,
    )

    if config.verbose:
        print(f"\n  📍 Стратегия: {strategy_name}")

    results = model.train(
        X_initial=data['X_initial'],
        y_initial=data['y_initial'],
        X_test=data['X_test'],
        y_test=data['y_test_original'],
        bounds=data['bounds'],
        evaluate_fn=data['evaluate_fn'],
        scaler_X=preprocessor.scaler_X,
        seed=seed,
        verbose=config.verbose or config.n_seeds == 1,
    )

    return {
        'strategy_name': strategy_name,
        'final_rmse': results['final_rmse'],
        'final_r2': results['final_r2'],
        'final_mae': results['final_mae'],
        'n_iterations': results['n_iterations'],
        'n_final_points': results['n_final_points'],
        'total_time': results['total_time'],
        'time_per_iteration': results['time_per_iteration'],
        'rmse_history': results['rmse_history'],
        'r2_history': results['r2_history'],
        'n_points_history': results['n_points_history'],
    }


def run_single_seed_experiment(
    args: argparse.Namespace,
    strategies_to_run: List[str],
    seed: int,
    data_path: str,
    kernel: str,
) -> Dict:
    """Полный цикл эксперимента для одного seed и одного ядра."""
    verbose = args.verbose or (args.n_seeds == 1 and not args.kernels)
    if args.n_seeds > 1 or (args.kernels and len(args.kernels) > 1):
        print(f"\n--- Kernel={kernel}, Seed={seed} ---")

    data, preprocessor = prepare_data(args, data_path, seed, verbose=verbose)

    na_start = time.time()
    non_adaptive_model = NonAdaptiveModel(kernel=kernel, scaler_y=preprocessor.scaler_y)
    non_adaptive_model.train(data['X_initial'], data['y_initial'])
    non_adaptive_time = time.time() - na_start

    y_pred_na = non_adaptive_model.predict(data['X_test'])
    debug_metrics = args.verbose and seed == args.seed_start
    non_adaptive_rmse = calculate_rmse(
        y_pred_na, data['y_test_original'], debug=debug_metrics
    )
    non_adaptive_r2 = calculate_r2(y_pred_na, data['y_test_original'])

    if verbose:
        print(f"  Неадаптивная RMSE: {non_adaptive_rmse:.2f} кН/м (время: {non_adaptive_time:.2f} с)")

    all_results = []
    for strategy_name in strategies_to_run:
        result = run_single_strategy(
            strategy_name=strategy_name,
            data=data,
            preprocessor=preprocessor,
            non_adaptive_rmse=non_adaptive_rmse,
            config=args,
            seed=seed,
            kernel=kernel,
        )
        all_results.append(result)
        if verbose:
            print(f"  {strategy_name}: RMSE={result['final_rmse']:.2f}, "
                  f"итераций={result['n_iterations']}, "
                  f"время={result['total_time']:.2f} с")

    random_baseline_results = None
    if args.random_baseline:
        random_baseline_results = run_random_baseline_sizes(
            bounds=data['bounds'],
            evaluate_fn=data['evaluate_fn'],
            X_test=data['X_test'],
            y_test=data['y_test_original'],
            preprocessor=preprocessor,
            kernel=kernel,
            sizes=args.baseline_sizes,
            n_repeats=args.baseline_repeats,
            verbose=args.verbose,
            base_seed=seed,
        )

    return {
        'kernel': kernel,
        'seed': seed,
        'non_adaptive_rmse': non_adaptive_rmse,
        'non_adaptive_r2': non_adaptive_r2,
        'non_adaptive_time': non_adaptive_time,
        'all_results': all_results,
        'random_baseline': random_baseline_results,
    }


def aggregate_multi_seed_results(per_seed_results: List[Dict]) -> Dict[str, Dict]:
    """Агрегирует метрики по всем seed для каждой стратегии."""
    strategy_names = [r['strategy_name'] for r in per_seed_results[0]['all_results']]
    aggregated = {}

    for name in strategy_names:
        runs = [
            r
            for seed_res in per_seed_results
            for r in seed_res['all_results']
            if r['strategy_name'] == name
        ]

        aggregated[name] = {
            'rmse_values': [r['final_rmse'] for r in runs],
            'r2_values': [r['final_r2'] for r in runs],
            'n_iterations_values': [r['n_iterations'] for r in runs],
            'n_final_points_values': [r['n_final_points'] for r in runs],
            'total_time_values': [r['total_time'] for r in runs],
            'time_per_iter_values': [r['time_per_iteration'] for r in runs],
            'mean_rmse': float(np.mean([r['final_rmse'] for r in runs])),
            'std_rmse': float(np.std([r['final_rmse'] for r in runs])),
            'mean_r2': float(np.mean([r['final_r2'] for r in runs])),
            'std_r2': float(np.std([r['final_r2'] for r in runs])),
            'mean_iterations': float(np.mean([r['n_iterations'] for r in runs])),
            'std_iterations': float(np.std([r['n_iterations'] for r in runs])),
            'mean_total_time': float(np.mean([r['total_time'] for r in runs])),
            'std_total_time': float(np.std([r['total_time'] for r in runs])),
            'mean_time_per_iter': float(np.mean([r['time_per_iteration'] for r in runs])),
            'std_time_per_iter': float(np.std([r['time_per_iteration'] for r in runs])),
        }

    na_rmses = [s['non_adaptive_rmse'] for s in per_seed_results]
    na_times = [s['non_adaptive_time'] for s in per_seed_results]
    aggregated['_non_adaptive'] = {
        'mean_rmse': float(np.mean(na_rmses)),
        'std_rmse': float(np.std(na_rmses)),
        'rmse_values': na_rmses,
        'mean_total_time': float(np.mean(na_times)),
        'std_total_time': float(np.std(na_times)),
        'total_time_values': na_times,
    }

    return aggregated


def run_random_baseline_sizes(
    bounds: List,
    evaluate_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: DataPreprocessor,
    kernel: str = "RBF",
    sizes: List[int] = None,
    n_repeats: int = 5,
    verbose: bool = False,
    base_seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    """
    Контрольный эксперимент: GPR на случайных выборках из пространства параметров.

    Для каждого target_size генерируются случайные точки в bounds,
    y вычисляется через физическую модель (evaluate_fn).
    """
    if sizes is None:
        sizes = [55, 75, 95]

    results = {}

    for target_size in sizes:
        rmse_values = []
        for repeat_idx in range(n_repeats):
            rng = np.random.default_rng(base_seed + repeat_idx + target_size * 100)
            X_raw = AdaptiveModel._sample_candidates(bounds, target_size, rng)
            y_knm = np.array([evaluate_fn(x) for x in X_raw])

            X_train = preprocessor.scaler_X.transform(X_raw)
            y_train = preprocessor.scaler_y.transform(
                y_knm.reshape(-1, 1)
            ).ravel()

            model = NonAdaptiveModel(kernel=kernel, scaler_y=preprocessor.scaler_y)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse_values.append(calculate_rmse(y_pred, y_test))

            if verbose:
                print(
                    f"    size={target_size}, repeat={repeat_idx + 1}/{n_repeats}: "
                    f"RMSE={rmse_values[-1]:.2f} кН/м"
                )

        results[target_size] = {
            'mean_rmse': float(np.mean(rmse_values)),
            'std_rmse': float(np.std(rmse_values)),
            'rmse_values': rmse_values,
        }

    return results


def aggregate_kernel_results(all_runs: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    Агрегирует результаты по парам (стратегия, ядро) по всем seed.

    Returns:
        {strategy_name: {kernel: {mean_rmse, std_rmse, mean_iterations, ...}}}
    """
    grouped: Dict[str, Dict[str, List[Dict]]] = {}

    for run in all_runs:
        kernel = run['kernel']
        for result in run['all_results']:
            name = result['strategy_name']
            grouped.setdefault(name, {}).setdefault(kernel, []).append(result)

    aggregated = {}
    for strategy_name, kernel_runs in grouped.items():
        aggregated[strategy_name] = {}
        for kernel, runs in kernel_runs.items():
            aggregated[strategy_name][kernel] = {
                'rmse_values': [r['final_rmse'] for r in runs],
                'n_iterations_values': [r['n_iterations'] for r in runs],
                'mean_rmse': float(np.mean([r['final_rmse'] for r in runs])),
                'std_rmse': float(np.std([r['final_rmse'] for r in runs])),
                'mean_r2': float(np.mean([r['final_r2'] for r in runs])),
                'mean_iterations': float(np.mean([r['n_iterations'] for r in runs])),
                'std_iterations': float(np.std([r['n_iterations'] for r in runs])),
                'mean_total_time': float(np.mean([r['total_time'] for r in runs])),
                'mean_time_per_iter': float(np.mean([r['time_per_iteration'] for r in runs])),
            }

    return aggregated


def compare_kernels(
    kernel_aggregated: Dict[str, Dict[str, Dict]],
    kernels: List[str],
    seeds: List[int],
    save_dir: str,
    noise_type: str,
    font_scale: float = 1.0,
):
    """Таблица сравнения ядер для каждой стратегии."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"kernels_comparison_{timestamp}.txt")

    print("\n" + "=" * 95)
    print(f"📊 СРАВНЕНИЕ ЯДЕР ({noise_type}, {len(seeds)} seed)")
    print("=" * 95)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 95 + "\n")
        f.write("СРАВНЕНИЕ ЯДЕР GP\n")
        f.write(f"Тип данных: {noise_type}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Ядра: {kernels}\n")
        f.write("=" * 95 + "\n\n")

        for strategy_name in sorted(kernel_aggregated.keys()):
            print(f"\nСтратегия: {strategy_name}")
            print(
                f"{'Ядро':<22} | {'Средний RMSE':<14} | {'Стд. RMSE':<12} | "
                f"{'Сред. итер.':<12} | {'Время (с)':<10}"
            )
            print("-" * 80)

            f.write(f"Стратегия: {strategy_name}\n")
            f.write(
                f"{'Ядро':<22} | {'Средний RMSE':<14} | {'Стд. RMSE':<12} | "
                f"{'Сред. итер.':<12} | {'Время (с)':<10}\n"
            )
            f.write("-" * 80 + "\n")

            kernel_stats = kernel_aggregated[strategy_name]
            sorted_kernels = sorted(
                kernels,
                key=lambda k: kernel_stats[k]['mean_rmse'] if k in kernel_stats else float('inf')
            )

            for kernel in sorted_kernels:
                if kernel not in kernel_stats:
                    continue
                stats = kernel_stats[kernel]
                line = (
                    f"{kernel:<22} | {stats['mean_rmse']:>10.2f} кН/м | "
                    f"{stats['std_rmse']:>8.2f}   | "
                    f"{stats['mean_iterations']:>8.1f} ± {stats['std_iterations']:.1f} | "
                    f"{stats['mean_total_time']:>8.2f}"
                )
                print(line)
                f.write(line + "\n")

            f.write("\n")

            best_kernel = sorted_kernels[0]
            best = kernel_stats[best_kernel]
            summary = (
                f"  → Лучшее ядро: {best_kernel} "
                f"(RMSE {best['mean_rmse']:.2f} ± {best['std_rmse']:.2f} кН/м, "
                f"итераций {best['mean_iterations']:.1f})"
            )
            print(summary)
            f.write(summary + "\n\n")

    print("=" * 95)
    print(f"\n✅ Отчёт по ядрам сохранён: {report_path}")

    try:
        import matplotlib.pyplot as plt

        apply_plot_style(font_scale)

        for strategy_name, kernel_stats in kernel_aggregated.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            available = [k for k in kernels if k in kernel_stats]
            rmse_data = [kernel_stats[k]['rmse_values'] for k in available]
            ax.boxplot(rmse_data, labels=available, patch_artist=True)
            ax.set_ylabel('RMSE (кН/м)')
            ax.set_xlabel('Ядро')
            ax.set_title(f'Сравнение ядер: {strategy_name} ({len(seeds)} seed)')
            ax.grid(True, alpha=0.3, axis='y')

            plot_path = os.path.join(
                save_dir,
                f"kernels_boxplot_{strategy_name}_{timestamp}.png",
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Boxplot ядер ({strategy_name}): {plot_path}")

    except Exception as e:
        print(f"⚠️ Не удалось создать boxplot ядер: {e}")

    return kernel_aggregated


def compare_multi_seed(
    aggregated: Dict[str, Dict],
    per_seed_results: List[Dict],
    save_dir: str,
    seeds: List[int],
    n_initial: int = 35,
    kernel: str = "",
    include_boundaries: bool = False,
    discrete: bool = False,
    font_scale: float = 1.0,
):
    """Сравнение стратегий по нескольким seed: таблицы, boxplot, время."""
    strategy_names = [k for k in aggregated.keys() if not k.startswith('_')]
    sorted_names = sorted(
        strategy_names,
        key=lambda n: aggregated[n]['mean_rmse']
    )

    print("\n" + "=" * 100)
    print(f"📊 СРАВНЕНИЕ ПО {len(seeds)} SEED: {seeds[0]}..{seeds[-1]}")
    print("=" * 100)

    na = aggregated['_non_adaptive']
    baseline_rmse = na['mean_rmse']

    print(
        f"\n{'Стратегия':<25} | {'Средний RMSE':<14} | {'Стд. RMSE':<12} | "
        f"{'Сред. R²':<10} | {'Сред. итер.':<12} | {'Улучшение, %':<14}"
    )
    print("-" * 115)
    for name in sorted_names:
        agg = aggregated[name]
        improvement = (baseline_rmse - agg['mean_rmse']) / baseline_rmse * 100
        print(
            f"{name:<25} | {agg['mean_rmse']:>10.2f} кН/м | "
            f"{agg['std_rmse']:>8.2f}   | {agg['mean_r2']:>8.4f} | "
            f"{agg['mean_iterations']:>8.1f} ± {agg['std_iterations']:.1f} | "
            f"{improvement:>10.2f}%"
        )

    print("-" * 115)
    print(
        f"{'Неадаптивная (LHS)':<25} | {na['mean_rmse']:>10.2f} кН/м | "
        f"{na['std_rmse']:>8.2f}   | {'-':>8} | {'-':>12}"
    )
    print("=" * 100)

    print("\n" + "=" * 100)
    print("⏱️  ВРЕМЯ ОБУЧЕНИЯ")
    print("=" * 100)
    print(
        f"\n{'Стратегия':<25} | {'Общее время (с)':<18} | {'Время/итерация (с)':<20}"
    )
    print("-" * 70)
    for name in sorted_names:
        agg = aggregated[name]
        print(
            f"{name:<25} | "
            f"{agg['mean_total_time']:>6.2f} ± {agg['std_total_time']:<6.2f}   | "
            f"{agg['mean_time_per_iter']:>6.3f} ± {agg['std_time_per_iter']:.3f}"
        )
    print("-" * 70)
    print(
        f"{'Неадаптивная (LHS)':<25} | "
        f"{na['mean_total_time']:>6.2f} ± {na['std_total_time']:<6.2f}   | "
        f"{'—':>20}"
    )
    print("=" * 100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"strategies_multiseed_{timestamp}.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"СРАВНЕНИЕ СТРАТЕГИЙ ПО {len(seeds)} SEED\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Ядро: {kernel}\n")
        f.write(f"Граничные точки в кандидатах: {'да' if include_boundaries else 'нет'}\n")
        f.write(f"Дискретный режим: {'да' if discrete else 'нет'}\n")
        f.write("=" * 100 + "\n\n")

        f.write(
            f"{'Стратегия':<25} | {'Средний RMSE':<14} | {'Стд. RMSE':<12} | "
            f"{'Сред. R²':<10} | {'Сред. итер.':<12} | {'Улучшение от baseline, %':<24}\n"
        )
        f.write("-" * 115 + "\n")
        for name in sorted_names:
            agg = aggregated[name]
            improvement = (baseline_rmse - agg['mean_rmse']) / baseline_rmse * 100
            f.write(
                f"{name:<25} | {agg['mean_rmse']:>10.2f} кН/м | "
                f"{agg['std_rmse']:>8.2f}   | {agg['mean_r2']:>8.4f} | "
                f"{agg['mean_iterations']:>8.1f} ± {agg['std_iterations']:.1f} | "
                f"{improvement:>18.2f}%\n"
            )
        f.write("-" * 115 + "\n")
        f.write(
            f"{'Неадаптивная (LHS)':<25} | {na['mean_rmse']:>10.2f} кН/м | "
            f"{na['std_rmse']:>8.2f}   | {'-':>8} | {'-':>12}\n"
        )

        f.write("\n" + "=" * 100 + "\n")
        f.write("ВРЕМЯ ОБУЧЕНИЯ\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Стратегия':<25} | {'Общее время (с)':<18} | {'Время/итерация (с)':<20}\n")
        f.write("-" * 70 + "\n")
        for name in sorted_names:
            agg = aggregated[name]
            f.write(
                f"{name:<25} | "
                f"{agg['mean_total_time']:>6.2f} ± {agg['std_total_time']:<6.2f}   | "
                f"{agg['mean_time_per_iter']:>6.3f} ± {agg['std_time_per_iter']:.3f}\n"
            )
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Неадаптивная (LHS)':<25} | "
            f"{na['mean_total_time']:>6.2f} ± {na['std_total_time']:<6.2f}   | "
            f"{'—':>20}\n"
        )

    print(f"\n✅ Отчёт сохранён: {report_path}")

    try:
        import matplotlib.pyplot as plt

        apply_plot_style(font_scale)

        fig, ax = plt.subplots(figsize=(12, 6))
        rmse_data = [aggregated[name]['rmse_values'] for name in sorted_names]
        ax.boxplot(rmse_data, labels=sorted_names, patch_artist=True)
        ax.axhline(
            y=na['mean_rmse'],
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Неадаптивная (baseline): {na['mean_rmse']:.0f} кН/м",
        )
        ax.set_ylabel('RMSE (кН/м)')
        ax.set_xlabel('Стратегия')
        ax.set_title(f'Распределение RMSE по {len(seeds)} seed')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=25, ha='right')

        boxplot_path = os.path.join(save_dir, f"strategies_rmse_boxplot_{timestamp}.png")
        plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Boxplot сохранён: {boxplot_path}")

    except Exception as e:
        print(f"⚠️ Не удалось создать boxplot: {e}")

    all_results_flat = [
        res for run in per_seed_results for res in run['all_results']
    ]
    plot_convergence_all_strategies(
        all_results=all_results_flat,
        non_adaptive_rmse=aggregated['_non_adaptive']['mean_rmse'],
        save_dir=save_dir,
        per_seed_results=per_seed_results,
        title=f'Сходимость RMSE по итерациям (среднее по {len(seeds)} seed)',
        font_scale=font_scale,
    )
    plot_convergence_with_ci(
        per_seed_results=per_seed_results,
        non_adaptive_rmse=aggregated['_non_adaptive']['mean_rmse'],
        save_dir=save_dir,
        n_seeds=len(seeds),
        font_scale=font_scale,
    )

    best = sorted_names[0]
    print(f"\n🏆 Лучшая стратегия (по среднему RMSE): {best}")
    print(f"   Средний RMSE: {aggregated[best]['mean_rmse']:.2f} ± "
          f"{aggregated[best]['std_rmse']:.2f} кН/м")

    return aggregated


def _mean_rmse_history(histories: List[List[float]]) -> np.ndarray:
    """Усредняет rmse_history по seed (разной длины) по номеру итерации."""
    if not histories:
        return np.array([])
    max_len = max(len(h) for h in histories)
    padded = np.full((len(histories), max_len), np.nan)
    for i, history in enumerate(histories):
        padded[i, : len(history)] = history
    return np.nanmean(padded, axis=0)


def _mean_std_rmse_history(histories: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Среднее и std rmse_history по seed для каждой итерации."""
    if not histories:
        return np.array([]), np.array([])
    max_len = max(len(h) for h in histories)
    padded = np.full((len(histories), max_len), np.nan)
    for i, history in enumerate(histories):
        padded[i, : len(history)] = history
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def plot_convergence_with_ci(
    per_seed_results: List[Dict],
    non_adaptive_rmse: float,
    save_dir: str,
    n_seeds: int,
    font_scale: float = 1.0,
) -> Optional[str]:
    """График сходимости RMSE с доверительными интервалами ±1 std и линией baseline."""
    try:
        import matplotlib.pyplot as plt

        apply_plot_style(font_scale)
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))

        strategy_names = [
            r['strategy_name'] for r in per_seed_results[0]['all_results']
        ]
        histories_by_strategy = {name: [] for name in strategy_names}
        for run in per_seed_results:
            for res in run['all_results']:
                histories_by_strategy[res['strategy_name']].append(res['rmse_history'])

        sorted_names = sorted(
            strategy_names,
            key=lambda n: _mean_std_rmse_history(histories_by_strategy[n])[0][-1],
        )
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_names)))

        for i, name in enumerate(sorted_names):
            mean_hist, std_hist = _mean_std_rmse_history(histories_by_strategy[name])
            iterations = np.arange(1, len(mean_hist) + 1)
            ax.plot(
                iterations,
                mean_hist,
                marker='o',
                label=name,
                color=colors[i],
                linewidth=2,
                markersize=4,
            )
            ax.fill_between(
                iterations,
                mean_hist - std_hist,
                mean_hist + std_hist,
                color=colors[i],
                alpha=0.2,
            )

        ax.axhline(
            y=non_adaptive_rmse,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Неадаптивная (baseline): {non_adaptive_rmse:.2f} кН/м',
        )

        ax.set_xlabel('Адаптивная итерация')
        ax.set_ylabel('RMSE (кН/м)')
        ax.set_title(
            f'Сходимость RMSE с доверительными интервалами (±1 std, {n_seeds} seed)'
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(save_dir, 'convergence_with_ci.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ График сходимости с CI сохранён: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"⚠️ Не удалось создать график сходимости с CI: {e}")
        return None


def plot_convergence_all_strategies(
    all_results: List[Dict],
    non_adaptive_rmse: float,
    save_dir: str,
    per_seed_results: Optional[List[Dict]] = None,
    title: str = "Сходимость RMSE по итерациям",
    font_scale: float = 1.0,
) -> Optional[str]:
    """
    Строит график RMSE vs номер итерации для всех стратегий + baseline.

    При нескольких seed рисует среднюю кривую по каждой стратегии.
    """
    try:
        import matplotlib.pyplot as plt

        apply_plot_style(font_scale)
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))

        if per_seed_results and len(per_seed_results) > 1:
            strategy_names = [r['strategy_name'] for r in per_seed_results[0]['all_results']]
            histories_by_strategy = {name: [] for name in strategy_names}
            for run in per_seed_results:
                for res in run['all_results']:
                    histories_by_strategy[res['strategy_name']].append(res['rmse_history'])

            sorted_names = sorted(
                strategy_names,
                key=lambda n: _mean_rmse_history(histories_by_strategy[n])[-1],
            )
            plot_items = [
                (name, _mean_rmse_history(histories_by_strategy[name]))
                for name in sorted_names
            ]
        else:
            sorted_results = sorted(all_results, key=lambda x: x['final_rmse'])
            plot_items = [
                (res['strategy_name'], np.array(res['rmse_history']))
                for res in sorted_results
            ]

        colors = plt.cm.tab10(np.linspace(0, 1, len(plot_items)))

        for i, (name, history) in enumerate(plot_items):
            iterations = np.arange(1, len(history) + 1)
            ax.plot(
                iterations,
                history,
                marker='o',
                label=f"{name} (финал: {history[-1]:.1f} кН/м)",
                color=colors[i],
                linewidth=2,
                markersize=4,
            )

        ax.axhline(
            y=non_adaptive_rmse,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Неадаптивная (baseline): {non_adaptive_rmse:.2f} кН/м',
        )

        ax.set_xlabel('Адаптивная итерация')
        ax.set_ylabel('RMSE (кН/м)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(save_dir, 'convergence_all_strategies.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ График сходимости сохранён: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"⚠️ Не удалось создать график сходимости: {e}")
        return None


def run_strategies_mean_rmse(
    args: argparse.Namespace,
    strategies_to_run: List[str],
    seeds: List[int],
    kernel: str,
    data_path: str,
    noise_type: str,
    noise_level: float = 0.05,
    force_add_noise: Optional[bool] = None,
    verbose_label: str = "",
) -> Dict[str, float]:
    """Средний final_rmse по seed для каждой стратегии при заданной конфигурации шума."""
    strategy_rmse: Dict[str, List[float]] = {name: [] for name in strategies_to_run}

    if verbose_label:
        print(f"\n--- {verbose_label} ---")

    for seed in seeds:
        data, preprocessor = prepare_data(
            args,
            data_path,
            seed,
            verbose=False,
            noise_type=noise_type,
            noise_level=noise_level,
            force_add_noise=force_add_noise,
        )

        non_adaptive_model = NonAdaptiveModel(kernel=kernel, scaler_y=preprocessor.scaler_y)
        non_adaptive_model.train(data['X_initial'], data['y_initial'])
        y_pred_na = non_adaptive_model.predict(data['X_test'])
        non_adaptive_rmse = calculate_rmse(y_pred_na, data['y_test_original'])

        for strategy_name in strategies_to_run:
            result = run_single_strategy(
                strategy_name=strategy_name,
                data=data,
                preprocessor=preprocessor,
                non_adaptive_rmse=non_adaptive_rmse,
                config=args,
                seed=seed,
                kernel=kernel,
            )
            strategy_rmse[strategy_name].append(result['final_rmse'])

    return {name: float(np.mean(values)) for name, values in strategy_rmse.items()}


def save_noise_comparison_table(
    clean_rmse: Dict[str, float],
    noisy_rmse: Dict[str, float],
    save_dir: str,
    seeds: List[int],
    kernel: str,
):
    """Таблица сравнения RMSE на чистых и зашумлённых данных (шум 5%)."""
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'noise_comparison_table.txt')

    sorted_names = sorted(clean_rmse.keys(), key=lambda n: clean_rmse[n])

    print("\n" + "=" * 90)
    print("📊 СРАВНЕНИЕ: ЧИСТЫЕ vs ЗАШУМЛЁННЫЕ ДАННЫЕ (шум 5%)")
    print("=" * 90)
    print(
        f"\n{'Стратегия':<25} | {'RMSE (чистые)':<16} | "
        f"{'RMSE (шум 5%)':<16} | {'Рост RMSE, %':<14}"
    )
    print("-" * 90)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("СРАВНЕНИЕ ЧИСТЫХ И ЗАШУМЛЁННЫХ ДАННЫХ\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Ядро: {kernel}\n")
        f.write(f"Шум: 5% от std чистых N_cr\n\n")
        f.write(
            f"{'Стратегия':<25} | {'RMSE (чистые)':<16} | "
            f"{'RMSE (шум 5%)':<16} | {'Рост RMSE, %':<14}\n"
        )
        f.write("-" * 90 + "\n")

        for name in sorted_names:
            clean = clean_rmse[name]
            noisy = noisy_rmse[name]
            growth = (noisy - clean) / clean * 100 if clean > 0 else 0.0
            line = (
                f"{name:<25} | {clean:>12.2f} кН/м | "
                f"{noisy:>12.2f} кН/м | {growth:>10.2f}%"
            )
            print(line)
            f.write(line + "\n")

        f.write("=" * 90 + "\n")

    print("=" * 90)
    print(f"✅ Таблица сравнения шума сохранена: {report_path}")
    return report_path


def run_noise_sensitivity_experiment(
    args: argparse.Namespace,
    strategies_to_run: List[str],
    seeds: List[int],
    kernel: str,
    save_dir: str,
):
    """Эксперимент чувствительности к разным уровням шума для всех стратегий."""
    noise_levels = args.noise_levels
    data_path = find_latest_data_file(data_dir=args.data_dir, noise_type='no_noise')

    print("\n" + "=" * 80)
    print("📊 ЭКСПЕРИМЕНТ ЧУВСТВИТЕЛЬНОСТИ К ШУМУ")
    print(f"Уровни шума: {[f'{l * 100:.0f}%' for l in noise_levels]}")
    print("=" * 80)

    sensitivity: Dict[str, Dict[float, float]] = {s: {} for s in strategies_to_run}

    for level in noise_levels:
        level_results = run_strategies_mean_rmse(
            args=args,
            strategies_to_run=strategies_to_run,
            seeds=seeds,
            kernel=kernel,
            data_path=data_path,
            noise_type='no_noise',
            noise_level=level,
            force_add_noise=True,
            verbose_label=f"Уровень шума: {level * 100:.0f}%",
        )
        for strategy_name, rmse in level_results.items():
            sensitivity[strategy_name][level] = rmse

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f"noise_sensitivity_{timestamp}.txt")

    header_levels = [f"{l * 100:.0f}%" for l in noise_levels]
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ЧУВСТВИТЕЛЬНОСТЬ СТРАТЕГИЙ К УРОВНЮ ШУМА\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Ядро: {kernel}\n")
        f.write(f"Данные: no_noise (шум только в evaluate_fn)\n\n")

        col_header = " | ".join([f"{'Стратегия':<25}"] + [f"RMSE {h:>8}" for h in header_levels])
        f.write(col_header + "\n")
        f.write("-" * len(col_header) + "\n")

        for strategy_name in sorted(sensitivity.keys()):
            values = [f"{sensitivity[strategy_name][l]:>10.2f}" for l in noise_levels]
            f.write(f"{strategy_name:<25} | " + " | ".join(values) + "\n")

    print(f"✅ Отчёт чувствительности сохранён: {report_path}")

    plot_noise_sensitivity(
        sensitivity=sensitivity,
        noise_levels=noise_levels,
        save_dir=save_dir,
        font_scale=args.font_scale,
    )

    return sensitivity, report_path


def plot_noise_sensitivity(
    sensitivity: Dict[str, Dict[float, float]],
    noise_levels: List[float],
    save_dir: str,
    font_scale: float = 1.0,
):
    """Визуализация RMSE vs уровень шума для каждой стратегии."""
    try:
        import matplotlib.pyplot as plt

        apply_plot_style(font_scale)
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        noise_pct = [l * 100 for l in noise_levels]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sensitivity)))

        for i, (strategy_name, level_map) in enumerate(sorted(sensitivity.items())):
            rmse_values = [level_map[l] for l in noise_levels]
            ax.plot(
                noise_pct,
                rmse_values,
                marker='o',
                label=strategy_name,
                color=colors[i],
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel('Уровень шума, %')
        ax.set_ylabel('RMSE (кН/м)')
        ax.set_title('Чувствительность стратегий к уровню шума')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(save_dir, 'noise_sensitivity.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ График чувствительности к шуму сохранён: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"⚠️ Не удалось создать график чувствительности: {e}")
        return None


def compare_strategies(
    all_results: List[Dict],
    non_adaptive_rmse: float,
    save_dir: str,
    random_baseline: Optional[Dict[int, Dict[str, float]]] = None,
    n_initial: int = 35,
    non_adaptive_time: Optional[float] = None,
    font_scale: float = 1.0,
):
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

    print("\n" + "=" * 80)
    print("⏱️  ВРЕМЯ ОБУЧЕНИЯ")
    print("=" * 80)
    print(f"\n{'Стратегия':<25} | {'Общее время (с)':<16} | {'Время/итерация (с)':<18}")
    print("-" * 65)
    for res in sorted_results:
        print(
            f"{res['strategy_name']:<25} | {res['total_time']:>12.2f}   | "
            f"{res['time_per_iteration']:>14.3f}"
        )
    if non_adaptive_time is not None:
        print("-" * 65)
        print(f"{'Неадаптивная (LHS)':<25} | {non_adaptive_time:>12.2f}   | {'—':>14}")
    print("=" * 80)

    if random_baseline:
        print("\n" + "=" * 90)
        print("📊 СРАВНЕНИЕ: АДАПТИВНЫЕ СТРАТЕГИИ vs СЛУЧАЙНЫЕ ВЫБОРКИ (контроль)")
        print("=" * 90)
        print(
            f"\n{'Метод':<30} | {'Точек':<8} | {'RMSE (кН/м)':<18} | "
            f"{'vs random (тот же N)':<22}"
        )
        print("-" * 90)

        for res in sorted_results:
            n_pts = res['n_final_points']
            baseline = random_baseline.get(n_pts)
            if baseline:
                delta = baseline['mean_rmse'] - res['final_rmse']
                verdict = f"адаптивная лучше на {delta:.2f}" if delta > 0 else (
                    f"random лучше на {-delta:.2f}" if delta < 0 else "одинаково"
                )
            else:
                verdict = "нет контроля для этого N"

            print(
                f"{res['strategy_name']:<30} | {n_pts:<8} | "
                f"{res['final_rmse']:>8.2f}           | {verdict}"
            )

        print("-" * 90)
        print(f"{'Неадаптивная (LHS)':<30} | {n_initial:<8} | "
              f"{non_adaptive_rmse:>8.2f}           | базовая линия")
        print("-" * 90)

        for size in sorted(random_baseline.keys()):
            bl = random_baseline[size]
            print(
                f"{'Random (среднее ± std)':<30} | {size:<8} | "
                f"{bl['mean_rmse']:>6.2f} ± {bl['std_rmse']:<6.2f} | контроль"
            )

        print("=" * 90)

        best = sorted_results[0]
        best_n = best['n_final_points']
        if best_n in random_baseline:
            bl_rmse = random_baseline[best_n]['mean_rmse']
            if best['final_rmse'] < bl_rmse:
                print(
                    f"\n✅ Лучшая адаптивная стратегия ({best['strategy_name']}) "
                    f"превосходит случайную выборку из {best_n} точек "
                    f"({best['final_rmse']:.2f} vs {bl_rmse:.2f} кН/м)"
                )
            else:
                print(
                    f"\n⚠️  Случайная выборка из {best_n} точек не хуже адаптивной "
                    f"({bl_rmse:.2f} vs {best['final_rmse']:.2f} кН/м) — "
                    f"выигрыш может быть за счёт числа точек, а не адаптации"
                )

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

        f.write("\n" + "=" * 80 + "\n")
        f.write("ВРЕМЯ ОБУЧЕНИЯ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Стратегия':<25} | {'Общее время (с)':<16} | {'Время/итерация (с)':<18}\n")
        f.write("-" * 65 + "\n")
        for res in sorted_results:
            f.write(
                f"{res['strategy_name']:<25} | {res['total_time']:>12.2f}   | "
                f"{res['time_per_iteration']:>14.3f}\n"
            )
        if non_adaptive_time is not None:
            f.write("-" * 65 + "\n")
            f.write(f"{'Неадаптивная (LHS)':<25} | {non_adaptive_time:>12.2f}   | {'—':>14}\n")

        if random_baseline:
            f.write("\n" + "=" * 80 + "\n")
            f.write("КОНТРОЛЬ: СЛУЧАЙНЫЕ ВЫБОРКИ ФИКСИРОВАННОГО РАЗМЕРА\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Размер':<10} | {'Mean RMSE':<12} | {'Std RMSE':<12}\n")
            f.write("-" * 40 + "\n")
            for size in sorted(random_baseline.keys()):
                bl = random_baseline[size]
                f.write(
                    f"{size:<10} | {bl['mean_rmse']:>8.2f} кН/м | "
                    f"{bl['std_rmse']:>8.2f} кН/м\n"
                )
            f.write("\n")
            f.write(f"{'Метод':<30} | {'Точек':<8} | {'RMSE':<12}\n")
            f.write("-" * 55 + "\n")
            for res in sorted_results:
                f.write(
                    f"{res['strategy_name']:<30} | {res['n_final_points']:<8} | "
                    f"{res['final_rmse']:>8.2f} кН/м\n"
                )
            f.write("-" * 55 + "\n")
            for size in sorted(random_baseline.keys()):
                bl = random_baseline[size]
                f.write(
                    f"{'Random baseline':<30} | {size:<8} | "
                    f"{bl['mean_rmse']:>8.2f} кН/м\n"
                )

    print(f"\n✅ Отчёт сохранён: {report_path}")

    plot_convergence_all_strategies(
        all_results=sorted_results,
        non_adaptive_rmse=non_adaptive_rmse,
        save_dir=save_dir,
        font_scale=font_scale,
    )

    return sorted_results


def main():
    args = parse_args()

    if args.all:
        strategies_to_run = list(AVAILABLE_STRATEGIES.keys())
    elif args.strategy:
        strategies_to_run = [args.strategy]
    else:
        strategies_to_run = args.strategies

    kernels = get_kernels_list(args)
    seeds = get_seeds(args.n_seeds, args.seed_start)
    compare_kernels_mode = len(kernels) > 1

    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК ЭКСПЕРИМЕНТА СО СТРАТЕГИЯМИ")
    print("=" * 80)
    print(f"Стратегии: {', '.join(strategies_to_run)}")
    print(f"Тип данных: {args.noise_type}")
    print(f"Начальных точек: {args.n_initial}")
    print(f"Макс. итераций: {args.max_iter}")
    print(f"Ядра: {', '.join(kernels)}")
    print(f"Число seed: {args.n_seeds} ({seeds[0]}..{seeds[-1]})")
    print(f"Граничные точки: {'да' if args.include_boundaries else 'нет'}")
    print(f"Дискретный режим: {'да' if args.discrete else 'нет'}")
    print("=" * 80)

    print(f"\n[1/2] Загрузка данных...")
    data_path = find_latest_data_file(data_dir=args.data_dir, noise_type=args.noise_type)
    print(f"  Файл: {os.path.basename(data_path)}")

    print(f"\n[2/2] Запуск экспериментов...")
    all_runs = []
    for kernel in kernels:
        for seed in seeds:
            run_result = run_single_seed_experiment(
                args=args,
                strategies_to_run=strategies_to_run,
                seed=seed,
                data_path=data_path,
                kernel=kernel,
            )
            all_runs.append(run_result)

    if compare_kernels_mode:
        kernel_aggregated = aggregate_kernel_results(all_runs)
        compare_kernels(
            kernel_aggregated=kernel_aggregated,
            kernels=kernels,
            seeds=seeds,
            save_dir=args.save_dir,
            noise_type=args.noise_type,
            font_scale=args.font_scale,
        )

    if compare_kernels_mode:
        print(f"\n🎉 Сравнение ядер завершено ({args.n_seeds} seed)!")
        print(f"📁 Результаты: {os.path.abspath(args.save_dir)}/")
        return kernel_aggregated, all_runs

    per_seed_results = all_runs

    if args.n_seeds > 1:
        aggregated = aggregate_multi_seed_results(per_seed_results)
        compare_multi_seed(
            aggregated=aggregated,
            per_seed_results=per_seed_results,
            save_dir=args.save_dir,
            seeds=seeds,
            n_initial=args.n_initial,
            kernel=kernels[0],
            include_boundaries=args.include_boundaries,
            discrete=args.discrete,
            font_scale=args.font_scale,
        )

        clean_path = find_latest_data_file(data_dir=args.data_dir, noise_type='no_noise')
        noisy_path = find_latest_data_file(data_dir=args.data_dir, noise_type='with_noise')
        clean_rmse = run_strategies_mean_rmse(
            args, strategies_to_run, seeds, kernels[0], clean_path,
            noise_type='no_noise', noise_level=0.05, force_add_noise=False,
            verbose_label='Таблица шума: чистые данные',
        )
        noisy_rmse = run_strategies_mean_rmse(
            args, strategies_to_run, seeds, kernels[0], noisy_path,
            noise_type='with_noise', noise_level=0.05, force_add_noise=True,
            verbose_label='Таблица шума: зашумлённые данные (5%)',
        )
        save_noise_comparison_table(
            clean_rmse, noisy_rmse, args.save_dir, seeds, kernels[0],
        )
        run_noise_sensitivity_experiment(
            args, strategies_to_run, seeds, kernels[0], args.save_dir,
        )

        print(f"\n🎉 Эксперимент завершён ({args.n_seeds} seed)!")
        print(f"📁 Результаты: {os.path.abspath(args.save_dir)}/")
        return aggregated, per_seed_results

    last = per_seed_results[0]
    compare_strategies(
        last['all_results'],
        last['non_adaptive_rmse'],
        args.save_dir,
        random_baseline=last['random_baseline'],
        n_initial=args.n_initial,
        non_adaptive_time=last['non_adaptive_time'],
        font_scale=args.font_scale,
    )

    clean_path = find_latest_data_file(data_dir=args.data_dir, noise_type='no_noise')
    noisy_path = find_latest_data_file(data_dir=args.data_dir, noise_type='with_noise')
    clean_rmse = run_strategies_mean_rmse(
        args, strategies_to_run, seeds, kernels[0], clean_path,
        noise_type='no_noise', noise_level=0.05, force_add_noise=False,
        verbose_label='Таблица шума: чистые данные',
    )
    noisy_rmse = run_strategies_mean_rmse(
        args, strategies_to_run, seeds, kernels[0], noisy_path,
        noise_type='with_noise', noise_level=0.05, force_add_noise=True,
        verbose_label='Таблица шума: зашумлённые данные (5%)',
    )
    save_noise_comparison_table(
        clean_rmse, noisy_rmse, args.save_dir, seeds, kernels[0],
    )
    run_noise_sensitivity_experiment(
        args, strategies_to_run, seeds, kernels[0], args.save_dir,
    )

    print(f"\n🎉 Эксперимент завершён!")
    print(f"📁 Результаты: {os.path.abspath(args.save_dir)}/")

    return last['all_results'], last['random_baseline']


if __name__ == "__main__":
    main()
