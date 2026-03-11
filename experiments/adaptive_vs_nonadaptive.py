"""
Эксперимент: Сравнение адаптивной и неадаптивной моделей Гауссовского процесса.
Модульная структура для легкого переиспользования и расширения.
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data import load_and_split_data, find_latest_data_file
from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae
from src.utils.plots import plot_convergence, plot_data_distribution
from src.utils.preprocessor import DataPreprocessor
from src.models.non_adaptive import NonAdaptiveModel
from src.models.adaptive import AdaptiveModel
from src.strategies.max_variance import MaxVarianceStrategy


# ======================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ======================
class ExperimentConfig:
    """Конфигурация эксперимента для гибкого управления параметрами."""

    def __init__(
        self,
        noise_type: str = "with_noise",          # "with_noise" или "no_noise"
        use_all_features: bool = True,            # 3 признака или 2
        n_initial: int = 35,                      # Начальных точек для обучения
        test_size: int = 20,                      # Тестовых точек
        max_iterations: int = 50,                 # Макс. итераций адаптации
        target_improvement: float = 0.30,         # Целевое улучшение (5%)
        kernel: str = "RBF",                      # Тип ядра GP
        convergence_patience: int = 2             # Для стабильности
    ):
        self.noise_type = noise_type
        self.use_all_features = use_all_features
        self.n_initial = n_initial
        self.test_size = test_size
        self.max_iterations = max_iterations
        self.target_improvement = target_improvement
        self.kernel = kernel
        self.convergence_patience = convergence_patience

        # Автоматический расчёт целевого RMSE (будет установлен в runtime)
        self.target_rmse = None

    def get_feature_dim(self) -> int:
        """Возвращает размерность задачи."""
        return 3 if self.use_all_features else 2

    def get_save_suffix(self) -> str:
        """Генерирует суффикс для имён файлов результатов."""
        features = "3d" if self.use_all_features else "2d"
        noise = "noise" if self.noise_type == "with_noise" else "clean"
        return f"{features}_{noise}"


# ======================
# ФУНКЦИИ ЗАГРУЗКИ И ПОДГОТОВКИ
# ======================
def load_experiment_data(config: ExperimentConfig, data_dir: str = "../data") -> Dict:
    """
    Загружает и подготавливает данные для эксперимента.

    Returns:
        Dict с X_initial, y_initial, X_candidates, y_candidates, X_test, y_test_original
    """
    print(f"\n[1/6] Загрузка данных...")
    print(f"  Тип данных: {config.noise_type}")
    print(f"  Признаки: {config.get_feature_dim()}D")

    # Находим последний файл в нужной папке
    data_path = find_latest_data_file(data_dir=data_dir, noise_type=config.noise_type)
    print(f"  Файл: {os.path.basename(data_path)}")

    # Загружаем и разделяем данные
    X_initial, y_initial, X_candidates, y_candidates, X_test, y_test = load_and_split_data(
        data_path=data_path,
        n_initial=config.n_initial,
        test_size=config.test_size,
        use_all_features=config.use_all_features
    )

    return {
        'X_initial': X_initial,
        'y_initial': y_initial,
        'X_candidates': X_candidates,
        'y_candidates': y_candidates,
        'X_test': X_test,
        'y_test': y_test,
        'data_path': data_path
    }


def preprocess_data(data: Dict, config: ExperimentConfig) -> Tuple[Dict, DataPreprocessor]:
    """
    Нормализует данные с помощью StandardScaler.

    Returns:
        Tuple с нормализованными данными и предобработчиком
    """
    print(f"\n[2/6] Предобработка данных (нормализация)...")

    preprocessor = DataPreprocessor()
    preprocessor.fit(data['X_initial'], data['y_initial'])

    # Transform всех наборов
    X_initial, y_initial = preprocessor.transform(data['X_initial'], data['y_initial'])
    X_candidates, y_candidates = preprocessor.transform(data['X_candidates'], data['y_candidates'])
    X_test, y_test = preprocessor.transform(data['X_test'], data['y_test'])

    # Сохраняем оригинальный y_test для метрик
    y_test_original = preprocessor.get_original_y_test(y_test)

    print("  ✅ Данные нормализованы")
    print("  ✅ Оригинальный y_test сохранён для метрик")

    # Обновляем словарь данных
    data['X_initial'] = X_initial
    data['y_initial'] = y_initial
    data['X_candidates'] = X_candidates
    data['y_candidates'] = y_candidates
    data['X_test'] = X_test
    data['y_test_original'] = y_test_original

    return data, preprocessor


# ======================
# ФУНКЦИИ ОБУЧЕНИЯ МОДЕЛЕЙ
# ======================
def train_non_adaptive_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: DataPreprocessor,
    kernel: str = "RBF"
) -> Dict:
    """
    Обучает неадаптивную модель и оценивает качество.

    Returns:
        Dict с метриками и предсказаниями
    """
    print(f"\n[3/6] Обучение неадаптивной модели (LHS)...")

    model = NonAdaptiveModel(kernel=kernel, scaler_y=preprocessor.scaler_y)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'rmse': calculate_rmse(y_pred, y_test),
        'r2': calculate_r2(y_pred, y_test),
        'mae': calculate_mae(y_pred, y_test),
        'n_points': len(X_train),
        'y_pred': y_pred,
        'model': model
    }

    print(f"  ✅ RMSE: {metrics['rmse']:.2f} кН/м")
    print(f"  ✅ R²:  {metrics['r2']:.4f}")
    print(f"  ✅ MAE: {metrics['mae']:.2f} кН/м")

    return metrics


def train_adaptive_model(
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    X_candidates: np.ndarray,
    y_candidates: np.ndarray,
    X_test: np.ndarray,
    y_test_original: np.ndarray,
    preprocessor: DataPreprocessor,
    target_rmse: float,
    max_iterations: int,
    kernel: str = "RBF",
    convergence_patience: int = 3,
    strategy=None
) -> Dict:
    """
    Обучает адаптивную модель с итеративным выбором точек.

    Returns:
        Dict с историей обучения и метриками
    """
    print(f"\n[4/6] Обучение адаптивной модели...")
    print(f"  Стратегия: {strategy.__class__.__name__ if strategy else 'MaxVariance'}")
    print(f"  Целевой RMSE: {target_rmse:.2f} кН/м")

    if strategy is None:
        strategy = MaxVarianceStrategy()

    model = AdaptiveModel(
        strategy=strategy,
        kernel=kernel,
        max_iterations=max_iterations,
        target_rmse=target_rmse,
        scaler_y=preprocessor.scaler_y,
        convergence_patience=convergence_patience
    )

    results = model.train(
        X_initial=X_initial,
        y_initial=y_initial,
        X_candidates=X_candidates,
        y_candidates=y_candidates,
        X_test=X_test,
        y_test=y_test_original
    )

    print(f"  ✅ Финальный RMSE: {results['final_rmse']:.2f} кН/м")
    print(f"  ✅ Итераций: {results['n_iterations']}")
    print(f"  ✅ Точек в финале: {results['n_final_points']}")

    return results


# ======================
# ФУНКЦИИ АНАЛИЗА РЕЗУЛЬТАТОВ
# ======================
def calculate_convergence_points(
    rmse_history: List[float],
    target_rmse: float,
    n_initial: int
) -> int:
    """
    Находит количество точек, необходимое для достижения целевого RMSE.

    Returns:
        Количество точек до сходимости
    """
    for i, rmse in enumerate(rmse_history):
        if rmse <= target_rmse:
            return n_initial + i + 1
    return n_initial + len(rmse_history)


def compile_results(
    non_adaptive_metrics: Dict,
    adaptive_results: Dict,
    config: ExperimentConfig
) -> Dict:
    """
    Компилирует все результаты эксперимента в единый словарь.
    """
    points_to_convergence = calculate_convergence_points(
        adaptive_results['rmse_history'],
        config.target_rmse,
        config.n_initial
    )

    improvement_pct = (
        (non_adaptive_metrics['rmse'] - adaptive_results['final_rmse']) /
        non_adaptive_metrics['rmse'] * 100
    )

    return {
        'config': {
            'noise_type': config.noise_type,
            'feature_dim': config.get_feature_dim(),
            'n_initial': config.n_initial,
            'test_size': config.test_size,
            'kernel': config.kernel,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        'non_adaptive': {
            'rmse': non_adaptive_metrics['rmse'],
            'r2': non_adaptive_metrics['r2'],
            'mae': non_adaptive_metrics['mae'],
            'n_points': non_adaptive_metrics['n_points']
        },
        'adaptive': {
            'final_rmse': adaptive_results['final_rmse'],
            'final_r2': adaptive_results['final_r2'],
            'final_mae': adaptive_results['final_mae'],
            'rmse_history': adaptive_results['rmse_history'],
            'r2_history': adaptive_results['r2_history'],
            'mae_history': adaptive_results['mae_history'],
            'n_iterations': adaptive_results['n_iterations'],
            'n_final_points': adaptive_results['n_final_points'],
            'points_to_convergence': points_to_convergence,
            'selected_points_history': adaptive_results['selected_points_history']
        },
        'comparison': {
            'rmse_improvement_pct': improvement_pct,
            'points_saved': config.n_initial - points_to_convergence,
            'iterations_to_convergence': points_to_convergence - config.n_initial
        }
    }


# ======================
# ФУНКЦИИ СОХРАНЕНИЯ И ВИЗУАЛИЗАЦИИ
# ======================
def save_results(results: Dict, save_dir: str = "results", suffix: str = "") -> str:
    """
    Сохраняет результаты эксперимента в файлы.

    Returns:
        Путь к сохранённому файлу отчёта
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = results['config']['timestamp']
    txt_path = os.path.join(save_dir, f"results_summary_{suffix}_{timestamp}.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: Адаптивная vs Неадаптивная модель\n")
        f.write("=" * 70 + "\n\n")

        f.write("КОНФИГУРАЦИЯ:\n")
        f.write(f"  Размерность: {results['config']['feature_dim']}D\n")
        f.write(f"  Тип данных: {results['config']['noise_type']}\n")
        f.write(f"  Начальных точек: {results['config']['n_initial']}\n")
        f.write(f"  Тестовых точек: {results['config']['test_size']}\n")
        f.write(f"  Ядро: {results['config']['kernel']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("НЕАДАПТИВНАЯ МОДЕЛЬ (LHS):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  RMSE: {results['non_adaptive']['rmse']:.2f} кН/м\n")
        f.write(f"  R²:   {results['non_adaptive']['r2']:.4f}\n")
        f.write(f"  MAE:  {results['non_adaptive']['mae']:.2f} кН/м\n")
        f.write(f"  Точек: {results['non_adaptive']['n_points']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("АДАПТИВНАЯ МОДЕЛЬ (Максимизация дисперсии):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Финальный RMSE: {results['adaptive']['final_rmse']:.2f} кН/м\n")
        f.write(f"  Финальный R²:   {results['adaptive']['final_r2']:.4f}\n")
        f.write(f"  Финальный MAE:  {results['adaptive']['final_mae']:.2f} кН/м\n")
        f.write(f"  Итераций: {results['adaptive']['n_iterations']}\n")
        f.write(f"  Точек в финале: {results['adaptive']['n_final_points']}\n")
        f.write(f"  Точек до сходимости: {results['adaptive']['points_to_convergence']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("СРАВНЕНИЕ:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Улучшение RMSE: {results['comparison']['rmse_improvement_pct']:.2f}%\n")
        f.write(f"  Экономия точек: {results['comparison']['points_saved']} точек\n")
        f.write(f"  Итераций до сходимости: {results['comparison']['iterations_to_convergence']}\n")

        f.write("\n" + "=" * 70 + "\n")

    # Сохраняем в numpy для дальнейшего анализа
    np.save(os.path.join(save_dir, f"experiment_results_{suffix}_{timestamp}.npy"), results)

    print(f"✅ Результаты сохранены в: {txt_path}")
    return txt_path


def create_visualizations(results: Dict, data_path: str, save_dir: str = "results", suffix: str = ""):
    """
    Создаёт и сохраняет все визуализации эксперимента.
    """
    print(f"\n[6/6] Создание визуализаций...")

    # График сходимости
    plot_convergence(
        adaptive_rmse_history=results['adaptive']['rmse_history'],
        non_adaptive_rmse=results['non_adaptive']['rmse'],
        save_path=os.path.join(save_dir, f"convergence_plot_{suffix}.png"),
        title=f"Сравнение сходимости ({results['config']['feature_dim']}D, {results['config']['noise_type']})"
    )
    print(f"  ✅ График сходимости: convergence_plot_{suffix}.png")

    # Распределение данных
    plot_data_distribution(
        data_path=data_path,
        save_path=os.path.join(save_dir, f"data_distribution_{suffix}.png")
    )
    print(f"  ✅ Распределение данных: data_distribution_{suffix}.png")


# ======================
# ГЛАВНАЯ ФУНКЦИЯ ЭКСПЕРИМЕНТА
# ======================
def run_experiment(config: ExperimentConfig = None, data_dir: str = "../data", save_dir: str = "results"):
    """
    Запускает полный цикл эксперимента.

    Args:
        config: Конфигурация эксперимента (если None — используется default)
        data_dir: Папка с данными
        save_dir: Папка для результатов

    Returns:
        Dict с результатами эксперимента
    """
    if config is None:
        config = ExperimentConfig()

    print("\n" + "=" * 70)
    print("🚀 ЭКСПЕРИМЕНТ: Адаптивная vs Неадаптивная модель Гауссовского процесса")
    print("=" * 70)
    print(f"Конфигурация:")
    print(f"  • Размерность: {config.get_feature_dim()}D")
    print(f"  • Тип данных: {config.noise_type}")
    print(f"  • Начальных точек: {config.n_initial}")
    print(f"  • Макс. итераций: {config.max_iterations}")
    print("=" * 70)

    # 1. Загрузка данных
    data = load_experiment_data(config, data_dir)

    # 2. Предобработка
    data, preprocessor = preprocess_data(data, config)

    # 3. Неадаптивная модель
    non_adaptive_metrics = train_non_adaptive_model(
        X_train=data['X_initial'],
        y_train=data['y_initial'],
        X_test=data['X_test'],
        y_test=data['y_test_original'],
        preprocessor=preprocessor,
        kernel=config.kernel
    )

    # Устанавливаем целевой RMSE для адаптивной модели
    config.target_rmse = non_adaptive_metrics['rmse'] * (1 - config.target_improvement)

    # 4. Адаптивная модель
    adaptive_results = train_adaptive_model(
        X_initial=data['X_initial'],
        y_initial=data['y_initial'],
        X_candidates=data['X_candidates'],
        y_candidates=data['y_candidates'],
        X_test=data['X_test'],
        y_test_original=data['y_test_original'],
        preprocessor=preprocessor,
        target_rmse=config.target_rmse,
        max_iterations=config.max_iterations,
        kernel=config.kernel,
        convergence_patience=config.convergence_patience
    )

    # 5. Компиляция и сохранение результатов
    results = compile_results(non_adaptive_metrics, adaptive_results, config)
    suffix = config.get_save_suffix()
    save_results(results, save_dir, suffix)

    # 6. Визуализация
    create_visualizations(results, data['data_path'], save_dir, suffix)

    # 7. Вывод сводки
    print("\n" + "=" * 70)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 70)
    print(f"Неадаптивная модель:")
    print(f"  RMSE: {results['non_adaptive']['rmse']:.2f} кН/м")
    print(f"  R²:   {results['non_adaptive']['r2']:.4f}")
    print(f"\nАдаптивная модель:")
    print(f"  RMSE: {results['adaptive']['final_rmse']:.2f} кН/м")
    print(f"  R²:   {results['adaptive']['final_r2']:.4f}")
    print(f"  Итераций: {results['adaptive']['n_iterations']}")
    print(f"\nУлучшение: {results['comparison']['rmse_improvement_pct']:.2f}%")
    print(f"Экономия точек: {results['comparison']['points_saved']}")
    print("=" * 70)
    print(f"\n🎉 Эксперимент завершён!")
    print(f"📁 Результаты: {os.path.abspath(save_dir)}/")

    return results


# ======================
# ТОЧКА ВХОДА
# ======================
if __name__ == "__main__":
    # Запуск с конфигурацией по умолчанию (3D, зашумленные данные)
    run_experiment()
    # В main.py, после run_experiment():
    results = run_experiment()

    # Добавьте вывод истории:
    print("\n📊 ДЕТАЛЬНАЯ ИСТОРИЯ СХОДИМОСТИ:")
    print("-" * 60)
    print(f"{'Итерация':<10} | {'Точек':<10} | {'RMSE (кН/м)':<15} | {'Ниже базы?':<10}")
    print("-" * 60)

    baseline = results['non_adaptive']['rmse']
    for i, (rmse, n_points) in enumerate(zip(
            results['adaptive']['rmse_history'],
            results['adaptive']['n_points_history']
    )):
        below = "✅" if rmse <= baseline else "❌"
        print(f"{i + 1:<10} | {n_points:<10} | {rmse / 1e3:>10.2f}    | {below:<10}")

        # Покажем первые 5 и последние 5 итераций
        if i == 4:
            print("... (пропущено) ...")
        if i >= 4 and i < len(results['adaptive']['rmse_history']) - 5:
            continue

    # # Пример запуска с кастомной конфигурацией:
    # custom_config = ExperimentConfig(
    #     noise_type="no_noise",
    #     use_all_features=True,
    #     n_initial=40,
    #     max_iterations=30
    # )
    # run_experiment(config=custom_config)
