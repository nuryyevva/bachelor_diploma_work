"""
Эксперимент: Сравнение адаптивной (максимизация дисперсии) 
и неадаптивной (LHS) моделей Гауссовского процесса.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler

# Добавляем путь к src в sys.path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data import load_and_split_data
from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae
from src.utils.plots import plot_convergence, plot_data_distribution
from src.utils.preprocessor import DataPreprocessor
from src.models.non_adaptive import NonAdaptiveModel
from src.models.adaptive import AdaptiveModel
from src.strategies.max_variance import MaxVarianceStrategy


def find_latest_data_file(data_dir="data"):
    """
    Находит последний сгенерированный CSV-файл с данными.

    Args:
        data_dir (str): Путь к папке с данными.

    Returns:
        str: Путь к последнему CSV-файлу.

    Raises:
        FileNotFoundError: Если файлы не найдены.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Папка {data_dir} не существует. Запустите сначала generate_data.py")

    csv_files = [
        f for f in os.listdir(data_dir)
        if f.startswith("composite_plate_buckling_data") and f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "Нет сгенерированных данных. Запустите сначала generate_data.py"
        )

    # Сортируем по дате создания (или по имени, если дата в имени)
    latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
    return os.path.join(data_dir, latest_csv)


def save_results(results, save_dir="results"):
    """
    Сохраняет результаты эксперимента в файлы.

    Args:
        results (dict): Словарь с результатами.
        save_dir (str): Папка для сохранения.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Сохраняем в numpy формате
    np.save(os.path.join(save_dir, "experiment_results.npy"), results)

    # Сохраняем в текстовом формате для удобства чтения
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(save_dir, f"results_summary_{timestamp}.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: Адаптивная vs Неадаптивная модель\n")
        f.write("=" * 60 + "\n\n")

        f.write("НЕАДАПТИВНАЯ МОДЕЛЬ (LHS):\n")
        f.write(f"  RMSE: {results['non_adaptive']['rmse']:.2f} кН/м\n")
        f.write(f"  R²: {results['non_adaptive']['r2']:.4f}\n")
        f.write(f"  MAE: {results['non_adaptive']['mae']:.2f} кН/м\n")
        f.write(f"  Количество точек: {results['non_adaptive']['n_points']}\n\n")

        f.write("АДАПТИВНАЯ МОДЕЛЬ (Максимизация дисперсии):\n")
        f.write(f"  Финальный RMSE: {results['adaptive']['final_rmse']:.2f} кН/м\n")
        f.write(f"  Финальный R²: {results['adaptive']['final_r2']:.4f}\n")
        f.write(f"  Финальный MAE: {results['adaptive']['final_mae']:.2f} кН/м\n")
        f.write(f"  Количество итераций: {results['adaptive']['n_iterations']}\n")
        f.write(f"  Финальное количество точек: {results['adaptive']['n_final_points']}\n")
        f.write(f"  Точек до сходимости: {results['adaptive']['points_to_convergence']}\n\n")

        f.write("СРАВНЕНИЕ:\n")
        improvement = (
                (results['non_adaptive']['rmse'] - results['adaptive']['final_rmse']) /
                results['non_adaptive']['rmse'] * 100
        )
        f.write(f"  Улучшение RMSE: {improvement:.2f}%\n")
        f.write(
            f"  Экономия точек: {results['non_adaptive']['n_points'] - results['adaptive']['points_to_convergence']} точек\n")

    print(f"✅ Результаты сохранены в: {txt_path}")


def print_summary(results):
    """
    Выводит краткую сводку результатов в консоль.

    Args:
        results (dict): Словарь с результатами.
    """
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
    print("=" * 60)

    print("\nНЕАДАПТИВНАЯ МОДЕЛЬ (LHS):")
    print(f"  RMSE: {results['non_adaptive']['rmse']:.2f} кН/м")
    print(f"  R²: {results['non_adaptive']['r2']:.4f}")
    print(f"  MAE: {results['non_adaptive']['mae']:.2f} кН/м")
    print(f"  Количество точек: {results['non_adaptive']['n_points']}")

    print("\nАДАПТИВНАЯ МОДЕЛЬ (Максимизация дисперсии):")
    print(f"  Финальный RMSE: {results['adaptive']['final_rmse']:.2f} кН/м")
    print(f"  Финальный R²: {results['adaptive']['final_r2']:.4f}")
    print(f"  Финальный MAE: {results['adaptive']['final_mae']:.2f} кН/м")
    print(f"  Количество итераций: {results['adaptive']['n_iterations']}")
    print(f"  Финальное количество точек: {results['adaptive']['n_final_points']}")

    print("\nСРАВНЕНИЕ:")
    improvement = (
            (results['non_adaptive']['rmse'] - results['adaptive']['final_rmse']) /
            results['non_adaptive']['rmse'] * 100
    )
    print(f"  Улучшение RMSE: {improvement:.2f}%")
    print(
        f"  Экономия точек: {results['non_adaptive']['n_points'] - results['adaptive']['points_to_convergence']} точек")

    print("=" * 60)

def main():
    """
    Основная функция эксперимента.
    """
    print("🚀 Запуск эксперимента: Адаптивная vs Неадаптивная модель")
    print("=" * 60)

    # ======================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ======================
    print("\n[1/6] Загрузка данных...")
    data_path = find_latest_data_file("../data")
    print(f"  Найден файл: {os.path.basename(data_path)}")

    # Загружаем RAW данные (без нормализации)
    X_initial, y_initial, X_candidates, y_candidates, X_test, y_test = load_and_split_data(
        data_path,
        n_initial=15,
        test_size=15
    )

    print(f"  Начальный набор: {len(X_initial)} точек")
    print(f"  Пул кандидатов: {len(X_candidates)} точек")
    print(f"  Тестовый набор: {len(X_test)} точек")

    # ======================
    # 2. ПРЕДОБРАБОТКА (PREPROCESSING)
    # ======================
    print("\n[2/6] Предобработка данных (нормализация)...")

    preprocessor = DataPreprocessor()

    # Fit ТОЛЬКО на initial данных (избегаем data leakage)
    preprocessor.fit(X_initial, y_initial)

    # Transform всех наборов
    X_initial, y_initial = preprocessor.transform(X_initial, y_initial)
    X_candidates, y_candidates = preprocessor.transform(X_candidates, y_candidates)
    X_test, y_test = preprocessor.transform(X_test, y_test)

    # Сохраняем оригинальный y_test для метрик (в кН/м)
    y_test_original = preprocessor.get_original_y_test(y_test)

    print("  ✅ Данные нормализованы")
    print("  ✅ Оригинальный y_test сохранен для метрик")

    # ======================
    # 3. НЕАДАПТИВНАЯ МОДЕЛЬ
    # ======================
    print("\n[3/6] Обучение неадаптивной модели (LHS)...")
    non_adaptive = NonAdaptiveModel(kernel="RBF", scaler_y=preprocessor.scaler_y)
    non_adaptive.train(X_initial, y_initial)

    # Оцениваем качество (сравниваем предсказания модели с оригинальным y_test)
    # non_adaptive_rmse = calculate_rmse(non_adaptive, X_test, y_test_original)
    # non_adaptive_r2 = calculate_r2(non_adaptive, X_test, y_test_original)
    # non_adaptive_mae = calculate_mae(non_adaptive, X_test, y_test_original)
    y_pred_non_adaptive = non_adaptive.predict(X_test)
    non_adaptive_rmse = calculate_rmse(y_pred_non_adaptive, y_test_original)
    non_adaptive_r2 = calculate_r2(y_pred_non_adaptive, y_test_original)
    non_adaptive_mae = calculate_mae(y_pred_non_adaptive, y_test_original)

    print(f"  ✅ RMSE: {non_adaptive_rmse:.2f} кН/м")
    print(f"  ✅ R²: {non_adaptive_r2:.4f}")
    print(f"  ✅ MAE: {non_adaptive_mae:.2f} кН/м")

    # ======================
    # 4. АДАПТИВНАЯ МОДЕЛЬ
    # ======================
    print("\n[4/6] Обучение адаптивной модели (Максимизация дисперсии)...")
    adaptive = AdaptiveModel(
        strategy=MaxVarianceStrategy(),
        kernel="RBF",
        max_iterations=200,
        target_rmse=non_adaptive_rmse * 0.95,  # Цель: улучшить на 5%
        scaler_y=preprocessor.scaler_y
    )

    adaptive_results = adaptive.train(
        X_initial=X_initial,
        y_initial=y_initial,
        X_candidates=X_candidates,
        y_candidates=y_candidates,
        X_test=X_test,
        y_test=y_test_original  # Передаем нормализованный, модель сама оценит на нем
    )

    # Находим количество точек до достижения целевого RMSE
    target_reached = False
    points_to_convergence = len(X_initial)

    for i, rmse in enumerate(adaptive_results['rmse_history']):
        if rmse <= non_adaptive_rmse:
            points_to_convergence = len(X_initial) + i + 1
            target_reached = True
            break

    if not target_reached:
        points_to_convergence = adaptive_results['n_final_points']

    print(f"  ✅ Финальный RMSE: {adaptive_results['final_rmse']:.2f} кН/м")
    print(f"  ✅ Точек до сходимости: {points_to_convergence}")

    # ======================
    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ======================
    print("\n[5/6] Сохранение результатов...")

    results = {
        'non_adaptive': {
            'rmse': non_adaptive_rmse,
            'r2': non_adaptive_r2,
            'mae': non_adaptive_mae,
            'n_points': len(X_initial)
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
            'rmse_improvement_pct': (
                    (non_adaptive_rmse - adaptive_results['final_rmse']) /
                    non_adaptive_rmse * 100
            ),
            'points_saved': len(X_initial) - points_to_convergence
        }
    }

    save_results(results, "results")

    # ======================
    # 6. ВИЗУАЛИЗАЦИЯ
    # ======================
    print("\n[6/6] Создание визуализаций...")

    # График сходимости
    plot_convergence(
        adaptive_rmse_history=adaptive_results['rmse_history'],
        non_adaptive_rmse=non_adaptive_rmse,
        save_path="results/convergence_plot.png",
        title="Сравнение сходимости: Адаптивная vs Неадаптивная модель"
    )
    print("  ✅ График сходимости сохранен: results/convergence_plot.png")

    # Распределение данных
    plot_data_distribution(
        data_path=data_path,
        save_path="results/data_distribution.png"
    )
    print("  ✅ График распределения сохранен: results/data_distribution.png")

    # ======================
    # 7. ВЫВОД СВОДКИ
    # ======================
    print_summary(results)

    print("\n🎉 Эксперимент завершен успешно!")
    print(f"📁 Все результаты сохранены в папке: results/")


if __name__ == "__main__":
    main()