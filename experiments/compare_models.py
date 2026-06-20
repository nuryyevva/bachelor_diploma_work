"""
Сравнение суррогатных моделей на начальной выборке (35 точек LHS).

Модели:
  - Полиномиальная регрессия 2-й степени (с взаимодействиями)
  - Random Forest Regressor (n_estimators=100)
  - GPR с ядром Matérn (ν=1.5)

Пример запуска:
    poetry run python experiments/compare_models.py
    poetry run python experiments/compare_models.py --n_seeds 10 --noise_type with_noise
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.models.non_adaptive import NonAdaptiveModel
from src.utils.data import load_and_split_data, find_latest_data_file
from src.utils.metrics import calculate_rmse, calculate_r2
from src.utils.preprocessor import DataPreprocessor

MODEL_NAMES = [
    "polynomial_deg2",
    "random_forest",
    "gpr_matern",
]

MODEL_LABELS = {
    "polynomial_deg2": "Полиномиальная (deg=2)",
    "random_forest": "Random Forest (100)",
    "gpr_matern": "GPR (Matérn ν=1.5)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сравнение суррогатных моделей на начальной выборке LHS"
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="with_noise",
        choices=["with_noise", "no_noise"],
    )
    parser.add_argument("--n_initial", type=int, default=35)
    parser.add_argument("--test_size", type=int, default=20)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--rf_random_state", type=int, default=42)
    return parser.parse_args()


def get_seeds(n_seeds: int, seed_start: int) -> List[int]:
    return list(range(seed_start, seed_start + n_seeds))


def train_polynomial(X_train, y_train):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int, random_state: int):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_gpr_matern(X_train, y_train, scaler_y):
    model = NonAdaptiveModel(kernel="Matern", scaler_y=scaler_y)
    model.train(X_train, y_train)
    return model


def predict_in_original_units(model, X_test, preprocessor, model_type: str) -> np.ndarray:
    if model_type == "gpr_matern":
        return model.predict(X_test)

    y_pred_norm = model.predict(X_test)
    return preprocessor.inverse_transform_y(y_pred_norm)


def run_single_seed(
    seed: int,
    data_path: str,
    n_initial: int,
    test_size: int,
    rf_n_estimators: int,
    rf_random_state: int,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    X_initial, y_initial, _, _, X_test, y_test = load_and_split_data(
        data_path=data_path,
        n_initial=n_initial,
        test_size=test_size,
        use_all_features=True,
        random_state=seed,
        verbose=False,
    )

    preprocessor = DataPreprocessor()
    preprocessor.fit(X_initial, y_initial)

    X_train, y_train = preprocessor.transform(X_initial, y_initial)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)
    y_test_original = preprocessor.get_original_y_test(y_test_scaled)

    models = {
        "polynomial_deg2": train_polynomial(X_train, y_train),
        "random_forest": train_random_forest(
            X_train, y_train, rf_n_estimators, rf_random_state
        ),
        "gpr_matern": train_gpr_matern(X_train, y_train, preprocessor.scaler_y),
    }

    results = {}
    for name, model in models.items():
        y_pred = predict_in_original_units(
            model, X_test_scaled, preprocessor, name
        )
        results[name] = {
            "rmse": calculate_rmse(y_pred, y_test_original),
            "r2": calculate_r2(y_pred, y_test_original),
            "seed": seed,
        }
        if verbose:
            print(
                f"  seed={seed}, {MODEL_LABELS[name]}: "
                f"RMSE={results[name]['rmse']:.2f}, R²={results[name]['r2']:.4f}"
            )

    return results


def aggregate_results(per_seed: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict]:
    aggregated = {}
    for name in MODEL_NAMES:
        rmses = [run[name]["rmse"] for run in per_seed]
        r2s = [run[name]["r2"] for run in per_seed]
        aggregated[name] = {
            "rmse_values": rmses,
            "r2_values": r2s,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
        }
    return aggregated


def print_summary_table(aggregated: Dict[str, Dict], seeds: List[int]):
    print("\n" + "=" * 88)
    print("=== Сравнение суррогатных моделей на начальной выборке (35 точек LHS) ===")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} запусков)")
    print("=" * 88)
    print(
        f"{'Модель':<28} | {'RMSE (среднее)':<16} | {'RMSE (std)':<12} | "
        f"{'R² (среднее)':<14} | {'R² (std)':<10}"
    )
    print("-" * 88)

    sorted_names = sorted(MODEL_NAMES, key=lambda n: aggregated[n]["mean_rmse"])
    for name in sorted_names:
        agg = aggregated[name]
        print(
            f"{MODEL_LABELS[name]:<28} | {agg['mean_rmse']:>12.2f} кН/м | "
            f"{agg['std_rmse']:>8.2f}   | {agg['mean_r2']:>12.4f}   | "
            f"{agg['std_r2']:>8.4f}"
        )
    print("=" * 88)


def save_report(
    aggregated: Dict[str, Dict],
    seeds: List[int],
    save_dir: str,
    noise_type: str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f"compare_models_{timestamp}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("СРАВНЕНИЕ СУРРОГАТНЫХ МОДЕЛЕЙ НА НАЧАЛЬНОЙ ВЫБОРКЕ\n")
        f.write(f"Тип данных: {noise_type}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Начальных точек: 35, тестовых: 20\n\n")

        f.write(
            f"{'Модель':<28} | {'RMSE (среднее)':<16} | {'RMSE (std)':<12} | "
            f"{'R² (среднее)':<14} | {'R² (std)':<10}\n"
        )
        f.write("-" * 88 + "\n")

        sorted_names = sorted(MODEL_NAMES, key=lambda n: aggregated[n]["mean_rmse"])
        for name in sorted_names:
            agg = aggregated[name]
            f.write(
                f"{MODEL_LABELS[name]:<28} | {agg['mean_rmse']:>12.2f} кН/м | "
                f"{agg['std_rmse']:>8.2f}   | {agg['mean_r2']:>12.4f}   | "
                f"{agg['std_r2']:>8.4f}\n"
            )

        f.write("\nДетализация по seed:\n")
        for name in MODEL_NAMES:
            f.write(f"\n{MODEL_LABELS[name]}:\n")
            for i, rmse in enumerate(aggregated[name]["rmse_values"]):
                r2 = aggregated[name]["r2_values"][i]
                f.write(f"  seed={seeds[i]}: RMSE={rmse:.2f}, R²={r2:.4f}\n")

    return report_path


def plot_boxplot(aggregated: Dict[str, Dict], save_dir: str) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    sorted_names = sorted(MODEL_NAMES, key=lambda n: aggregated[n]["mean_rmse"])
    labels = [MODEL_LABELS[n] for n in sorted_names]
    data = [aggregated[n]["rmse_values"] for n in sorted_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylabel("RMSE (кН/м)", fontsize=12)
    ax.set_xlabel("Модель", fontsize=12)
    ax.set_title(
        "Сравнение суррогатных моделей на начальной выборке (35 точек LHS)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")

    plot_path = os.path.join(save_dir, "comparison_models_boxplot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def main():
    args = parse_args()
    seeds = get_seeds(args.n_seeds, args.seed_start)

    print("=" * 80)
    print("СРАВНЕНИЕ СУРРОГАТНЫХ МОДЕЛЕЙ")
    print("=" * 80)
    print(f"Тип данных: {args.noise_type}")
    print(f"Начальных точек: {args.n_initial}, тестовых: {args.test_size}")
    print(f"Seeds: {seeds}")

    data_path = find_latest_data_file(
        data_dir=args.data_dir, noise_type=args.noise_type
    )
    print(f"Датасет: {os.path.basename(data_path)}")

    per_seed = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        per_seed.append(
            run_single_seed(
                seed=seed,
                data_path=data_path,
                n_initial=args.n_initial,
                test_size=args.test_size,
                rf_n_estimators=args.rf_n_estimators,
                rf_random_state=args.rf_random_state,
                verbose=True,
            )
        )

    aggregated = aggregate_results(per_seed)
    print_summary_table(aggregated, seeds)

    report_path = save_report(aggregated, seeds, args.save_dir, args.noise_type)
    plot_path = plot_boxplot(aggregated, args.save_dir)

    print(f"\n✅ Отчёт сохранён: {report_path}")
    print(f"✅ Boxplot сохранён: {plot_path}")
    print(f"📁 Результаты: {os.path.abspath(args.save_dir)}/")

    return aggregated, per_seed


if __name__ == "__main__":
    main()
