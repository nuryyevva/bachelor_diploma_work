"""
Адаптивная модель Гауссовского процесса с итеративным выбором точек данных.
На каждой итерации генерирует кандидатов в пределах bounds и выбирает лучшую
точку по acquisition function; y вычисляется через физическую модель.
"""

import itertools
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
)

from src.utils.discrete_params import get_all_discrete_combinations
from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae


class AdaptiveModel:
    """Адаптивная GPR-модель с генерацией кандидатов в пространстве параметров."""

    def __init__(
        self,
        strategy,
        kernel="RBF",
        max_iterations=20,
        target_rmse=None,
        scaler_y=None,
        convergence_patience=3,
        n_candidates=1000,
        include_boundaries=False,
        discrete=False,
    ):
        self.strategy = strategy
        self.kernel = kernel
        self.max_iterations = max_iterations
        self.target_rmse = target_rmse
        self.scaler_y = scaler_y
        self.convergence_patience = convergence_patience
        self.n_candidates = n_candidates
        self.include_boundaries = include_boundaries
        self.discrete = discrete

        self.model = None
        self.rmse_history = []
        self.r2_history = []
        self.mae_history = []
        self.selected_points_history = []
        self.n_points_history = []
        self.time_history = []

    def _create_kernel(self):
        if self.kernel == "RBF":
            return ConstantKernel() * RBF()
        if self.kernel == "Matern":
            return ConstantKernel() * Matern(nu=1.5)
        if self.kernel == "RationalQuadratic":
            return ConstantKernel() * RationalQuadratic()
        raise ValueError(f"Неизвестное ядро: {self.kernel}")

    @staticmethod
    def _get_corner_points(bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Все 8 угловых точек гиперкуба (min/max для каждого параметра)."""
        corner_values = [[b[0], b[1]] for b in bounds]
        return np.array(list(itertools.product(*corner_values)), dtype=float)

    @staticmethod
    def _get_boundary_face_points(
        bounds: List[Tuple[float, float]],
        n_faces: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Точки на гранях: один параметр на границе, остальные случайно."""
        d = len(bounds)
        points = []
        for _ in range(n_faces):
            dim = int(rng.integers(0, d))
            boundary_val = bounds[dim][int(rng.integers(0, 2))]
            point = []
            for j in range(d):
                if j == dim:
                    point.append(boundary_val)
                else:
                    point.append(rng.uniform(bounds[j][0], bounds[j][1]))
            points.append(point)
        return np.array(points, dtype=float)

    @staticmethod
    def _remove_duplicates(
        points: np.ndarray,
        bounds: List[Tuple[float, float]],
        tol: float = 1e-9,
    ) -> np.ndarray:
        """Удаляет дубликаты с учётом масштаба каждого параметра."""
        if len(points) == 0:
            return points

        scales = np.array([max(b[1] - b[0], tol) for b in bounds])
        normalized = points / scales
        rounded = np.round(normalized / tol)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        return points[np.sort(unique_idx)]

    @staticmethod
    def _sample_lhs(
        bounds: List[Tuple[float, float]],
        n_candidates: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Генерирует кандидатов в исходном масштабе признаков (LHS)."""
        d = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        sampler = qmc.LatinHypercube(d=d, seed=int(rng.integers(0, 2**31)))
        unit = sampler.random(n=n_candidates)
        return qmc.scale(unit, lows, highs)

    @staticmethod
    def _sample_candidates_discrete(
        n_candidates: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Случайная выборка из дискретной сетки параметров."""
        all_combos = get_all_discrete_combinations()
        n_total = len(all_combos)
        n_sample = min(n_candidates, n_total)
        indices = rng.choice(n_total, size=n_sample, replace=False)
        return all_combos[indices]

    @staticmethod
    def _sample_candidates(
        bounds: List[Tuple[float, float]],
        n_candidates: int,
        rng: np.random.Generator,
        include_boundaries: bool = False,
        discrete: bool = False,
    ) -> np.ndarray:
        """
        Генерирует пул кандидатов.

        При discrete=True — случайная выборка из дискретной сетки.
        При include_boundaries=True — LHS + 8 углов + ~10 точек на гранях, без дубликатов.
        """
        if discrete:
            return AdaptiveModel._sample_candidates_discrete(n_candidates, rng)

        lhs = AdaptiveModel._sample_lhs(bounds, n_candidates, rng)

        if not include_boundaries:
            return lhs

        corners = AdaptiveModel._get_corner_points(bounds)
        faces = AdaptiveModel._get_boundary_face_points(bounds, 10, rng)
        combined = np.vstack([lhs, corners, faces])
        return AdaptiveModel._remove_duplicates(combined, bounds)

    def train(
        self,
        X_initial,
        y_initial,
        X_test,
        y_test,
        bounds: List[Tuple[float, float]],
        evaluate_fn: Callable[[np.ndarray], float],
        scaler_X=None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Адаптивный цикл обучения.

        Args:
            X_initial, y_initial: Начальная выборка (нормализованные).
            X_test: Тестовые признаки (нормализованные).
            y_test: Тестовые метки в кН/м (исходный масштаб).
            bounds: [(min, max), ...] для каждого признака в исходных единицах.
            evaluate_fn: X_raw (3,) -> y в кН/м через физическую модель.
            scaler_X: StandardScaler признаков для нормализации новых точек.
            seed: Seed генерации кандидатов на итерациях.
            verbose: Подробный вывод.

        Returns:
            dict с историей метрик.
        """
        if scaler_X is None:
            raise ValueError("scaler_X обязателен для нормализации новых точек")
        if self.scaler_y is None:
            raise ValueError("scaler_y обязателен для нормализации y")

        X_train = X_initial.copy()
        y_train = y_initial.copy()
        rng = np.random.default_rng(seed)

        if verbose:
            print(f"\nНачало адаптивного обучения (динамические кандидаты)...")
            print(f"  Начальный набор: {len(X_train)} точек")
            print(f"  Кандидатов/итерация: {self.n_candidates}")
            print(f"  Тестовый набор: {len(X_test)} точек")
            print(f"  Стратегия: {self.strategy.__class__.__name__}")
            print(f"  Ядро: {self.kernel}")
            print(f"  Граничные точки: {self.include_boundaries}")
            print(f"  Дискретный режим: {self.discrete}")
            if self.target_rmse is not None:
                print(f"  Target RMSE: {self.target_rmse:.2f} кН/м")
            print("-" * 50)

        consecutive_below_threshold = 0
        start_time = time.time()
        self.time_history = []
        self.rmse_history = []
        self.r2_history = []
        self.mae_history = []
        self.n_points_history = []
        self.selected_points_history = []

        for iteration in range(self.max_iterations):
            iter_start = time.time()

            self.model = GaussianProcessRegressor(
                kernel=self._create_kernel(),
                n_restarts_optimizer=10,
                alpha=1e-10,
                random_state=42,
            )
            self.model.fit(X_train, y_train)

            y_pred = self.predict(X_test)
            current_rmse = calculate_rmse(y_pred, y_test)
            current_r2 = calculate_r2(y_pred, y_test)
            current_mae = calculate_mae(y_pred, y_test)

            self.rmse_history.append(current_rmse)
            self.r2_history.append(current_r2)
            self.mae_history.append(current_mae)
            self.n_points_history.append(len(X_train))

            iter_time = time.time() - iter_start
            self.time_history.append(iter_time)

            if verbose:
                print(
                    f"Итерация {iteration + 1}/{self.max_iterations} | "
                    f"Точек: {len(X_train)} | "
                    f"RMSE: {current_rmse:.2f} кН/м | "
                    f"R²: {current_r2:.4f} | "
                    f"MAE: {current_mae:.2f} кН/м | "
                    f"Время: {iter_time:.2f} с"
                )

            if self.target_rmse is not None and current_rmse <= self.target_rmse:
                consecutive_below_threshold += 1
                if verbose:
                    print(
                        f"  ⚡ Ниже порога: {consecutive_below_threshold}/"
                        f"{self.convergence_patience}"
                    )
                if consecutive_below_threshold >= self.convergence_patience:
                    if verbose:
                        print(f"\n✅ Достигнут целевой RMSE ({self.target_rmse:.2f} кН/м)")
                    break
            else:
                consecutive_below_threshold = 0

            if iteration + 1 >= self.max_iterations:
                break

            X_cand_raw = self._sample_candidates(
                bounds,
                self.n_candidates,
                rng,
                include_boundaries=self.include_boundaries,
                discrete=self.discrete,
            )
            X_cand = scaler_X.transform(X_cand_raw)

            next_idx = self.strategy.select_next_point(self.model, X_cand)
            x_new_raw = X_cand_raw[next_idx]
            y_new_knm = float(evaluate_fn(x_new_raw))
            y_new = self.scaler_y.transform(np.array([[y_new_knm]])).ravel()[0]
            x_new = X_cand[next_idx : next_idx + 1]

            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, y_new)

            self.selected_points_history.append({
                "iteration": iteration + 1,
                "features_raw": x_new_raw,
                "features": x_new[0],
                "target_knm": y_new_knm,
                "target": y_new,
            })

        total_time = time.time() - start_time
        n_iterations = len(self.rmse_history)
        time_per_iteration = total_time / n_iterations if n_iterations > 0 else 0.0

        if verbose:
            print("-" * 50)
            print(f"Обучение завершено. Финальный RMSE: {self.rmse_history[-1]:.2f} кН/м")
            print(f"Всего итераций: {n_iterations}")
            print(
                f"Общее время: {total_time:.2f} с | "
                f"Среднее на итерацию: {time_per_iteration:.2f} с"
            )

        return {
            "rmse_history": self.rmse_history,
            "r2_history": self.r2_history,
            "mae_history": self.mae_history,
            "n_points_history": self.n_points_history,
            "selected_points_history": self.selected_points_history,
            "final_rmse": self.rmse_history[-1],
            "final_r2": self.r2_history[-1],
            "final_mae": self.mae_history[-1],
            "n_iterations": n_iterations,
            "n_final_points": len(X_train),
            "consecutive_below_threshold": consecutive_below_threshold,
            "total_time": total_time,
            "time_per_iteration": time_per_iteration,
            "time_history": self.time_history,
        }

    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите train() сначала.")
        y_pred = self.model.predict(X)
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()
        return y_pred

    def predict_with_uncertainty(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите train() сначала.")
        y_pred, std = self.model.predict(X, return_std=True)
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()
            std = std * self.scaler_y.scale_
        return y_pred, std

    def get_training_history(self):
        return {
            "iterations": list(range(1, len(self.rmse_history) + 1)),
            "n_points": self.n_points_history,
            "rmse": self.rmse_history,
            "r2": self.r2_history,
            "mae": self.mae_history,
            "selected_points": self.selected_points_history,
        }
