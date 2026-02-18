"""
Адаптивная модель Гауссовского процесса с итеративным выбором точек данных.
Поддерживает различные стратегии (максимизация дисперсии, Expected Improvement и др.).
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic
)
from src.utils.metrics import calculate_rmse, calculate_r2, calculate_mae


class AdaptiveModel:
    """
    Адаптивная модель Гауссовского процесса.

    На каждой итерации выбирает новую точку из пула кандидатов
    с использованием заданной стратегии (acquisition function),
    добавляет её в обучающий набор и переобучает модель.

    Attributes:
        strategy: Стратегия выбора точек (объект с методом select_next_point).
        kernel (str): Тип ядра ('RBF', 'Matern', 'RationalQuadratic').
        max_iterations (int): Максимальное количество итераций.
        target_rmse (float): Целевой RMSE для ранней остановки.
        model: Текущая обученная модель GP.
        rmse_history (list): История значений RMSE на тестовых данных.
        r2_history (list): История значений R² на тестовых данных.
        mae_history (list): История значений MAE на тестовых данных.
        selected_points_history (list): История выбранных точек (индексы).
        n_points_history (list): Количество точек в обучении на каждой итерации.
    """

    def __init__(
            self,
            strategy,
            kernel="RBF",
            max_iterations=20,
            target_rmse=None,
            scaler_y=None
    ):
        """
        Инициализация адаптивной модели.

        Args:
            strategy: Стратегия выбора точек (должна иметь метод select_next_point).
            kernel (str): Тип ядра. По умолчанию 'RBF'.
            max_iterations (int): Максимальное количество итераций. По умолчанию 20.
            target_rmse (float): Целевой RMSE для ранней остановки. Если None — не останавливается.
            scaler_y (StandardScaler): Скалер целевой переменной для обратного преобразования.
        """
        self.strategy = strategy
        self.kernel = kernel
        self.max_iterations = max_iterations
        self.target_rmse = target_rmse
        self.scaler_y = scaler_y

        self.model = None
        self.rmse_history = []
        self.r2_history = []
        self.mae_history = []
        self.selected_points_history = []
        self.n_points_history = []

    def _create_kernel(self):
        """
        Создает ядро в зависимости от параметра.

        Returns:
            sklearn.gaussian_process.kernels.Kernel: Ядро для GP.
        """
        if self.kernel == "RBF":
            return ConstantKernel() * RBF()
            # Явно задаем границы для стабильности
            # return ConstantKernel(1.0, (1e-6, 1e6)) * RBF(
            #     length_scale=[1.0, 1.0],
            #     length_scale_bounds=(1e-4, 1e4)
            # )
        elif self.kernel == "Matern":
            return ConstantKernel() * Matern(nu=1.5)
        elif self.kernel == "RationalQuadratic":
            return ConstantKernel() * RationalQuadratic()
        else:
            raise ValueError(f"Неизвестное ядро: {self.kernel}")

    def train(self, X_initial, y_initial, X_candidates, y_candidates, X_test, y_test):
        """
        Запускает адаптивный цикл обучения.

        На каждой итерации:
        1. Обучает модель на текущих данных
        2. Оценивает качество на тестовых данных
        3. Выбирает следующую точку из кандидатов
        4. Добавляет точку в обучающий набор
        5. Удаляет точку из кандидатов

        Args:
            X_initial (np.array): Начальный обучающий набор (признаки).
            y_initial (np.array): Начальный обучающий набор (целевые значения).
            X_candidates (np.array): Пул кандидатов для выбора.
            y_candidates (np.array): Целевые значения кандидатов.
            X_test (np.array): Тестовые данные (признаки).
            y_test (np.array): Тестовые данные (целевые значения).

        Returns:
            dict: Словарь с историей метрик и выбранных точек.
        """
        # Копируем данные, чтобы не изменять оригиналы
        X_train = X_initial.copy()
        y_train = y_initial.copy()
        X_pool = X_candidates.copy()
        y_pool = y_candidates.copy()

        print(f"\n🔍 ПРОВЕРКА МАСШТАБА ПЕРЕД FIT:")
        print(f"  X_train mean: {X_train.mean():.4f} (должно быть ~0)")
        print(f"  y_train mean: {y_train.mean():.4f} (должно быть ~0)")
        print(f"  y_test (для метрик) mean: {y_test.mean():.2f} (должно быть ~4 500 000)")

        print(f"Начало адаптивного обучения...")
        print(f"Начальный набор: {len(X_train)} точек")
        print(f"Пул кандидатов: {len(X_pool)} точек")
        print(f"Тестовый набор: {len(X_test)} точек")
        print(f"Стратегия: {self.strategy.__class__.__name__}")
        print(f"Ядро: {self.kernel}")
        print("-" * 50)

        for iteration in range(self.max_iterations):
            # 1. Обучаем модель на текущих данных
            self.model = GaussianProcessRegressor(
                kernel=self._create_kernel(),
                n_restarts_optimizer=10,
                alpha=1e-10,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            if iteration == 0:  # Только на первой итерации
                test_pred_raw = self.model.predict(X_test)  # Нормализованное
                test_pred_final = self.predict(X_test)  # После inverse_transform

                print(f"\n🔍 ОТЛАДКА ПРЕДСКАЗАНИЙ (итерация 1):")
                print(f"  test_pred_raw mean: {test_pred_raw.mean():.4f} (должно быть ~0)")
                print(f"  test_pred_final mean: {test_pred_final.mean():.2f} (должно быть ~{y_test.mean():.2f})")
                print(f"  y_test mean: {y_test.mean():.2f}")
                print(f"  scaler_y.scale_: {self.scaler_y.scale_}")
                print(f"  scaler_y.mean_: {self.scaler_y.mean_}")

            # 2. Оцениваем качество на тестовых данных
            # current_rmse = calculate_rmse(self.model, X_test, y_test)
            # current_r2 = calculate_r2(self.model, X_test, y_test)
            # current_mae = calculate_mae(self.model, X_test, y_test)
            y_pred = self.predict(X_test)  # Использует self.predict() со скалером!
            current_rmse = calculate_rmse(y_pred, y_test)
            current_r2 = calculate_r2(y_pred, y_test)
            current_mae = calculate_mae(y_pred, y_test)

            # Сохраняем метрики
            self.rmse_history.append(current_rmse)
            self.r2_history.append(current_r2)
            self.mae_history.append(current_mae)
            self.n_points_history.append(len(X_train))

            print(f"Итерация {iteration + 1}/{self.max_iterations} | "
                  f"Точек: {len(X_train)} | "
                  f"RMSE: {current_rmse:.2f} кН/м | "
                  f"R²: {current_r2:.4f} | "
                  f"MAE: {current_mae:.2f} кН/м")

            # 3. Проверяем условие ранней остановки
            if self.target_rmse is not None and current_rmse <= self.target_rmse:
                print(f"\n✅ Достигнут целевой RMSE ({self.target_rmse:.2f}) на итерации {iteration + 1}")
                break

            # 4. Если кандидатов больше нет — выходим
            if len(X_pool) == 0:
                print("\n⚠️ Пул кандидатов исчерпан")
                break

            # 5. Выбираем следующую точку с использованием стратегии
            next_point_idx = self.strategy.select_next_point(self.model, X_pool)

            # 6. Добавляем выбранную точку в обучающий набор
            X_train = np.vstack([X_train, X_pool[next_point_idx:next_point_idx + 1]])
            y_train = np.append(y_train, y_pool[next_point_idx])

            # 7. Сохраняем информацию о выбранной точке
            self.selected_points_history.append({
                'iteration': iteration + 1,
                'index_in_pool': next_point_idx,
                'features': X_pool[next_point_idx],
                'target': y_pool[next_point_idx]
            })

            # 8. Удаляем выбранную точку из пула кандидатов
            X_pool = np.delete(X_pool, next_point_idx, axis=0)
            y_pool = np.delete(y_pool, next_point_idx)

        print("-" * 50)
        print(f"Обучение завершено. Финальный RMSE: {self.rmse_history[-1]:.2f} кН/м")

        return {
            'rmse_history': self.rmse_history,
            'r2_history': self.r2_history,
            'mae_history': self.mae_history,
            'n_points_history': self.n_points_history,
            'selected_points_history': self.selected_points_history,
            'final_rmse': self.rmse_history[-1],
            'final_r2': self.r2_history[-1],
            'final_mae': self.mae_history[-1],
            'n_iterations': len(self.rmse_history),
            'n_final_points': len(X_train)
        }

    def predict(self, X):
        """
        Предсказывает значения для новых данных.

        Args:
            X (np.array): Входные данные.

        Returns:
            np.array: Предсказанные значения.
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите метод train() сначала.")
        y_pred = self.model.predict(X)

        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()  # Возвращаем в оригинальный масштаб

        return y_pred

    def predict_with_uncertainty(self, X):
        """
        Предсказывает значения и неопределенность (стандартное отклонение).

        Args:
            X (np.array): Входные данные.

        Returns:
            tuple: (предсказанные значения, стандартное отклонение)
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите метод train() сначала.")
        y_pred, std = self.model.predict(X, return_std=True)

        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()
            std = std * self.scaler_y.scale_  # Масштабируем std

        return y_pred, std

    def get_training_history(self):
        """
        Возвращает полную историю обучения.

        Returns:
            dict: Словарь с историей метрик и выбранных точек.
        """
        return {
            'iterations': list(range(1, len(self.rmse_history) + 1)),
            'n_points': self.n_points_history,
            'rmse': self.rmse_history,
            'r2': self.r2_history,
            'mae': self.mae_history,
            'selected_points': self.selected_points_history
        }
