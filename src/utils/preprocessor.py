from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Класс для предобработки данных (нормализация).
    Fit выполняется только на начальных данных, чтобы избежать data leakage.
    """

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Обучает скалеры на тренировочных данных.

        Args:
            X_train: Признаки обучающей выборки.
            y_train: Целевая переменная обучающей выборки.
        """
        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train.reshape(-1, 1))
        self.is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Преобразует данные используя обученные скалеры.

        Returns:
            Если y передан: (X_transformed, y_transformed)
            Если y не передан: X_transformed
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted! Вызовите fit() сначала.")

        X_transformed = self.scaler_X.transform(X)

        if y is not None:
            y_transformed = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
            return X_transformed, y_transformed
        return X_transformed

    def inverse_transform_y(self, y):
        """Возвращает целевую переменную в исходный масштаб."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        return self.scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()

    def get_original_y_test(self, y_test_normalized):
        """
        Удобный метод для получения оригинального y_test из нормализованного.
        Важно для расчета метрик в реальных единицах.
        """
        return self.inverse_transform_y(y_test_normalized)
