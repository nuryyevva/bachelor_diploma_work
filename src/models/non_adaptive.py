from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic


class NonAdaptiveModel:
    """
    Неадаптивная модель Гауссовского процесса, обученная на фиксированном наборе данных (LHS).

    Attributes:
        kernel (str): Тип ядра ('RBF', 'Matern', 'RationalQuadratic').
        model (GaussianProcessRegressor): Обученная модель.
    """

    def __init__(self, kernel="RBF", scaler_y=None):
        """
        Инициализация модели с указанным ядром.

        Args:
            kernel (str): Тип ядра. По умолчанию 'RBF'.
            scaler_y (StandardScaler): Скалер целевой переменной для обратного преобразования.
        """
        self.kernel = kernel
        self.model = None
        self.scaler_y = scaler_y

    def _create_kernel(self):
        """Создает ядро в зависимости от параметра."""
        if self.kernel == "RBF":
            return ConstantKernel() * RBF()
        elif self.kernel == "Matern":
            return ConstantKernel() * Matern(nu=1.5)
        elif self.kernel == "RationalQuadratic":
            return ConstantKernel() * RationalQuadratic()
        else:
            raise ValueError(f"Неизвестное ядро: {self.kernel}")

    def train(self, X, y):
        """
        Обучает модель на заданных данных.

        Args:
            X (np.array): Входные данные (признаки).
            y (np.array): Целевые значения.
        """
        self.model = GaussianProcessRegressor(
            kernel=self._create_kernel(),
            n_restarts_optimizer=10,
            random_state=42
        )
        self.model.fit(X, y)

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
            ).ravel()

        return y_pred
