import matplotlib.pyplot as plt
import os


def plot_convergence(
        adaptive_rmse_history,
        non_adaptive_rmse,
        save_path="results/convergence_plot.png",
        title="Сравнение сходимости моделей"
):
    """
    Строит график сходимости RMSE для адаптивной и неадаптивной моделей.

    Args:
        adaptive_rmse_history (list): История RMSE адаптивной модели.
        non_adaptive_rmse (float): RMSE неадаптивной модели.
        save_path (str): Путь для сохранения графика.
        title (str): Заголовок графика.
    """
    plt.figure(figsize=(10, 6))

    # График адаптивной модели
    plt.plot(
        range(1, len(adaptive_rmse_history) + 1),
        adaptive_rmse_history,
        'b-o',
        label=f"Адаптивная модель (мин. RMSE = {min(adaptive_rmse_history):.2f})"
    )

    # Горизонтальная линия для неадаптивной модели
    plt.axhline(
        non_adaptive_rmse,
        color='r',
        linestyle='--',
        label=f"Неадаптивная модель (RMSE = {non_adaptive_rmse:.2f})"
    )

    plt.xlabel("Количество итераций")
    plt.ylabel("RMSE (кН/м)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Создаем папку results, если её нет
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_data_distribution(data_path, save_path="results/data_distribution.png"):
    """
    Строит распределение критической нагрузки из данных.

    Args:
        data_path (str): Путь к CSV-файлу с данными.
        save_path (str): Путь для сохранения графика.
    """
    import pandas as pd
    df = pd.read_csv(data_path)

    plt.figure(figsize=(8, 5))
    plt.hist(
        df["critical_load_N_per_m"] / 1e3,
        bins=15,
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )
    plt.xlabel("Критическая нагрузка (кН/м)")
    plt.ylabel("Частота")
    plt.title("Распределение критической нагрузки")
    plt.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
