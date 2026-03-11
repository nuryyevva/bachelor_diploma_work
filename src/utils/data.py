import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(data_path, n_initial=40, test_size=20, use_all_features=True):
    """
    Загрузка и разделение данных на обучающую, кандидатскую и тестовую выборки.

    Args:
        data_path (str): Путь к CSV-файлу с данными.
        n_initial (int): Количество точек в начальной обучающей выборке.
        test_size (int): Количество точек в тестовой выборке.
        use_all_features (bool): Если True — использовать все 3 признака, иначе — только 2.

    Returns:
        tuple: (X_initial, y_initial, X_candidates, y_candidates, X_test, y_test)
    """
    df = pd.read_csv(data_path)

    # Выбор признаков: 3D (с aspect_ratio) или 2D (без него)
    if use_all_features:
        feature_cols = ["theta_base_deg", "total_thickness_m", "aspect_ratio"]
        print(f"  ✅ Используем 3 признака: {feature_cols}")
    else:
        feature_cols = ["theta_base_deg", "total_thickness_m"]
        print(f"  ✅ Используем 2 признака: {feature_cols}")

    X = df[feature_cols].values
    y = df["critical_load_N_per_m"].values

    print(f"\n🔍 ВСЕГО ДАННЫХ: {len(df)} точек")
    print(f"  Признаки: {X.shape[1]}D")
    print(f"  Диапазон y: {y.min() / 1e3:.2f} - {y.max() / 1e3:.2f} кН/м")

    # 1. Сначала отделяем ТЕСТ
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"  После 1-го split: temp={len(X_temp)}, test={len(X_test)}")

    # 2. Из ОСТАВШИХСЯ берем INITIAL
    X_initial, X_candidates, y_initial, y_candidates = train_test_split(
        X_temp, y_temp, train_size=n_initial, random_state=42, shuffle=True
    )

    print(f"  После 2-го split: initial={len(X_initial)}, candidates={len(X_candidates)}")
    print(
        f"  Проверка: {len(X_initial)} + {len(X_candidates)} + {len(X_test)} = "
        f"{len(X_initial) + len(X_candidates) + len(X_test)} (должно быть {len(df)})"
    )

    # 3. Дополнительная проверка для 3D задачи
    if use_all_features and X.shape[1] == 3:
        print(f"\n  ⚠️ 3D задача: рекомендуется n_initial >= 30-40 точек")
        if n_initial < 30:
            print(f"  ⚠️ Внимание: n_initial={n_initial} может быть недостаточно для 3D!")

    return X_initial, y_initial, X_candidates, y_candidates, X_test, y_test

def find_latest_data_file(data_dir="../../data", noise_type="with_noise"):
    """
    Находит последний CSV-файл в указанной папке (no_noise или with_noise).

    Args:
        data_dir (str): Корневая папка с данными.
        noise_type (str): Тип данных: "no_noise" или "with_noise".

    Returns:
        str: Путь к последнему CSV-файлу.
    """
    import os

    folder_path = os.path.join(data_dir, noise_type)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка {folder_path} не существует!")

    csv_files = [
        f for f in os.listdir(folder_path)
        if f.startswith("composite_plate_buckling_data") and f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"Нет CSV-файлов в {folder_path}")

    # Сортируем по времени создания
    latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest_csv)
