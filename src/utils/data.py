import pandas as pd
from sklearn.model_selection import train_test_split


# def load_and_split_data(data_path, n_initial=30, test_size=10):
#     """Загружает данные и разделяет на начальный набор, кандидатов и тест."""
#     df = pd.read_csv(data_path)
#     X = df[["theta_base_deg", "total_thickness_m"]].values
#     y = df["critical_load_N_per_m"].values
#
#     # Разделяем на начальный набор (LHS) и остальные
#     X_initial, X_rest, y_initial, y_rest = train_test_split(
#         X, y, train_size=n_initial, random_state=42
#     )
#
#     # Разделяем остальные на кандидатов и тест
#     X_candidates, X_test, y_candidates, y_test = train_test_split(
#         X_rest, y_rest, test_size=test_size, random_state=42
#     )
#
#     return X_initial, y_initial, X_candidates, y_candidates, X_test, y_test

# src/utils/data.py

def load_and_split_data(data_path, n_initial=30, test_size=10):
    df = pd.read_csv(data_path)
    X = df[["theta_base_deg", "total_thickness_m"]].values
    y = df["critical_load_N_per_m"].values

    print(f"\n🔍 ВСЕГО ДАННЫХ: {len(df)} точек")

    # 1. Сначала отделяем ТЕСТ (10 точек)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"  После 1-го split: temp={len(X_temp)}, test={len(X_test)}")

    # 2. Из ОСТАВШИХСЯ берем INITIAL (30 точек)
    X_initial, X_candidates, y_initial, y_candidates = train_test_split(
        X_temp, y_temp, train_size=n_initial, random_state=42, shuffle=True
    )

    print(f"  После 2-го split: initial={len(X_initial)}, candidates={len(X_candidates)}")
    print(
        f"  Проверка: {len(X_initial)} + {len(X_candidates)} + {len(X_test)} = {len(X_initial) + len(X_candidates) + len(X_test)} (должно быть {len(df)})")

    return X_initial, y_initial, X_candidates, y_candidates, X_test, y_test
