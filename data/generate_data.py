import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import os

# ======================
# 1. ПАРАМЕТРЫ МОДЕЛИ
# ======================
# Свойства материала (углепластик T300/934)
E1 = 150e9  # ГПа -> Па (модуль вдоль волокон)
E2 = 9e9  # ГПа -> Па (модуль поперек волокон)
G12 = 7e9  # ГПа -> Па (модуль сдвига)
nu12 = 0.32  # Коэффициент Пуассона
density = 1600  # кг/м³ (плотность)

# Геометрия пластины
a = 1.0  # Длина пластины вдоль оси x (м)
b = 0.5  # Ширина пластины вдоль оси y (м)
m = 1  # Число полуволн вдоль длины (минимальная критическая нагрузка)

# Граничные условия: просто опертая пластина (SSSS) -> коэффициент K=4 в формуле Эйлера
K = 4  # Для SSSS пластины при одноосном сжатии

# Параметры генерации данных
n_points = 80
theta_range = (0, 45)  # Диапазон углов укладки (градусы)
thickness_range = (2e-3, 20e-3)  # Диапазон общей толщины (м)


# ======================
# 2. ФУНКЦИИ РАСЧЕТА
# ======================
def calculate_Q_matrix(E1, E2, G12, nu12):
    """Вычисление матрицы жесткости для ортотропного слоя в локальной системе координат"""
    nu21 = nu12 * E2 / E1
    Q11 = E1 / (1 - nu12 * nu21)
    Q12 = nu12 * E2 / (1 - nu12 * nu21)
    Q22 = E2 / (1 - nu12 * nu21)
    Q66 = G12
    return np.array([[Q11, Q12, 0],
                     [Q12, Q22, 0],
                     [0, 0, Q66]])


def transform_Q(Q, theta_deg):
    """Трансформация матрицы жесткости при повороте слоя на угол theta"""
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    T = np.array([[c ** 2, s ** 2, 2 * c * s],
                  [s ** 2, c ** 2, -2 * c * s],
                  [-c * s, c * s, c ** 2 - s ** 2]])
    T_inv = np.array([[c ** 2, s ** 2, -2 * c * s],
                      [s ** 2, c ** 2, 2 * c * s],
                      [c * s, -c * s, c ** 2 - s ** 2]])
    return T_inv.T @ Q @ T_inv


def calculate_D_matrix(angles, thicknesses, Q0):
    """Вычисление матрицы изгибных жесткостей D для симметричного ламината"""
    total_thickness = sum(thicknesses)
    z = []  # Координаты границ слоев по толщине
    current_z = -total_thickness / 2
    z.append(current_z)
    for t in thicknesses:
        current_z += t
        z.append(current_z)

    D = np.zeros((3, 3))
    for i, theta in enumerate(angles):
        Q = transform_Q(Q0, theta)
        zk = z[i]
        zk1 = z[i + 1]
        D += Q * (zk1 ** 3 - zk ** 3) / 3
    return D


def critical_buckling_load(D11, D22, D12, D66, a, b, m=1):
    """Аналитическая формула критической нагрузки для ортотропной пластины
    Источник: Jones, R.M. Mechanics of Composite Materials (1999)"""
    term1 = D11 * (m * b / a) ** 2
    term2 = 2 * (D12 + 2 * D66)
    term3 = D22 * (a / (m * b)) ** 2
    N_cr = (np.pi ** 2 / b ** 2) * (term1 + term2 + term3)
    return N_cr  # Н/м (нагрузка на единицу длины)


# ======================
# 3. ГЕНЕРАЦИЯ ДАННЫХ
# ======================
np.random.seed(42)  # Для воспроизводимости

# Базовая матрица Q для материала
Q0 = calculate_Q_matrix(E1, E2, G12, nu12)

data = []
for i in range(n_points):
    # Генерация случайных параметров
    theta_base = np.random.uniform(*theta_range)  # Базовый угол укладки
    total_thickness = np.random.uniform(*thickness_range)  # Общая толщина

    # Схема укладки: [θ, -θ, 90, 0]s (симметричная, 8 слоев)
    angles = [theta_base, -theta_base, 90, 0, 0, 90, -theta_base, theta_base]

    # Равномерное распределение толщины по слоям
    layer_thickness = total_thickness / len(angles)
    thicknesses = [layer_thickness] * len(angles)

    # Расчет матрицы D
    D = calculate_D_matrix(angles, thicknesses, Q0)
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]

    # Расчет критической нагрузки
    N_cr = critical_buckling_load(D11, D22, D12, D66, a, b, m)

    # Сохранение данных
    data.append({
        "theta_base_deg": theta_base,
        "total_thickness_m": total_thickness,
        "D11_Nm": D11,
        "D12_Nm": D12,
        "D22_Nm": D22,
        "D66_Nm": D66,
        "critical_load_N_per_m": N_cr
    })

# ======================
# 4. СОХРАНЕНИЕ И ВИЗУАЛИЗАЦИЯ
# ======================
# Создаем DataFrame
df = pd.DataFrame(data)

# Сохранение в CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"composite_plate_buckling_data_{timestamp}.csv"
df.to_csv(csv_filename, index=False)
print(f"✅ Данные сохранены в: {csv_filename}")

# Статистика
stats = (
    f"Количество точек: {len(df)}\n"
    f"Диапазон толщины: {df['total_thickness_m'].min() * 1e3:.2f} - {df['total_thickness_m'].max() * 1e3:.2f} мм\n"
    f"Диапазон углов: {df['theta_base_deg'].min():.2f} - {df['theta_base_deg'].max():.2f} град\n"
    f"Диапазон критической нагрузки: {df['critical_load_N_per_m'].min() / 1e3:.2f} - {df['critical_load_N_per_m'].max() / 1e3:.2f} кН/м\n"
    f"Средняя критическая нагрузка: {df['critical_load_N_per_m'].mean() / 1e3:.2f} кН/м"
)
print("\n📊 Статистика данных:")
print(stats)

# Сохранение статистики в файл
stats_filename = f"data_statistics_{timestamp}.txt"
with open(stats_filename, 'w') as f:
    f.write(stats)
print(f"✅ Статистика сохранена в: {stats_filename}")

# Визуализация
plt.figure(figsize=(15, 10))

# 1. Нагрузка vs Толщина
plt.subplot(2, 2, 1)
plt.scatter(df['total_thickness_m'] * 1e3, df['critical_load_N_per_m'] / 1e3, alpha=0.7, c='blue')
plt.xlabel('Толщина пластины (мм)')
plt.ylabel('Критическая нагрузка (кН/м)')
plt.title('Зависимость критической нагрузки от толщины')
plt.grid(True)

# 2. Нагрузка vs Угол укладки
plt.subplot(2, 2, 2)
plt.scatter(df['theta_base_deg'], df['critical_load_N_per_m'] / 1e3, alpha=0.7, c='green')
plt.xlabel('Базовый угол укладки (градусы)')
plt.ylabel('Критическая нагрузка (кН/м)')
plt.title('Зависимость критической нагрузки от угла укладки')
plt.grid(True)

# 3. Гистограмма распределения нагрузки
plt.subplot(2, 2, 3)
plt.hist(df['critical_load_N_per_m'] / 1e3, bins=15, color='purple', alpha=0.7)
plt.xlabel('Критическая нагрузка (кН/м)')
plt.ylabel('Частота')
plt.title('Распределение критической нагрузки')
plt.grid(True)

# 4. 3D-график (толщина, угол, нагрузка)
ax = plt.subplot(2, 2, 4, projection='3d')
sc = ax.scatter(df['total_thickness_m'] * 1e3,
                df['theta_base_deg'],
                df['critical_load_N_per_m'] / 1e3,
                c=df['critical_load_N_per_m'] / 1e3,
                cmap=cm.viridis,
                s=50,
                alpha=0.8)
ax.set_xlabel('Толщина (мм)')
ax.set_ylabel('Угол (град)')
ax.set_zlabel('Нагрузка (кН/м)')
ax.set_title('3D-зависимость')
plt.colorbar(sc, ax=ax, label='Критическая нагрузка (кН/м)')

plt.tight_layout()
plot_filename = f"buckling_data_analysis_{timestamp}.png"
plt.savefig(plot_filename, dpi=300)
print(f"✅ Графики сохранены в: {plot_filename}")
plt.show()

print("\n🎉 Генерация данных завершена успешно!")
