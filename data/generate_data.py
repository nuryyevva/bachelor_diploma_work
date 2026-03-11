import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import os

# ======================
# 1. ПАРАМЕТРЫ МОДЕЛИ
# ======================
# seed рандомайзера
seed = 132

# Свойства материала (углепластик T300/934)
E1 = 150e9  # Па (модуль вдоль волокон)
E2 = 9e9  # Па (модуль поперек волокон)
G12 = 7e9  # Па (модуль сдвига)
nu12 = 0.32  # Коэффициент Пуассона
density = 1600  # кг/м³ (плотность)

# Базовая геометрия пластины
b = 0.5  # Ширина пластины вдоль оси y (м) - фиксирована
m = 1  # Число полуволн вдоль длины (минимальная критическая нагрузка)

# Граничные условия: просто опертая пластина (SSSS)
K = 4  # Для SSSS пластины при одноосном сжатии

# Параметры генерации данных
n_points = 150  # Увеличили количество точек для 3D задачи
theta_range = (0, 45)  # Диапазон углов укладки (градусы)
thickness_range = (2e-3, 20e-3)  # Диапазон общей толщины (м)
aspect_ratio_range = (1.0, 4.0)  # Соотношение сторон a/b

# Параметры шума
noise_std = 0.05  # 5% шум от стандартного отклонения целевой переменной
add_noise = True  # Флаг добавления шума


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


def generate_data(n_points, add_noise=False, noise_std=0.05, seed=42):
    """
    Генерация данных с опциональным добавлением шума.

    Args:
        n_points: Количество точек данных
        add_noise: Добавлять ли шум к целевой переменной
        noise_std: Стандартное отклонение шума (доля от std целевой переменной)
        seed: Seed для воспроизводимости

    Returns:
        df: DataFrame с данными
    """
    np.random.seed(seed)

    # Базовая матрица Q для материала
    Q0 = calculate_Q_matrix(E1, E2, G12, nu12)

    data = []
    for i in range(n_points):
        # Генерация случайных параметров
        theta_base = np.random.uniform(*theta_range)  # Базовый угол укладки
        total_thickness = np.random.uniform(*thickness_range)  # Общая толщина
        aspect_ratio = np.random.uniform(*aspect_ratio_range)  # Соотношение сторон a/b

        # Вычисляем длину пластины из соотношения сторон
        a = aspect_ratio * b

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

        # Добавление шума (если включено)
        if add_noise:
            noise = np.random.normal(0, noise_std * N_cr)
            N_cr_noisy = N_cr + noise
            # Гарантируем положительность нагрузки
            N_cr_noisy = max(N_cr_noisy, 0.1 * N_cr)
        else:
            N_cr_noisy = N_cr

        # Сохранение данных
        data.append({
            "theta_base_deg": theta_base,
            "total_thickness_m": total_thickness,
            "aspect_ratio": aspect_ratio,
            "a_m": a,  # Сохраняем вычисленную длину
            "D11_Nm": D11,
            "D12_Nm": D12,
            "D22_Nm": D22,
            "D66_Nm": D66,
            "critical_load_N_per_m": N_cr_noisy,
            "critical_load_clean_N_per_m": N_cr  # Чистое значение для сравнения
        })

    return pd.DataFrame(data)


def save_and_visualize(df, seed, output_dir, suffix=""):
    """
    Сохранение данных и создание визуализаций.

    Args:
        df: DataFrame с данными
        seed: Seed рандомайзкра
        output_dir: Папка для сохранения
        suffix: Суффикс для имен файлов
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ======================
    # СОХРАНЕНИЕ В CSV
    # ======================
    csv_filename = os.path.join(output_dir, f"composite_plate_buckling_data_{suffix}{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"✅ Данные сохранены в: {csv_filename}")

    # ======================
    # СТАТИСТИКА
    # ======================
    stats = (
        f"Random seed: {seed}\n"
        f"Количество точек: {len(df)}\n"
        f"Диапазон толщины: {df['total_thickness_m'].min() * 1e3:.2f} - {df['total_thickness_m'].max() * 1e3:.2f} мм\n"
        f"Диапазон углов: {df['theta_base_deg'].min():.2f} - {df['theta_base_deg'].max():.2f} град\n"
        f"Диапазон соотношения сторон (a/b): {df['aspect_ratio'].min():.2f} - {df['aspect_ratio'].max():.2f}\n"
        f"Диапазон критической нагрузки: {df['critical_load_N_per_m'].min() / 1e3:.2f} - {df['critical_load_N_per_m'].max() / 1e3:.2f} кН/м\n"
        f"Средняя критическая нагрузка: {df['critical_load_N_per_m'].mean() / 1e3:.2f} кН/м"
    )

    if 'critical_load_clean_N_per_m' in df.columns:
        clean_stats = (
            f"\n(Без шума: {df['critical_load_clean_N_per_m'].min() / 1e3:.2f} - "
            f"{df['critical_load_clean_N_per_m'].max() / 1e3:.2f} кН/м)"
        )
        stats += clean_stats

    print("\n📊 Статистика данных:")
    print(stats)

    # Сохранение статистики в файл
    stats_filename = os.path.join(output_dir, f"data_statistics_{suffix}{timestamp}.txt")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        f.write(stats)
    print(f"✅ Статистика сохранена в: {stats_filename}")

    # ======================
    # ВИЗУАЛИЗАЦИЯ
    # ======================
    fig = plt.figure(figsize=(20, 12))

    # 1. Нагрузка vs Толщина
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(df['total_thickness_m'] * 1e3, df['critical_load_N_per_m'] / 1e3,
                alpha=0.7, c='blue', s=50)
    ax1.set_xlabel('Толщина пластины (мм)', fontsize=11)
    ax1.set_ylabel('Критическая нагрузка (кН/м)', fontsize=11)
    ax1.set_title('Зависимость критической нагрузки от толщины', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Нагрузка vs Угол укладки
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(df['theta_base_deg'], df['critical_load_N_per_m'] / 1e3,
                alpha=0.7, c='green', s=50)
    ax2.set_xlabel('Базовый угол укладки (градусы)', fontsize=11)
    ax2.set_ylabel('Критическая нагрузка (кН/м)', fontsize=11)
    ax2.set_title('Зависимость критической нагрузки от угла укладки', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Нагрузка vs Соотношение сторон
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(df['aspect_ratio'], df['critical_load_N_per_m'] / 1e3,
                alpha=0.7, c='red', s=50)
    ax3.set_xlabel('Соотношение сторон (a/b)', fontsize=11)
    ax3.set_ylabel('Критическая нагрузка (кН/м)', fontsize=11)
    ax3.set_title('Зависимость критической нагрузки от соотношения сторон', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Гистограмма распределения нагрузки
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(df['critical_load_N_per_m'] / 1e3, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Критическая нагрузка (кН/м)', fontsize=11)
    ax4.set_ylabel('Частота', fontsize=11)
    ax4.set_title('Распределение критической нагрузки', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 3D-график (толщина, угол, нагрузка)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    sc5 = ax5.scatter(df['total_thickness_m'] * 1e3,
                      df['theta_base_deg'],
                      df['critical_load_N_per_m'] / 1e3,
                      c=df['critical_load_N_per_m'] / 1e3,
                      cmap=cm.viridis,
                      s=50,
                      alpha=0.8)
    ax5.set_xlabel('Толщина (мм)', fontsize=10)
    ax5.set_ylabel('Угол (град)', fontsize=10)
    ax5.set_zlabel('Нагрузка (кН/м)', fontsize=10)
    ax5.set_title('3D: Толщина × Угол × Нагрузка', fontsize=11)
    plt.colorbar(sc5, ax=ax5, label='Нагрузка (кН/м)', shrink=0.6)

    # 6. 3D-график (толщина, aspect_ratio, нагрузка)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    sc6 = ax6.scatter(df['total_thickness_m'] * 1e3,
                      df['aspect_ratio'],
                      df['critical_load_N_per_m'] / 1e3,
                      c=df['critical_load_N_per_m'] / 1e3,
                      cmap=cm.plasma,
                      s=50,
                      alpha=0.8)
    ax6.set_xlabel('Толщина (мм)', fontsize=10)
    ax6.set_ylabel('Соотношение (a/b)', fontsize=10)
    ax6.set_zlabel('Нагрузка (кН/м)', fontsize=10)
    ax6.set_title('3D: Толщина × a/b × Нагрузка', fontsize=11)
    plt.colorbar(sc6, ax=ax6, label='Нагрузка (кН/м)', shrink=0.6)

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"buckling_data_analysis_{suffix}{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Графики сохранены в: {plot_filename}")
    plt.show()

    # ======================
    # ДОПОЛНИТЕЛЬНО: Сравнение зашумлённых и чистых данных
    # ======================
    if 'critical_load_clean_N_per_m' in df.columns and add_noise:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['critical_load_clean_N_per_m'] / 1e3,
                   df['critical_load_N_per_m'] / 1e3,
                   alpha=0.6, c='darkorange', s=50)

        # Линия идеального соответствия
        min_val = df['critical_load_clean_N_per_m'].min() / 1e3
        max_val = df['critical_load_clean_N_per_m'].max() / 1e3
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное соответствие')

        ax.set_xlabel('Чистая нагрузка (кН/м)', fontsize=11)
        ax.set_ylabel('Зашумлённая нагрузка (кН/м)', fontsize=11)
        ax.set_title('Сравнение чистых и зашумлённых данных', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        noise_plot_filename = os.path.join(output_dir, f"noise_comparison_{suffix}{timestamp}.png")
        plt.savefig(noise_plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ График сравнения шума сохранён в: {noise_plot_filename}")
        plt.show()


# ======================
# 3. ОСНОВНАЯ ЛОГИКА
# ======================
print("🚀 Запуск генерации данных...")
print("=" * 60)

# ======================
# ГЕНЕРАЦИЯ БЕЗ ШУМА
# ======================
print("\n[1/2] Генерация данных БЕЗ шума...")
df_no_noise = generate_data(n_points=n_points, add_noise=False, seed=seed)
save_and_visualize(df_no_noise, seed, "no_noise", suffix="no_noise_")

# ======================
# ГЕНЕРАЦИЯ С ШУМОМ
# ======================
print("\n[2/2] Генерация данных С шумом (5%)...")
df_with_noise = generate_data(n_points=n_points, add_noise=True, noise_std=noise_std, seed=seed)
save_and_visualize(df_with_noise, seed, "with_noise", suffix="with_noise_")

# ======================
# СРАВНЕНИЕ СТАТИСТИК
# ======================
print("\n" + "=" * 60)
print("📊 СРАВНЕНИЕ СТАТИСТИК:")
print("=" * 60)
print(f"\nБез шума:")
print(f"  Среднее: {df_no_noise['critical_load_N_per_m'].mean() / 1e3:.2f} кН/м")
print(f"  STD: {df_no_noise['critical_load_N_per_m'].std() / 1e3:.2f} кН/м")

print(f"\nС шумом:")
print(f"  Среднее: {df_with_noise['critical_load_N_per_m'].mean() / 1e3:.2f} кН/м")
print(f"  STD: {df_with_noise['critical_load_N_per_m'].std() / 1e3:.2f} кН/м")

# Вычисляем фактический уровень шума
actual_noise = (df_with_noise['critical_load_N_per_m'].std() - df_no_noise['critical_load_N_per_m'].std()) / \
               df_no_noise['critical_load_N_per_m'].mean() * 100
print(f"\n  Фактический уровень шума: ~{actual_noise:.1f}%")

print("\n" + "=" * 60)
print("🎉 Генерация данных завершена успешно!")
print(f"📁 Папка с данными: data")
print("   ├── no_noise/")
print("   └── with_noise/")
print("=" * 60)
