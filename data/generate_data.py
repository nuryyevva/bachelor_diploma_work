import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.physics.buckling import (
    B_WIDTH_M,
    FEATURE_BOUNDS,
    N_PER_M_TO_KN_PER_M,
    calibrate_noise_sigma,
    calculate_D_matrix,
    calculate_Q_matrix,
    critical_buckling_load,
    evaluate_clean,
    evaluate_point,
    get_feature_bounds,
)

seed = 132
n_points = 1000
noise_std = 0.05
add_noise = True

theta_range = FEATURE_BOUNDS[0]
thickness_range = FEATURE_BOUNDS[1]
aspect_ratio_range = FEATURE_BOUNDS[2]


def generate_data(n_points, add_noise=False, noise_std=0.05, seed=42):
    """
    Генерация данных с опциональным добавлением шума.

    Шум: N(0, sigma), sigma = noise_std * std(N_cr_clean) по всей выборке.
    """
    rng = np.random.default_rng(seed)
    Q0 = calculate_Q_matrix()

    rows = []
    clean_values = []

    for _ in range(n_points):
        theta_base = rng.uniform(*theta_range)
        total_thickness = rng.uniform(*thickness_range)
        aspect_ratio = rng.uniform(*aspect_ratio_range)
        a = aspect_ratio * B_WIDTH_M

        angles = [theta_base, -theta_base, 90, 0, 0, 90, -theta_base, theta_base]
        layer_thickness = total_thickness / len(angles)
        thicknesses = [layer_thickness] * len(angles)

        D = calculate_D_matrix(angles, thicknesses, Q0)
        N_cr = critical_buckling_load(D[0, 0], D[1, 1], D[0, 1], D[2, 2], a)
        clean_values.append(N_cr)

        rows.append({
            "theta_base_deg": theta_base,
            "total_thickness_m": total_thickness,
            "aspect_ratio": aspect_ratio,
            "a_m": a,
            "D11_Nm": D[0, 0],
            "D12_Nm": D[0, 1],
            "D22_Nm": D[1, 1],
            "D66_Nm": D[2, 2],
            "critical_load_clean_N_per_m": N_cr,
        })

    sigma = noise_std * float(np.std(clean_values))
    noises = rng.normal(0, sigma, size=n_points)
    print(f"  Sigma шума: {sigma:.2f} Н/м ({noise_std * 100:.0f}% от std чистых значений)")
    print(f"  Mean noise: {np.mean(noises):.4f} (должно быть ~0)")

    for i, row in enumerate(rows):
        N_cr = row["critical_load_clean_N_per_m"]
        if add_noise:
            N_cr_noisy = max(N_cr + noises[i], 1.0)
        else:
            N_cr_noisy = N_cr
        row["critical_load_N_per_m"] = N_cr_noisy

    return pd.DataFrame(rows)


def save_and_visualize(df, seed, output_dir, suffix="", add_noise_flag=False):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = os.path.join(output_dir, f"composite_plate_buckling_data_{suffix}{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"✅ Данные сохранены в: {csv_filename}")

    stats = (
        f"Random seed: {seed}\n"
        f"Количество точек: {len(df)}\n"
        f"Диапазон толщины: {df['total_thickness_m'].min() * 1e3:.2f} - {df['total_thickness_m'].max() * 1e3:.2f} мм\n"
        f"Диапазон углов: {df['theta_base_deg'].min():.2f} - {df['theta_base_deg'].max():.2f} град\n"
        f"Диапазон соотношения сторон (a/b): {df['aspect_ratio'].min():.2f} - {df['aspect_ratio'].max():.2f}\n"
        f"Диапазон критической нагрузки: {df['critical_load_N_per_m'].min() / 1e3:.2f} - {df['critical_load_N_per_m'].max() / 1e3:.2f} кН/м\n"
        f"Средняя критическая нагрузка: {df['critical_load_N_per_m'].mean() / 1e3:.2f} кН/м"
    )

    if "critical_load_clean_N_per_m" in df.columns:
        stats += (
            f"\n(Без шума: {df['critical_load_clean_N_per_m'].min() / 1e3:.2f} - "
            f"{df['critical_load_clean_N_per_m'].max() / 1e3:.2f} кН/м)"
        )

    print("\n📊 Статистика данных:")
    print(stats)

    stats_filename = os.path.join(output_dir, f"data_statistics_{suffix}{timestamp}.txt")
    with open(stats_filename, "w", encoding="utf-8") as f:
        f.write(stats)
    print(f"✅ Статистика сохранена в: {stats_filename}")

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
    plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
    print(f"✅ Графики сохранены в: {plot_filename}")
    plt.close()

    if "critical_load_clean_N_per_m" in df.columns and add_noise_flag:
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            df["critical_load_clean_N_per_m"] / 1e3,
            df["critical_load_N_per_m"] / 1e3,
            alpha=0.5,
            s=20,
        )
        min_val = df["critical_load_clean_N_per_m"].min() / 1e3
        max_val = df["critical_load_clean_N_per_m"].max() / 1e3
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax.set_xlabel("Чистая (кН/м)")
        ax.set_ylabel("Зашумлённая (кН/м)")
        ax.grid(True, alpha=0.3)
        noise_plot_filename = os.path.join(output_dir, f"noise_comparison_{suffix}{timestamp}.png")
        plt.savefig(noise_plot_filename, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ График шума: {noise_plot_filename}")


if __name__ == "__main__":
    data_root = os.path.dirname(os.path.abspath(__file__))
    no_noise_dir = os.path.join(data_root, "no_noise")
    with_noise_dir = os.path.join(data_root, "with_noise")

    print("🚀 Запуск генерации данных...")
    print("=" * 60)

    calibrate_noise_sigma(noise_std, n_samples=1000, seed=seed)

    print("\n[1/2] Генерация данных БЕЗ шума...")
    df_no_noise = generate_data(n_points=n_points, add_noise=False, seed=seed)
    save_and_visualize(df_no_noise, seed, no_noise_dir, suffix="no_noise_")

    print("\n[2/2] Генерация данных С шумом...")
    df_with_noise = generate_data(n_points=n_points, add_noise=True, noise_std=noise_std, seed=seed)
    save_and_visualize(df_with_noise, seed, with_noise_dir, suffix="with_noise_", add_noise_flag=True)

    mean_clean = df_no_noise["critical_load_N_per_m"].mean() / 1e3
    mean_noisy = df_with_noise["critical_load_N_per_m"].mean() / 1e3
    diff_pct = abs(mean_noisy - mean_clean) / mean_clean * 100

    print("\n" + "=" * 60)
    print("📊 СРАВНЕНИЕ СРЕДНИХ (n=1000):")
    print(f"  Без шума:  {mean_clean:.2f} кН/м")
    print(f"  С шумом:   {mean_noisy:.2f} кН/м")
    print(f"  Разница:   {diff_pct:.2f}%")
    if diff_pct < 1.0:
        print("  ✅ Разница < 1% — смещение среднего незначимо")
    else:
        print("  ⚠️  Разница >= 1% — проверьте параметры шума")

    print("\n🎉 Генерация завершена")
