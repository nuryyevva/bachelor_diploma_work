"""
Физическая модель критической нагрузки композитной пластины.
Используется при генерации данных и в адаптивном цикле (evaluate_point).
"""

import numpy as np

# Свойства материала (углепластик T300/934)
E1 = 150e9
E2 = 9e9
G12 = 7e9
nu12 = 0.32

# Геометрия и граничные условия
B_WIDTH_M = 0.5
M_HALF_WAVES = 1
K_SSSS = 4

# Допустимые диапазоны параметров
THETA_RANGE = (0.0, 45.0)
THICKNESS_RANGE = (2e-3, 20e-3)
ASPECT_RATIO_RANGE = (1.0, 4.0)

FEATURE_BOUNDS = [THETA_RANGE, THICKNESS_RANGE, ASPECT_RATIO_RANGE]
FEATURE_NAMES = ["theta_base_deg", "total_thickness_m", "aspect_ratio"]

N_PER_M_TO_KN_PER_M = 1e-3

_noise_sigma_cache: dict = {}


def get_feature_bounds():
    """Возвращает bounds [(min, max), ...] для [theta, thickness, aspect_ratio]."""
    return list(FEATURE_BOUNDS)


def calculate_Q_matrix(E1_val=E1, E2_val=E2, G12_val=G12, nu12_val=nu12):
    nu21 = nu12_val * E2_val / E1_val
    Q11 = E1_val / (1 - nu12_val * nu21)
    Q12 = nu12_val * E2_val / (1 - nu12_val * nu21)
    Q22 = E2_val / (1 - nu12_val * nu21)
    Q66 = G12_val
    return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])


def transform_Q(Q, theta_deg):
    theta = np.radians(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    T_inv = np.array([
        [c ** 2, s ** 2, -2 * c * s],
        [s ** 2, c ** 2, 2 * c * s],
        [c * s, -c * s, c ** 2 - s ** 2],
    ])
    return T_inv.T @ Q @ T_inv


def calculate_D_matrix(angles, thicknesses, Q0):
    total_thickness = sum(thicknesses)
    z = []
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


def critical_buckling_load(D11, D22, D12, D66, a, b=B_WIDTH_M, m=M_HALF_WAVES):
    term1 = D11 * (m * b / a) ** 2
    term2 = 2 * (D12 + 2 * D66)
    term3 = D22 * (a / (m * b)) ** 2
    return (np.pi ** 2 / b ** 2) * (term1 + term2 + term3)


def evaluate_clean(theta, thickness, aspect_ratio):
    """Чистое значение N_cr в Н/м (без шума)."""
    Q0 = calculate_Q_matrix()
    a = aspect_ratio * B_WIDTH_M
    angles = [theta, -theta, 90, 0, 0, 90, -theta, theta]
    layer_thickness = thickness / len(angles)
    thicknesses = [layer_thickness] * len(angles)
    D = calculate_D_matrix(angles, thicknesses, Q0)
    return critical_buckling_load(D[0, 0], D[1, 1], D[0, 1], D[2, 2], a)


def calibrate_noise_sigma(noise_level=0.05, n_samples=1000, seed=42):
    """
    Калибрует sigma шума как noise_level * std(N_cr) по выборке чистых значений.
    """
    cache_key = (noise_level, n_samples, seed)
    if cache_key in _noise_sigma_cache:
        return _noise_sigma_cache[cache_key]

    rng = np.random.default_rng(seed)
    clean = []
    for _ in range(n_samples):
        theta = rng.uniform(*THETA_RANGE)
        thickness = rng.uniform(*THICKNESS_RANGE)
        aspect = rng.uniform(*ASPECT_RATIO_RANGE)
        clean.append(evaluate_clean(theta, thickness, aspect))

    sigma = noise_level * float(np.std(clean))
    _noise_sigma_cache[cache_key] = sigma
    return sigma


def evaluate_point(
    theta,
    thickness,
    aspect_ratio,
    add_noise=False,
    noise_level=0.05,
    rng=None,
):
    """
    Вычисляет критическую нагрузку для точки параметров.

    Returns:
        float: N_cr в кН/м.
    """
    n_cr = evaluate_clean(theta, thickness, aspect_ratio)

    if add_noise:
        if rng is None:
            rng = np.random.default_rng()
        sigma = calibrate_noise_sigma(noise_level)
        noise = rng.normal(0, sigma)
        n_cr = max(n_cr + noise, 1.0)

    return n_cr * N_PER_M_TO_KN_PER_M


def evaluate_features(X, add_noise=False, noise_level=0.05, rng=None):
    """Вычисляет y для массива признаков shape (n, 3) или (3,)."""
    X = np.asarray(X)
    if X.ndim == 1:
        return evaluate_point(
            X[0], X[1], X[2],
            add_noise=add_noise,
            noise_level=noise_level,
            rng=rng,
        )

    if rng is None:
        rng = np.random.default_rng()

    return np.array([
        evaluate_point(
            row[0], row[1], row[2],
            add_noise=add_noise,
            noise_level=noise_level,
            rng=rng,
        )
        for row in X
    ])
