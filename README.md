# Разработка адаптивных суррогатных моделей для предсказания критической нагрузки композитной панели

Дипломный проект: сравнение адаптивных стратегий активного обучения на основе гауссовских процессов (GPR) для предсказания критической нагрузки при потере устойчивости (баклинге) композитной пластины.

Модели обучаются на **синтетических данных**, полученных из аналитической модели (теория ламинатов, формула Jones). Адаптивность достигается итеративным выбором новых точек в пространстве параметров с помощью acquisition functions.

**Автор:** Нурыева Айнур · [Telegram](https://t.me/nuryyevva)

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Структура проекта](#структура-проекта)
3. [Генерация данных](#генерация-данных)
4. [Сравнение суррогатных моделей](#сравнение-суррогатных-моделей)
5. [Эксперименты с адаптивными стратегиями](#эксперименты-с-адаптивными-стратегиями)
6. [Базовый эксперимент adaptive vs non-adaptive](#базовый-эксперимент-adaptive-vs-non-adaptive)
7. [Архитектура кода](#архитектура-кода)
8. [Результаты экспериментов](#результаты-экспериментов)

---

## Быстрый старт

### Установка окружения (Poetry)

```bash
pipx install poetry          # если Poetry ещё не установлен
cd bachelor_diploma_work
poetry install
```

Все команды ниже запускаются через `poetry run python ...` из корня проекта.

### Типичный порядок работы

```bash
# 1. Сгенерировать датасет (1000 точек, с шумом и без)
poetry run python data/generate_data.py

# Дискретная сетка параметров (все технологически допустимые комбинации)
poetry run python data/generate_data.py --discrete

# 2. Сравнить суррогатные модели на начальной выборке (35 точек)
poetry run python experiments/compare_models.py --n_seeds 10

# 3. Запустить все адаптивные стратегии (10 seed, статистика)
poetry run python experiments/run_strategies.py --all --n_seeds 10

# 4. Сравнение ядер для лучшей стратегии
poetry run python experiments/run_strategies.py \
    --strategy farthest_point \
    --kernels RBF Matern RationalQuadratic \
    --n_seeds 10 --noise_type with_noise

# 5. Граничные точки в пуле кандидатов + дискретный режим
poetry run python experiments/run_strategies.py --all --include_boundaries --discrete --n_seeds 10
```

---

## Структура проекта

```
bachelor_diploma_work/
├── data/
│   ├── generate_data.py          # Генерация синтетического датасета
│   ├── no_noise/                 # CSV без шума
│   └── with_noise/               # CSV с шумом (5% от std)
├── experiments/
│   ├── compare_models.py         # Сравнение GPR / Polynomial / RF
│   ├── run_strategies.py         # Основной скрипт адаптивных экспериментов
│   └── adaptive_vs_nonadaptive.py # Базовый эксперимент (одна стратегия)
├── src/
│   ├── models/
│   │   ├── adaptive.py           # Адаптивная GPR-модель
│   │   └── non_adaptive.py       # Неадаптивная GPR (baseline LHS)
│   ├── physics/
│   │   └── buckling.py           # Физическая модель + evaluate_point()
│   ├── strategies/                 # Стратегии выбора точек
│   │   ├── max_variance.py
│   │   ├── farthest_point.py
│   │   └── random_strategy.py
│   └── utils/
│       ├── data.py               # Загрузка и split данных
│       ├── discrete_params.py    # Дискретные значения параметров
│       ├── plot_style.py         # Единый стиль графиков (font_scale)
│       ├── preprocessor.py       # StandardScaler (без утечки)
│       ├── metrics.py            # RMSE, R², MAE
│       ├── evaluator.py          # evaluate_fn для адаптивного цикла
│       └── plots.py
├── results/                      # Отчёты, графики, boxplot'ы
├── pyproject.toml
└── README.md
```

---

## Генерация данных

### Запуск

```bash
poetry run python data/generate_data.py

# Дискретная сетка (все комбинации параметров)
poetry run python data/generate_data.py --discrete

# Увеличенный шрифт на графиках
poetry run python data/generate_data.py --font_scale 1.5
```

### Что происходит

- По умолчанию генерируется **1000 точек** в пространстве параметров (непрерывная выборка).
- С `--discrete` — **все комбинации** дискретных значений (сетка ~14 260 точек):
  - `theta_base_deg` — угол укладки (0–45°, шаг 5°)
  - `total_thickness_m` — толщина (2–20 мм, шаг 0.4 мм, N чётное)
  - `aspect_ratio` — соотношение сторон a/b (1.0–4.0, шаг 0.1)
- Укладка: симметричная `[θ, -θ, 90, 0]s` (8 слоёв), материал T300/934
- Целевая переменная: `critical_load_N_per_m` (Н/м), в экспериментах используется в **кН/м**
- Шум: `N(0, σ)`, где `σ = 0.05 × std(N_cr_clean)` по всей выборке (среднее шума ≈ 0)
- Проверка: разница средних чистого и зашумлённого датасета < 1%

### Выходные файлы

| Файл | Описание |
|------|----------|
| `data/no_noise/composite_plate_buckling_data_*.csv` | Данные без шума |
| `data/with_noise/composite_plate_buckling_data_*.csv` | Данные с шумом |
| `data/*/data_statistics_*.txt` | Статистика диапазонов |
| `data/*/buckling_data_analysis_*.png` | Визуализация зависимостей |
| `data/*/noise_comparison_*.png` | Чистые vs зашумлённые значения |

---

## Сравнение суррогатных моделей

Скрипт `experiments/compare_models.py` сравнивает три модели на **одной и той же** начальной выборке (35 точек) и тесте (20 точек) при фиксированных seed.

### Модели

| Модель | Реализация |
|--------|-----------|
| Полиномиальная (deg=2) | `PolynomialFeatures(degree=2)` + `LinearRegression` |
| Random Forest | `RandomForestRegressor(n_estimators=100, random_state=42)` |
| GPR Matérn | `NonAdaptiveModel(kernel='Matern')` — ν=1.5 |

### Запуск

```bash
# По умолчанию: 10 seed (42..51), зашумлённые данные
poetry run python experiments/compare_models.py

# Кастомные параметры
poetry run python experiments/compare_models.py \
    --noise_type with_noise \
    --n_seeds 10 \
    --seed_start 42 \
    --n_initial 35 \
    --test_size 20 \
    --save_dir results
```

### Аргументы

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--noise_type` | `with_noise` | `with_noise` или `no_noise` |
| `--n_initial` | 35 | Размер начальной обучающей выборки |
| `--test_size` | 20 | Размер тестовой выборки |
| `--n_seeds` | 10 | Число независимых запусков |
| `--seed_start` | 42 | Начальный seed (42, 43, …) |
| `--data_dir` | `data` | Папка с данными |
| `--save_dir` | `results` | Папка для результатов |
| `--rf_n_estimators` | 100 | Число деревьев RF |
| `--rf_random_state` | 42 | Seed для RF |
| `--font_scale` | 1.0 | Множитель размера шрифта на графиках |

### Результаты

| Файл | Описание |
|------|----------|
| `results/compare_models_*.txt` | Таблица RMSE/R² + детализация по seed |
| `results/comparison_models_boxplot.png` | Boxplot RMSE по трём моделям |

---

## Эксперименты с адаптивными стратегиями

Основной скрипт: `experiments/run_strategies.py`

### Доступные стратегии

| Ключ | Описание |
|------|----------|
| `max_variance` | Максимизация апостериорной дисперсии GP |
| `farthest_point` | Наиболее удалённая от обучающих точек |
| `random` | Случайный выбор из кандидатов |

> Стратегии Expected Improvement (EI) и Lower Confidence Bound (LCB) **удалены** из активных экспериментов.

### Как работает адаптивное обучение

1. Начальная выборка: **35 точек** из CSV-датасета (LHS split)
2. Тест: **20 точек** (фиксированный split по seed)
3. На **каждой итерации**:
   - Генерируется **1000 новых кандидатов** (LHS) в пределах `bounds` параметров
   - С `--include_boundaries`: дополнительно 8 угловых точек и ~10 точек на гранях гиперкуба
   - С `--discrete`: случайная выборка из дискретной сетки параметров
   - Стратегия выбирает лучшую точку по acquisition function
   - `y` вычисляется через **физическую модель** (`evaluate_point`), а не из пула
4. Метрики RMSE/R² считаются в **кН/м** на тестовой выборке

### Примеры запуска

```bash
# Все стратегии, один seed
poetry run python experiments/run_strategies.py --all

# Одна стратегия
poetry run python experiments/run_strategies.py --strategy farthest_point

# Несколько стратегий
poetry run python experiments/run_strategies.py --strategies max_variance farthest_point random

# 10 seed — статистически значимое сравнение
poetry run python experiments/run_strategies.py --all --n_seeds 10

# Контрольный эксперимент: random baseline (55/75/95 точек)
poetry run python experiments/run_strategies.py --all --random_baseline --n_seeds 10

# Сравнение ядер GP
poetry run python experiments/run_strategies.py \
    --strategy farthest_point \
    --kernels RBF Matern RationalQuadratic \
    --n_seeds 10

# Чистые данные, 50 итераций, подробный вывод
poetry run python experiments/run_strategies.py \
    --all --noise_type no_noise --max_iter 50 --verbose

# Чувствительность к шуму (1%, 3%, 5%, 10%) — автоматически при каждом запуске
poetry run python experiments/run_strategies.py --all --n_seeds 10 \
    --noise_levels 0.01 0.03 0.05 0.10

# Увеличенный шрифт для вставки в диплом
poetry run python experiments/run_strategies.py --all --n_seeds 10 --font_scale 1.5
```

### Аргументы `run_strategies.py`

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--all` | — | Запустить все стратегии |
| `--strategies` | — | Список стратегий |
| `--strategy` | — | Одна стратегия |
| `--noise_type` | `with_noise` | Тип данных |
| `--n_initial` | 35 | Начальных точек |
| `--test_size` | 20 | Тестовых точек |
| `--max_iter` | 50 | Макс. итераций адаптации |
| `--kernel` | `RBF` | Ядро GP (если `--kernels` не задан) |
| `--kernels` | — | Список ядер для сравнения |
| `--target_improvement` | 0.30 | Целевое улучшение RMSE для ранней остановки |
| `--n_seeds` | 1 | Число seed |
| `--seed_start` | 42 | Начальный seed |
| `--n_candidates` | 1000 | Кандидатов на итерацию |
| `--noise_levels` | 0.01 0.03 0.05 0.10 | Уровни шума для эксперимента чувствительности |
| `--include_boundaries` | false | Добавлять граничные/угловые точки в пул кандидатов |
| `--discrete` | false | Дискретный режим параметров |
| `--font_scale` | 1.0 | Множитель размера шрифта на графиках |
| `--random_baseline` | false | Контроль со случайными выборками |
| `--baseline_sizes` | 55 75 95 | Размеры для random baseline |
| `--baseline_repeats` | 5 | Повторов random baseline |
| `--save_dir` | `results` | Папка результатов |
| `--data_dir` | `data` | Папка данных |
| `--verbose` | false | Подробный вывод |

### Результаты `run_strategies.py`

| Режим | Файлы |
|-------|-------|
| 1 seed | `strategies_comparison_*.txt`, `convergence_all_strategies.png` |
| N seed | `strategies_multiseed_*.txt`, `strategies_rmse_boxplot_*.png`, `convergence_all_strategies.png`, `convergence_with_ci.png` |
| Сравнение ядер | `kernels_comparison_*.txt`, `kernels_boxplot_*.png` |
| Random baseline | раздел в отчёте с контрольными выборками |
| Шум | `noise_comparison_table.txt`, `noise_sensitivity_*.txt`, `noise_sensitivity.png` |

---

## Базовый эксперимент adaptive vs non-adaptive

```bash
poetry run python experiments/adaptive_vs_nonadaptive.py
```

Сравнивает одну адаптивную стратегию (`MaxVariance`) с неадаптивным GPR baseline. Результаты сохраняются в `results/`.

Настройка через класс `ExperimentConfig` внутри файла (noise_type, n_initial, max_iterations, kernel, n_candidates, seed).

---

## Архитектура кода

### Предобработка (`DataPreprocessor`)

- `fit()` только на начальной обучающей выборке (без утечки)
- `transform()` нормализует X и y
- `inverse_transform_y()` возвращает y в кН/м для метрик

### Физическая модель (`src/physics/buckling.py`)

- `evaluate_point(theta, thickness, aspect_ratio)` → N_cr в кН/м
- `get_feature_bounds()` → диапазоны параметров
- Используется в адаптивном цикле для оценки новых точек

### Метрики (`src/utils/metrics.py`)

- RMSE, R², MAE в физических единицах (кН/м)
- Параметр `debug=True` для диагностики единиц измерения

### Неадаптивная модель (`NonAdaptiveModel`)

GPR с ядрами: `RBF`, `Matern` (ν=1.5), `RationalQuadratic`. Используется как baseline (LHS на 35 точках).

### Адаптивная модель (`AdaptiveModel`)

- Динамическая генерация кандидатов (не фиксированный пул)
- Опционально: граничные точки (`include_boundaries`) и дискретная сетка (`discrete`)
- Замер времени обучения (`total_time`, `time_per_iteration`)
- Ранняя остановка при достижении `target_rmse`

### Стиль графиков (`src/utils/plot_style.py`)

- `apply_plot_style(font_scale)` — единые размеры шрифтов (мин. 12 pt) для всех элементов
- Используется в `run_strategies.py`, `compare_models.py`, `generate_data.py`

---

## Результаты экспериментов

Все результаты сохраняются в папку `results/` (или в `--save_dir`).

### Ключевые графики

| График | Описание |
|--------|----------|
| `comparison_models_boxplot.png` | Boxplot RMSE: Polynomial vs RF vs GPR |
| `convergence_all_strategies.png` | Сходимость RMSE по итерациям для всех стратегий |
| `convergence_with_ci.png` | Сходимость RMSE с доверительными интервалами ±1 std |
| `noise_sensitivity.png` | RMSE vs уровень шума для каждой стратегии |
| `strategies_rmse_boxplot_*.png` | Boxplot RMSE по seed для стратегий |
| `kernels_boxplot_*.png` | Boxplot RMSE по ядрам GP |

---

## Зависимости

- Python ≥ 3.12
- numpy, pandas, matplotlib, scipy, scikit-learn

Управление через Poetry (`pyproject.toml`, `poetry.lock`).

---

*Последнее обновление: июнь 2026*
