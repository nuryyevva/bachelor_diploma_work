"""Единый стиль графиков для экспериментов и диплома."""

FONT_SIZE = {
    "title": 14,
    "axis_labels": 12,
    "tick_labels": 12,
    "legend": 12,
    "annotation": 11,
}


def apply_plot_style(font_scale: float = 1.0):
    """
    Применяет единый стиль matplotlib через rcParams.

    Args:
        font_scale: Множитель размера шрифта (1.0 — базовый, ≥1.0 — для вставки в диплом).
    """
    import matplotlib.pyplot as plt

    sizes = {key: value * font_scale for key, value in FONT_SIZE.items()}
  # Базовый размер не менее 12 pt
    min_size = 12 * font_scale

    plt.rcParams.update(
        {
            "font.size": max(sizes["tick_labels"], min_size),
            "axes.titlesize": max(sizes["title"], min_size),
            "axes.labelsize": max(sizes["axis_labels"], min_size),
            "xtick.labelsize": max(sizes["tick_labels"], min_size),
            "ytick.labelsize": max(sizes["tick_labels"], min_size),
            "legend.fontsize": max(sizes["legend"], min_size),
        }
    )
