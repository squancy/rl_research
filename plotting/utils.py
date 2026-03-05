import numpy as np
from scipy.stats import gaussian_kde


def plot_kde(
    ax,
    data: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.85,
    fill_alpha: float = 0.10,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Plot a KDE curve with optional fill on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        data (np.ndarray): Data to plot.
        color (str): Color of the KDE plot.
        label (str): Label of the plot.
        linestyle (str = "-"): Line style of the plot.
        alpha (float = 0.85): Transparency of the plot.
        fill_alpha (float = 0.10): Transparency of the fill.
        xlim (tuple[float, float]): Lower and upper bound for values
            on the the x-axis.
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return
    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]
        if len(data) < 2:
            return
        xs = np.linspace(xlim[0], xlim[1], 400)
    else:
        pad = 0.05 * (data.max() - data.min())
        xs = np.linspace(data.min() - pad, data.max() + pad, 400)
    kde = gaussian_kde(data)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, alpha=alpha, lw=2, linestyle=linestyle, label=label)
    ax.fill_between(xs, ys, alpha=fill_alpha, color=color)
