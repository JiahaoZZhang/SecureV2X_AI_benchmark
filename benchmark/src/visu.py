import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Any

def plot_trajet(
    ax,
    y: np.ndarray,
    output: np.ndarray = None,
    context: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
    lim: Tuple[float, float] = None,
    colors: Tuple[str, str] = ("green", "red"),
    point_size: int = 10,
    line_width: float = 1.2):
    """
    Visualize a single (y, output) trajectory pair on one Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target subplot to draw on.
    y : np.ndarray
        Ground-truth trajectory, shape (T, 2).
    output : np.ndarray
        Predicted or reconstructed trajectory, shape (T, 2).
    context : np.ndarray | None
        Optional conditioning input (unused here but kept for API consistency).
    metadata : Any | None
        Optional sample metadata, e.g., ID or label.
    lim : tuple(float, float)
        Plot limits for both x and y axes.
    colors : tuple(str, str)
        Colors for (ground truth, prediction).
    point_size : int
        Size of scatter points.
    line_width : float
        Line width for plotted trajectories.
    """
    y = np.asarray(y)
    c_real, c_pred = colors
    # Ground truth trajectory
    ax.scatter(y[:, 0], y[:, 1], s=point_size, color=c_real, alpha=0.7)
    ax.plot(y[:, 0], y[:, 1], color=c_real, linewidth=line_width, label="Ground truth")

    # Predicted trajectory
    if(output is not None):
        output = np.asarray(output)
        ax.scatter(output[:, 0], output[:, 1], s=point_size, color=c_pred, alpha=0.7)
        ax.plot(output[:, 0], output[:, 1], color=c_pred, linewidth=line_width, linestyle="--", label="Predicted")
    if(lim is not None):
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if metadata is not None:
        ax.set_title(f"Sample {metadata}")

    ax.legend(fontsize=8, loc="upper right")