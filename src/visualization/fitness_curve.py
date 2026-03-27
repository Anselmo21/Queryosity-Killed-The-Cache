# GA convergence

"""
Plot 1: GA Fitness Curve

Best cache hit ratio per generation, with rolling average and optional
early stopping marker.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.visualization.style import (
    BASELINE_COLOR,
    GA_COLOR,
    ACCENT_COLOR,
    apply_style,
    OUTPUT_DIR,
)

apply_style()


def plot_fitness_curve(
    fitness_history: list[float],
    workload: str = "tpch",
    early_stop_gen: int | None = None,
) -> Path:
    """
    Plot best fitness (cache hit ratio) per GA generation.

    Parameters
    ----------
    fitness_history : list[float]
        Per-generation best fitness values from ScheduleResult.fitness_history.
    workload : str
        Workload name for the title.
    early_stop_gen : int | None
        Generation at which early stopping fired, if any.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    gens = list(range(len(fitness_history)))
    ax.plot(gens, [f * 100 for f in fitness_history],
            color=GA_COLOR, linewidth=2, label="Best fitness")

    if len(fitness_history) >= 10:
        window = 10
        rolling = np.convolve(fitness_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(fitness_history)),
                [r * 100 for r in rolling],
                color=BASELINE_COLOR, linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"Rolling avg (w={window})")

    if early_stop_gen is not None:
        ax.axvline(early_stop_gen, color=ACCENT_COLOR, linestyle=":",
                   linewidth=1.5, label=f"Early stop (gen {early_stop_gen})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cache Hit Ratio (%)")
    ax.set_title(f"GA Convergence — {workload.upper()}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(frameon=False)

    out = OUTPUT_DIR / f"fitness_curve_{workload}.png"
    fig.savefig(str(out.resolve()), bbox_inches="tight")
    plt.close(fig)
    return out