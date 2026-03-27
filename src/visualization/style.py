"""
Shared style constants and matplotlib configuration for all visualizations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_COLOR = "#4C72B0"
GA_COLOR       = "#DD8452"
ACCENT_COLOR   = "#55A868"
GRID_COLOR     = "#E5E5E5"

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


def apply_style() -> None:
    """Apply shared rcParams to all plots."""
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.8,
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
    })