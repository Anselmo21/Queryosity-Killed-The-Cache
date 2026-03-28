"""
Plot 6: Cumulative I/O Over Time

Cumulative page reads as queries execute, baseline vs GA-optimized.
A flatter slope = more cache reuse = fewer disk reads.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.visualization.style import (
    BASELINE_COLOR,
    GA_COLOR,
    ACCENT_COLOR,
    apply_style,
    OUTPUT_DIR,
)

apply_style()


def plot_cumulative_io(
    baseline_results: list[dict],
    ga_results: list[dict],
    workload: str = "tpch",
) -> Path:
    """
    Cumulative page reads as queries execute, baseline vs GA.

    Parameters
    ----------
    baseline_results : list[dict]
        Serialized QueryResult list in execution order.
        Keys: query_id, shared_hit_blocks, shared_read_blocks.
    ga_results : list[dict]
        Same for the GA-optimized run.
    workload : str
        Workload name.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    def _cumulative(results: list[dict]) -> list[int]:
        cumulative = []
        total = 0
        for r in results:
            total += r["shared_read_blocks"]
            cumulative.append(total)
        return cumulative

    b_cum = _cumulative(baseline_results)
    g_cum = _cumulative(ga_results)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(range(1, len(b_cum) + 1), b_cum,
            color=BASELINE_COLOR, linewidth=2, marker="o", markersize=4,
            label="Baseline")
    ax.plot(range(1, len(g_cum) + 1), g_cum,
            color=GA_COLOR, linewidth=2, marker="o", markersize=4,
            label="GA-Optimized")

    min_len = min(len(b_cum), len(g_cum))
    ax.fill_between(range(1, min_len + 1),
                    b_cum[:min_len], g_cum[:min_len],
                    alpha=0.15, color=ACCENT_COLOR)

    ax.set_xlabel("Query Execution Position")
    ax.set_ylabel("Cumulative Page Reads")
    ax.set_title(f"Cumulative Disk I/O — {workload.upper()}")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    ax.legend(frameon=False)

    out = OUTPUT_DIR / f"cumulative_io_{workload}.png"
    fig.savefig(out)
    plt.close(fig)
    return out