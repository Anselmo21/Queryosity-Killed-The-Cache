"""
Plot 3: Per-Query Hit Ratio Bar Chart

Side-by-side bars per query showing actual cache hit ratio
from executor results: baseline vs GA-optimized.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.visualization.style import BASELINE_COLOR, GA_COLOR, apply_style, OUTPUT_DIR

apply_style()


def plot_per_query_hit_ratio(
    baseline_results: list[dict],
    ga_results: list[dict],
    workload: str = "tpch",
) -> Path:
    """
    Side-by-side bars per query: baseline vs GA actual hit ratio.

    Parameters
    ----------
    baseline_results : list[dict]
        List of dicts with keys: query_id, shared_hit_blocks, shared_read_blocks.
    ga_results : list[dict]
        Same format for the GA-optimized run.
    workload : str
        Workload name for the title.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    def _hit_ratio(r: dict) -> float:
        total = r["shared_hit_blocks"] + r["shared_read_blocks"]
        return r["shared_hit_blocks"] / total if total > 0 else 0.0

    baseline_map = {r["query_id"]: _hit_ratio(r) for r in baseline_results}
    ga_map        = {r["query_id"]: _hit_ratio(r) for r in ga_results}
    all_ids = sorted(
        set(baseline_map) | set(ga_map),
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )

    x = np.arange(len(all_ids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(all_ids) * 0.6), 5))

    b_vals = [baseline_map.get(q, 0) * 100 for q in all_ids]
    g_vals = [ga_map.get(q, 0) * 100 for q in all_ids]

    ax.bar(x - width / 2, b_vals, width, label="Baseline",
           color=BASELINE_COLOR, alpha=0.85)
    ax.bar(x + width / 2, g_vals, width, label="GA-Optimized",
           color=GA_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cache Hit Ratio (%)")
    ax.set_title(f"Per-Query Cache Hit Ratio — {workload.upper()}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(frameon=False)

    out = OUTPUT_DIR / f"per_query_hit_ratio_{workload}.png"
    fig.savefig(out)
    plt.close(fig)
    return out