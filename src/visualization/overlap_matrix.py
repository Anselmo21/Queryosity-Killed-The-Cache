# page access overlap matrix

"""
Plot 4: Page Overlap Matrix

Heatmap of |P(qi) ∩ P(qj)| (shared pages) between all query pairs.
Optionally reorders axes by the GA schedule to reveal clustering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.visualization.style import apply_style, OUTPUT_DIR

apply_style()


def plot_page_overlap_matrix(
    profiles: list[dict],
    ga_schedule: list[int] | None = None,
    workload: str = "tpch",
) -> Path:
    """
    Heatmap of page access overlap between all query pairs.

    Parameters
    ----------
    profiles : list[dict]
        List of dicts with keys: query_id, table_pages (dict[str, int]).
    ga_schedule : list[int] | None
        GA-optimized permutation of query indices. If provided, reorders
        the matrix axes to reflect GA execution order.
    workload : str
        Workload name for the title.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    n = len(profiles)
    query_ids = [p["query_id"] for p in profiles]

    overlap = np.zeros((n, n), dtype=float)
    for i in range(n):
        tables_i = set(profiles[i]["table_pages"].keys())
        pages_i  = profiles[i]["table_pages"]
        for j in range(n):
            shared = tables_i & set(profiles[j]["table_pages"].keys())
            shared_pages = sum(
                min(pages_i[t], profiles[j]["table_pages"][t]) for t in shared
            )
            min_total = min(
                sum(pages_i.values()),
                sum(profiles[j]["table_pages"].values()),
            )
            overlap[i, j] = (shared_pages / min_total * 100) if min_total > 0 else 0.0

    order = ga_schedule if ga_schedule is not None else list(range(n))
    overlap_ordered = overlap[np.ix_(order, order)]
    labels_ordered = [query_ids[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))

    im = ax.imshow(overlap_ordered, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Shared Pages (min overlap)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_ordered, rotation=90, fontsize=7)
    ax.set_yticklabels(labels_ordered, fontsize=7)
    ax.grid(False)

    suffix = " (reordered by GA)" if ga_schedule is not None else ""
    ax.set_title(f"Page Access Overlap Matrix — {workload.upper()}{suffix}")

    out = OUTPUT_DIR / f"overlap_matrix_{workload}.png"
    fig.savefig(str(out.resolve()), bbox_inches="tight")
    plt.close(fig)
    return out
