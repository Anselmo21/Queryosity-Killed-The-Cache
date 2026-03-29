"""
Plot 5: Cache Capacity Sensitivity

Sweeps cache size and plots hit ratio for baseline vs GA,
plus the improvement delta. Shows the sweet spot where
reordering has the most impact.
"""

from __future__ import annotations

from collections import OrderedDict
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


class _LRU:
    """Minimal table-level LRU cache used for the sensitivity sweep."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._d: OrderedDict[str, int] = OrderedDict()
        self._used = 0

    def access(self, table: str, pages: int) -> bool:
        if table in self._d:
            self._d.move_to_end(table)
            return True
        if pages > self.cap:
            return False
        while self._used + pages > self.cap and self._d:
            _, ep = self._d.popitem(last=False)
            self._used -= ep
        self._d[table] = pages
        self._used += pages
        return False


def _simulate(profiles: list[dict], schedule: list[int], cap: int) -> float:
    lru = _LRU(cap)
    hits = reqs = 0
    for idx in schedule:
        for table, pages in profiles[idx]["table_pages"].items():
            reqs += pages
            if lru.access(table, pages):
                hits += pages
    return hits / reqs if reqs > 0 else 0.0


def plot_cache_sensitivity(
    profiles: list[dict],
    baseline_schedule: list[int],
    ga_schedule: list[int],
    workload: str = "tpch",
    cache_sizes: list[int] | None = None,
) -> Path:
    """
    Sweep cache capacity and plot hit ratio and improvement.

    Parameters
    ----------
    profiles : list[dict]
        Serialized AccessProfile list (query_id, table_pages).
    baseline_schedule : list[int]
        Baseline permutation of query indices.
    ga_schedule : list[int]
        GA-optimized permutation of query indices.
    workload : str
        Workload name.
    cache_sizes : list[int] | None
        Cache capacities in pages to sweep. Defaults to log-spaced range
        from 1% to 200% of total workload pages.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    total_pages = sum(sum(p["table_pages"].values()) for p in profiles)

    if cache_sizes is None:
        cache_sizes = [
            int(total_pages * f)
            for f in np.geomspace(0.01, 2.0, 30)
        ]
        cache_sizes = sorted(set(max(1, c) for c in cache_sizes))

    baseline_hits = [_simulate(profiles, baseline_schedule, c) * 100 for c in cache_sizes]
    ga_hits       = [_simulate(profiles, ga_schedule, c) * 100 for c in cache_sizes]
    improvement   = [g - b for g, b in zip(ga_hits, baseline_hits)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(cache_sizes, baseline_hits, color=BASELINE_COLOR,
             linewidth=2, label="Baseline")
    ax1.plot(cache_sizes, ga_hits, color=GA_COLOR,
             linewidth=2, label="GA-Optimized")
    ax1.axvline(total_pages, color="gray", linestyle=":", linewidth=1,
                label="Total workload pages")
    ax1.set_ylabel("Cache Hit Ratio (%)")
    ax1.set_title(f"Cache Capacity Sensitivity — {workload.upper()}")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.legend(frameon=False)

    ax2.fill_between(cache_sizes, improvement, 0,
                     where=[v >= 0 for v in improvement],
                     color=GA_COLOR, alpha=0.4, label="GA better")
    ax2.fill_between(cache_sizes, improvement, 0,
                     where=[v < 0 for v in improvement],
                     color=BASELINE_COLOR, alpha=0.4, label="Baseline better")
    ax2.plot(cache_sizes, improvement, color="black", linewidth=1.5)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Improvement (pp)")
    ax2.set_xlabel("Cache Capacity (pages)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.set_xscale("log")

    out = OUTPUT_DIR / f"cache_sensitivity_{workload}.png"
    fig.savefig(out)
    plt.close(fig)
    return out