import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from src.visualization.style import BASELINE_COLOR, GA_COLOR, apply_style, OUTPUT_DIR

with open(Path(__file__).parent.parent.parent / 'viz_data' /'logs'/ 'logs.json') as f:
    data = json.load(f)

tpch4_base  = {k: tuple(v) for k, v in data['tpch4_base'].items()}
tpch4_cust  = {k: tuple(v) for k, v in data['tpch4_cust'].items()}
tpcds4_base = {k: tuple(v) for k, v in data['tpcds4_base'].items()}
tpcds4_cust = {k: tuple(v) for k, v in data['tpcds4_cust'].items()}

def compute_deltas(base, cust):
    pts = []
    for q in base:
        if q not in cust:
            continue
        bh, bt = base[q]
        ch, ct = cust[q]
        delta_hit  = ch - bh                        # pp improvement in hit rate
        delta_time = (bt - ct) / bt * 100           # % time saved (positive = faster)
        pts.append((delta_hit, delta_time, q, bt))  # bt = baseline time for dot sizing
    return pts

def make_plot(base, cust, title, outpath, label_thresh_ms=10000):
    pts = compute_deltas(base, cust)

    dx = [p[0] for p in pts]
    dy = [p[1] for p in pts]
    queries = [p[2] for p in pts]
    base_times = [p[3] for p in pts]

    # Dot size scaled by baseline execution time (heavier queries = bigger dot)
    max_t = max(base_times)
    sizes = [30 + 220 * (t / max_t) for t in base_times]

    # Color by quadrant: ideal (bottom-right → teal), regressed (top-left → coral), neutral (gray)
    colors = []
    for x, y in zip(dx, dy):
        if x >= 0 and y >= 0:
            colors.append('#1D9E75')   # improved both
        elif x < 0 and y < 0:
            colors.append('#D85A30')   # regressed both
        else:
            colors.append('#D85A30')   # mixed

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#EEECEA', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Zero reference lines
    ax.axhline(0, color='#B4B2A9', linewidth=1.0, linestyle='--', zorder=1)
    ax.axvline(0, color='#B4B2A9', linewidth=1.0, linestyle='--', zorder=1)

    ax.scatter(dx, dy, s=sizes, c=colors, alpha=0.75, zorder=3, edgecolors='white', linewidths=0.5)

    # Adjust y-axis limits to data range (instead of starting at 0)
    ymin = min(dy)
    ymax = max(dy)
    pad = (ymax - ymin) * 0.15  # 15% padding

    ax.set_ylim(ymin - pad, ymax + pad)

    # Label queries where baseline was heavy or change was dramatic
    for x, y, q, bt in pts:
        if bt > label_thresh_ms or abs(y) > 30 or abs(x) > 40:
            ax.annotate(q, (x, y), textcoords='offset points',
                        xytext=(5, 4), fontsize=7.5, color='#444441')

    # Quadrant shading (very subtle)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.set_title(title, fontsize=14, fontweight='500', pad=12)
    ax.set_xlabel('Cache hit rate improvement (pp)', fontsize=11, color='#5F5E5A')
    ax.set_ylabel('Execution time saved (%)', fontsize=11, color='#5F5E5A')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0f}pp'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.0f}%'))
    ax.tick_params(colors='#5F5E5A')
    ax.xaxis.label.set_color('#5F5E5A')
    ax.yaxis.label.set_color('#5F5E5A')

    # Quadrant labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # Light green region for positive hit-rate improvement
    ax.axvspan(0, xmax, color='#1D9E75', alpha=0.08, zorder=0)

    pad_x = (xmax - xmin) * 0.02
    pad_y = (ymax - ymin) * 0.02
    ax.text(xmax - pad_x, ymax - pad_y, 'faster + more hits', ha='right', va='top',
            fontsize=8, color='#1D9E75', style='italic')
    ax.text(xmin + pad_x, ymin + pad_y, 'slower + fewer hits', ha='left', va='bottom',
            fontsize=8, color='#D85A30', style='italic')

    # Legend for dot size
    for ms_val, label in [(5000, '5s baseline'), (30000, '30s baseline'), (100000, '100s baseline')]:
        if ms_val <= max_t:
            sz = 30 + 220 * (ms_val / max_t)
            ax.scatter([], [], s=sz, c='#888780', alpha=0.6, label=label)
    ax.legend(title='Dot size = baseline time', title_fontsize=9,
              fontsize=9, frameon=True, framealpha=1.0, edgecolor='#D3D1C7',
              loc='lower right')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"saved {outpath}")

out1 = OUTPUT_DIR / f"per_query_scatterplot_tpch4.png"
out2 = OUTPUT_DIR / f"per_query_scatterplot_tpcds4.png"

make_plot(tpch4_base,  tpch4_cust,  'TPC-H 4GB — GA impact per query', out1,  label_thresh_ms=8000)
make_plot(tpcds4_base, tpcds4_cust, 'TPC-DS 4GB — GA impact per query', out2, label_thresh_ms=20000)