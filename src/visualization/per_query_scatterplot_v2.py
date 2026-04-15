import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.visualization.style import BASELINE_COLOR, GA_COLOR, apply_style, OUTPUT_DIR

with open(Path(__file__).parent.parent.parent / 'viz_data' / 'logs' /'logs.json') as f:
    data = json.load(f)

tpch4_base  = {k: tuple(v) for k, v in data['tpch4_base'].items()}
tpch4_cust  = {k: tuple(v) for k, v in data['tpch4_cust'].items()}
tpcds4_base = {k: tuple(v) for k, v in data['tpcds4_base'].items()}
tpcds4_cust = {k: tuple(v) for k, v in data['tpcds4_cust'].items()}

def compute(base, cust):
    pts = []
    for q in base:
        if q not in cust:
            continue
        bh, bt = base[q]
        ch, ct = cust[q]
        pct_saved = (bt - ct) / bt * 100
        ms_saved  = bt - ct
        pts.append((bt, pct_saved, ms_saved, q))
    return pts

def make_plot(base, cust, title, outpath, label_thresh_bt=8000, label_thresh_pct=25):
    pts = compute(base, cust)
    bt_vals  = [p[0] for p in pts]
    pct_vals = [p[1] for p in pts]
    ms_vals  = [p[2] for p in pts]
    names    = [p[3] for p in pts]

    # Dot size proportional to ms saved (floor at 20 so tiny dots are still visible)
    max_ms = max(ms_vals)
    sizes  = [max(20, 600 * max(0, v) / max_ms) for v in ms_vals]

    # Color: teal = saved time, coral = got slower, gray = negligible
    colors = []
    for ms in ms_vals:
        if ms > 500:       colors.append('#1D9E75')
        elif ms < -500:    colors.append('#D85A30')
        else:              colors.append('#B4B2A9')

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#EEECEA', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color='#B4B2A9', linewidth=1.0, linestyle='--', zorder=1)

    sc = ax.scatter(bt_vals, pct_vals, s=sizes, c=colors,
                    alpha=0.75, zorder=3, edgecolors='white', linewidths=0.5)
    
    # Adjust y-axis limits to data range (instead of starting at 0)
    y_min = min(pct_vals) - 2
    y_max = max(pct_vals) + 2
    ax.set_ylim(y_min, y_max)


    for bt, pct, ms, q in pts:
        if bt > label_thresh_bt or abs(pct) > label_thresh_pct:
            ax.annotate(q, (bt, pct), textcoords='offset points',
                        xytext=(5, 4), fontsize=7.5, color='#444441')

    ax.set_xscale('log')
    ax.set_xlabel('Baseline execution time (ms, log scale)', fontsize=11, color='#5F5E5A')
    ax.set_ylabel('Execution time saved (%)', fontsize=11, color='#5F5E5A')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{x/1000:.0f}s' if x >= 1000 else f'{x:.0f}ms'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.0f}%'))
    ax.tick_params(colors='#5F5E5A')
    ax.xaxis.label.set_color('#5F5E5A')
    ax.yaxis.label.set_color('#5F5E5A')
    ax.set_title(title, fontsize=13, fontweight='500', pad=12, color='#2C2C2A')

    # Dot size legend (ms saved)
    ref_vals = [1000, 10000, 50000]
    for rv in ref_vals:
        if rv <= max_ms:
            sz = max(20, 600 * rv / max_ms)
            label = f'{rv/1000:.0f}s saved' if rv >= 1000 else f'{rv}ms saved'
            ax.scatter([], [], s=sz, c='#1D9E75', alpha=0.7, label=label)
    ax.legend(title='Dot size = time saved', title_fontsize=9,
              fontsize=9, frameon=True, framealpha=1.0,
              edgecolor='#D3D1C7', loc='upper left', 
              #borderpad=1.1,      # space inside box
              #labelspacing=1.1,   # space between entries
              #handletextpad=1.2,  # space between marker and text
              markerscale=0.6
              )

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"saved {outpath}")

out1 = OUTPUT_DIR / f"per_query_scatterplot_v2_tpch4.png"
out2 = OUTPUT_DIR / f"per_query_scatterplot_v2_tpcds4.png"

# make_plot(tpch4_base,  tpch4_cust, 'TPC-H 4GB — GA impact per query', out1, label_thresh_bt=5000, label_thresh_pct=20)

make_plot(tpcds4_base, tpcds4_cust,
          'TPC-DS 4GB — GA impact per query',
          out2,
          label_thresh_bt=15000, label_thresh_pct=20)