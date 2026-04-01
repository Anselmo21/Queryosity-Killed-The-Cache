import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pathlib import Path
from src.visualization.style import BASELINE_COLOR, GA_COLOR, apply_style, OUTPUT_DIR


fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('white')

data = {
    'TPC-H\n(22 queries)':  {'approx': 3.5,  'exact': 32400, 'speedup': 9257},
    'TPC-DS\n(93 queries)': {'approx': 29.2, 'exact': 72000, 'speedup': 2466},
}

labels  = list(data.keys())
approx  = [data[k]['approx'] for k in labels]
exact   = [data[k]['exact']  for k in labels]
speedup = [data[k]['speedup'] for k in labels]

COLOR_APPROX = BASELINE_COLOR
COLOR_EXACT  = GA_COLOR
BAR_W = 0.38
x = np.array([0, 1])

# ── Left: log-scale grouped bar ──────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#EEECEA', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

bars_a = ax.bar(x - BAR_W/2, approx, BAR_W, color=COLOR_APPROX, zorder=3, label='Approximate (overlap matrix)')
bars_e = ax.bar(x + BAR_W/2, exact,  BAR_W, color=COLOR_EXACT,  zorder=3, label='Exact (cache simulation)')

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, color='#444441')
ax.set_ylabel('Runtime (seconds, log scale)', fontsize=11, color='#5F5E5A')
ax.tick_params(colors='#5F5E5A')
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda y, _: f'{y:,.0f}s' if y >= 1 else f'{y:.2f}s'))

# Value labels on bars
for bar, val in zip(bars_a, approx):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
            f'{val}s', ha='center', va='bottom', fontsize=9.5,
            color=COLOR_APPROX, fontweight='500')
for bar, val in zip(bars_e, exact):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
            f'{val:,}s', ha='center', va='bottom', fontsize=9.5,
            color=COLOR_EXACT, fontweight='500')

ax.legend(fontsize=9.5, frameon=True, framealpha=1.0, edgecolor='#D3D1C7',
          loc='upper left')
ax.set_title('Runtime comparison (log scale)', fontsize=12, fontweight='500', pad=10, color='#2C2C2A')

# ── Right: speedup bar ───────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.grid(axis='x', color='#EEECEA', linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)

colors_su = [COLOR_APPROX, COLOR_APPROX]
y_pos = [0, 1]
bars_su = ax2.barh(y_pos, speedup, color=COLOR_APPROX, height=0.45, zorder=3)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=11, color='#444441')
ax2.tick_params(left=False, colors='#5F5E5A')
ax2.set_xlabel('Speedup factor (×)', fontsize=11, color='#5F5E5A')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}×'))

for bar, val in zip(bars_su, speedup):
    ax2.text(bar.get_width() + max(speedup)*0.01, bar.get_y() + bar.get_height()/2,
             f'{val:,}×', va='center', fontsize=11, fontweight='500', color=COLOR_APPROX)

ax2.set_xlim(0, max(speedup) * 1.18)
ax2.set_title('Approximate vs. exact speedup', fontsize=12, fontweight='500', pad=10, color='#2C2C2A')

# Annotation
ax2.annotate('Approximate fitness is\n9,257× faster for TPC-H',
             xy=(9257, 1), xytext=(5500, 0.35),
             fontsize=8, color='#5F5E5A', style='italic',
             arrowprops=dict(arrowstyle='->', color='#B4B2A9', lw=1))

plt.tight_layout(w_pad=3)
out = OUTPUT_DIR / f"ga_overhead.png"
plt.savefig(out,
            dpi=180, bbox_inches='tight', facecolor='white')

print("saved")