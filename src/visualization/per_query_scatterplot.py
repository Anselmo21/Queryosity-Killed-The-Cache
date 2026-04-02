import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.visualization.style import BASELINE_COLOR, GA_COLOR, apply_style, OUTPUT_DIR


tpch4_base = {
    'q15':(0.00,8716.0),'q1':(0.00,13842.3),'q13':(98.45,9582.9),
    'q14':(1.12,2177.3),'q18':(63.87,23330.7),'q4':(68.54,2768.0),
    'q20':(47.93,5042.4),'q16':(49.02,2266.6),'q21':(76.15,11202.7),
    'q9':(49.73,12390.1),'q8':(16.76,3644.9),'q11':(76.44,1315.0),
    'q12':(32.64,9992.5),'q17':(95.67,1336.8),'q2':(51.80,4634.9),
    'q19':(39.06,652.4),'q6':(6.57,4419.5),'q22':(91.89,996.6),
    'q10':(68.41,3965.4),'q5':(38.19,24827.6),'q7':(3.68,8506.1),
    'q3':(2.90,10030.3),
}
tpch4_cust = {
    'q11':(79.65,809.0),'q16':(26.69,1815.0),'q19':(43.20,491.1),
    'q2':(90.69,3144.5),'q17':(97.86,947.6),'q14':(30.73,1747.5),
    'q20':(61.23,2765.3),'q9':(58.95,8223.8),'q13':(99.92,8784.3),
    'q8':(53.94,2161.1),'q22':(99.99,758.8),'q4':(76.24,2109.0),
    'q10':(74.27,3270.6),'q21':(80.00,9127.9),'q12':(42.87,6703.4),
    'q18':(68.41,15210.5),'q1':(15.00,10692.2),'q3':(17.47,8388.0),
    'q7':(17.60,6910.9),'q6':(15.00,3590.6),'q15':(15.00,3479.9),
    'q5':(47.02,19512.9),
}
tpcds4_base = {
    'q77':(23.14,9739.3),'q84':(41.98,185.0),'q98':(15.55,3537.0),
    'q92':(47.10,1070.0),'q80':(88.99,9717.8),'q39a':(0.50,101736.5),
    'q54':(25.09,3273.6),'q23a':(90.46,73894.1),'q83':(98.29,3961.5),
    'q12':(96.61,468.6),'q86':(33.97,3954.1),'q22':(1.37,67724.5),
    'q28':(0.07,7558.0),'q50':(73.67,2215.1),'q8':(84.08,3115.4),
    'q40':(56.75,825.4),'q15':(65.97,1019.1),'q90':(58.40,1350.7),
    'q88':(59.42,3349.0),'q36':(83.29,5692.5),'q5':(96.80,6287.7),
    'q47':(45.64,19567.0),'q7':(98.49,12639.0),'q66':(47.00,24224.1),
    'q23b':(90.60,59651.1),'q46':(95.65,2090.5),'q35':(82.24,26126.7),
    'q38':(50.78,13514.5),'q65':(87.83,12857.6),'q26':(99.07,6090.6),
    'q60':(68.14,22116.5),'q13':(91.41,2903.3),'q41':(99.92,27540.0),
    'q59':(98.63,14416.7),'q71':(97.42,2032.0),'q48':(88.46,4038.1),
    'q39b':(0.50,73845.9),'q89':(80.61,3936.4),'q61':(98.17,1037.1),
    'q57':(7.52,10122.1),'q67':(81.21,28313.7),'q85':(92.90,1936.0),
    'q21':(97.80,3101.8),'q62':(90.91,6042.7),'q43':(48.34,5514.7),
    'q93':(60.84,8561.5),'q24b':(97.15,6590.5),'q56':(62.46,18752.7),
    'q82':(92.52,194.1),'q70':(67.92,15979.3),'q2':(52.73,6895.5),
    'q25':(87.04,12602.3),'q97':(45.18,8351.2),'q94':(88.96,2805.1),
    'q31':(77.20,27979.3),'q19':(64.57,650.9),'q75':(89.87,16120.7),
    'q20':(25.68,201.6),'q18':(92.42,4787.0),'q32':(94.80,466.7),
    'q51':(48.21,16602.0),'q45':(98.74,887.3),'q78':(73.46,17694.1),
    'q91':(94.87,189.1),'q76':(25.95,2790.0),'q52':(89.04,1048.2),
    'q10':(63.40,17785.1),'q33':(88.51,53200.1),'q63':(58.80,1886.9),
    'q3':(82.29,96.1),'q69':(70.95,616.5),'q34':(80.40,971.3),
    'q79':(85.68,1278.8),'q37':(97.51,65.4),'q9':(86.03,24640.8),
    'q73':(86.95,560.1),'q58':(99.04,1447.5),'q44':(94.96,1586.4),
    'q55':(90.43,876.4),'q68':(97.34,904.6),'q87':(70.10,12140.3),
    'q53':(76.33,1485.6),'q74':(82.09,269765.6),'q64':(93.59,7582.1),
    'q49':(91.49,2720.1),'q99':(98.85,5993.3),'q96':(26.41,356.7),
    'q27':(98.30,9710.7),'q16':(99.27,25263.2),'q42':(97.38,995.0),
    'q24a':(97.86,4303.9),'q29':(81.66,2168.3),'q17':(94.15,5226.4),
}
tpcds4_cust = {
    'q64':(99.85,6098.4),'q66':(60.33,19486.1),'q2':(65.40,5520.6),
    'q33':(94.87,42249.0),'q35':(94.89,21138.1),'q5':(99.67,5066.8),
    'q45':(99.86,716.9),'q86':(47.46,3136.5),'q90':(71.38,1077.9),
    'q55':(99.97,703.4),'q94':(99.69,2266.0),'q24a':(99.75,3495.0),
    'q17':(99.70,4143.5),'q24b':(99.82,5328.8),'q67':(93.90,22519.9),
    'q52':(99.97,845.0),'q7':(95.49,10042.6),'q76':(39.25,2243.4),
    'q89':(92.75,3126.9),'q59':(97.57,11624.7),'q23b':(99.67,47322.0),
    'q23a':(99.78,59343.2),'q88':(72.58,2670.5),'q44':(99.95,1284.1),
    'q43':(61.26,4381.6),'q63':(72.28,1523.4),'q53':(89.11,1204.5),
    'q36':(96.40,4538.8),'q42':(99.92,788.7),'q98':(28.73,2855.4),
    'q70':(80.61,12690.2),'q58':(99.69,1159.0),'q8':(97.19,2476.6),
    'q27':(95.00,7838.5),'q40':(70.15,665.5),'q65':(99.69,10331.5),
    'q34':(93.71,772.6),'q46':(99.92,1661.8),'q10':(76.48,14180.5),
    'q48':(99.76,3276.6),'q38':(63.89,10730.8),'q87':(83.31,9686.1),
    'q78':(86.09,14358.2),'q97':(57.96,6699.1),'q37':(99.62,52.7),
    'q9':(97.11,19824.0),'q96':(39.10,287.9),'q79':(98.63,1033.4),
    'q50':(86.45,1781.0),'q82':(99.89,155.3),'q28':(12.25,6084.1),
    'q49':(99.98,2168.2),'q25':(99.66,10149.6),'q20':(38.62,161.3),
    'q29':(94.92,1733.5),'q12':(99.87,374.8),'q77':(35.74,7863.9),
    'q93':(73.72,6922.5),'q83':(99.94,3179.0),'q22':(14.91,54906.0),
    'q39b':(13.58,59972.7),'q61':(99.39,823.3),'q39a':(13.16,81523.3),
    'q41':(99.85,22088.3),'q84':(55.02,149.1),'q32':(99.88,374.6),
    'q21':(99.88,2513.4),'q69':(83.73,493.5),'q26':(99.57,4892.5),
    'q16':(99.90,20099.2),'q18':(99.85,3858.5),'q92':(60.70,851.4),
    'q3':(95.95,76.5),'q85':(99.92,1549.0),'q57':(20.73,8127.1),
    'q54':(38.59,2602.0),'q15':(79.06,814.8),'q71':(99.90,1611.5),
    'q80':(99.96,7892.8),'q99':(99.87,4837.5),'q73':(99.95,452.3),
    'q91':(99.84,150.4),'q19':(77.80,526.7),'q68':(99.69,725.2),
    'q13':(99.83,2344.3),'q56':(74.06,14903.9),'q60':(78.24,17747.8),
    'q75':(96.41,13011.7),'q47':(59.18,15761.6),'q51':(61.39,13249.1),
    'q74':(95.15,218286.3),'q31':(89.49,22214.3),'q62':(99.50,4826.4),
}

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
            colors.append('#888780')   # mixed

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