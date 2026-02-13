#!/usr/bin/python3.13
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
from math import inf
from sys import argv
from os.path import basename, realpath
from pathlib import Path

# ── 1. Load data ──────────────────────────────────────────────────────────────
rows = []
if len(argv) < 2:
    print(f"\nUsage {basename(realpath(argv[0]))} <relative_optimized_margin_sorted> [-timeout <seconds>]\n")  
    exit(0)
file_path = argv[1]
with open(file_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            label, value = parts[0], float(parts[1])
            m = re.match(r"o(\d+)_CH(\d+)RANK(\d+)Byte(\d+)", label)
            if m:
                o, ch, rank, byte = m.groups()
                rows.append({
                    "label": label,
                    "value": value,
                    "Octet": f"O{o}",
                    "Channel": f"CH{ch}",
                    "Rank": f"RANK{rank}",
                    "Byte": int(byte),
                    "group": f"O{o} CH{ch} RANK{rank}"
                })

df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
df["index"] = df.index

# ── 2. Colour palette by group ────────────────────────────────────────────────
groups = sorted(df["group"].unique())
palette = sns.color_palette("tab10", len(groups))
group_color = dict(zip(groups, palette))

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 9),
    gridspec_kw={"width_ratios": [2, 1]},
)
fig.patch.set_facecolor("#f8f9fb")

# ── LEFT: horizontal bar chart ────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#f0f2f5")

colors = [group_color[g] for g in df["group"]]
bars = ax.barh(
    df["index"], df["value"],
    color=colors, edgecolor="white", linewidth=0.4, height=0.85
)

# Add value labels on bars
for bar, val in zip(bars, df["value"]):
    x = bar.get_width()
    ax.text(
        x + 0.003, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", ha="left",
        fontsize=6.5, color="#333333"
    )

# Threshold reference line at 0.9
ax.axvline(0.9, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8, label="Threshold = 0.90")
ax.axvline(1.0, color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.8, label="Max = 1.00")

ax.set_yticks(df["index"])
ax.set_yticklabels(df["label"], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("Relative Optimized Margin", fontsize=11, labelpad=8)
substring_to_remove = "_relative_optimized_margin_sorted"
plot_name=Path(file_path).stem.replace('_relative_optimized_margin_sorted','')
ax.set_title(f"{plot_name} optimization Results — All Signals (sorted)", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=9, loc="lower right")

# Shade > 0.9 region
ax.axvspan(0.9, 1.08, alpha=0.06, color="#2ecc71")

# Grid
ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="white")
ax.set_axisbelow(True)

# ── RIGHT: dot-strip plot by Group ───────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#f0f2f5")

group_order = sorted(df["group"].unique())
for i, grp in enumerate(group_order):
    sub = df[df["group"] == grp]
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(sub))
    ax2.scatter(
        sub["value"], [i + j for j in jitter],
        color=group_color[grp], s=55, alpha=0.85,
        edgecolors="white", linewidths=0.5, zorder=3
    )
    # Mean line
    mean_val = sub["value"].mean()
    ax2.hlines(i, mean_val - 0.001, mean_val + 0.001, colors=group_color[grp],
               linewidths=0, zorder=2)
    ax2.scatter([mean_val], [i], color=group_color[grp], s=120,
                marker="D", edgecolors="#333", linewidths=0.8, zorder=4)

ax2.set_yticks(range(len(group_order)))
ax2.set_yticklabels(group_order, fontsize=9)
ax2.set_xlabel("Relative Optimized Margin", fontsize=11, labelpad=8)
ax2.set_title("Distribution by Group\n(◆ = mean)", fontsize=12, fontweight="bold", pad=12)
ax2.axvline(0.9, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)
ax2.axvspan(0.9, 1.08, alpha=0.06, color="#2ecc71")
ax2.set_xlim(0.4, 1.08)
ax2.xaxis.grid(True, linestyle="--", alpha=0.5, color="white")
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", labelsize=9)

# ── Summary stats box ─────────────────────────────────────────────────────────
n_total = len(df)
n_above = (df["value"] >= 0.9).sum()
stats_text = (
    f"Total signals: {n_total}\n"
    f"Signals ≥ 0.90: {n_above} ({100*n_above/n_total:.0f}%)\n"
    f"Signals < 0.90: {n_total - n_above} ({100*(n_total-n_above)/n_total:.0f}%)\n"
    f"Min: {df['value'].min():.4f}   Max: {df['value'].max():.4f}\n"
    f"Mean: {df['value'].mean():.4f}   Median: {df['value'].median():.4f}"
)
fig.text(
    0.5, 0.01, stats_text,
    ha="center", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#dce3ec", edgecolor="#aab4c2", alpha=0.9)
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"{plot_name}.png", dpi=150, bbox_inches="tight")
timeout = inf
if len(argv) > 3:
    if '-timeout' == argv[2]:
        timeout = int(argv[3]) 
if not inf == timeout:
    timer = fig.canvas.new_timer(interval=timeout*1000, callbacks=[(plt.close, [], {})])
    timer.start()
plt.show()
