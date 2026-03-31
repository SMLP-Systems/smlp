#!/usr/bin/env python3.11
"""
Witness Certification Visualizations
Produces  witness_geometry.png  – matplotlib geometry plot
"""

import math
import textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from math import inf
from sys import argv

# ── shared palette ───────────────────────────────────────────────────────────
C_BLUE      = "#378ADD"
C_GREEN     = "#1D9E75"
C_RED       = "#E24B4A"
C_NAVYBLUE  = "#000080"
C_GRAY      = "#888780"
C_MAGENTA   = "#FF00FF"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – Geometry (matplotlib → PNG)
# ════════════════════════════════════════════════════════════════════════════
def plot_geometry(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.85)
    ax.set_xlabel("x₁", fontsize=13)
    ax.set_ylabel("x₂", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title("Witness certification\n"
                 r"$f_1(x)=4x_1^2+4x_2^2$,  query: $(f_1-0.692)^2<4$ and $x_1 ≥ 0$ and $x_2 ≥ 0$",
                 fontsize=15, pad=10)

    # ── constraint boundary |x| = 0.580091 (quarter circle) ──────────────────
    R_constraint = 0.580091

    wedge = mpatches.Wedge((0,0), R_constraint, 0, 90, facecolor='lightgreen', edgecolor='blue', linestyle='dashed')
    ax.add_patch(wedge)

    ang = math.radians(48)
    ax.annotate(f"|x| = {R_constraint}",
                xy=(R_constraint * math.cos(ang), R_constraint * math.sin(ang)),
                xytext=(0.62, 0.50), fontsize=12, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))

    # ── Witness point ────────────────────────────────────────────────────
    px = py = 0.294118
    ax.plot(px, py, "o", color=C_NAVYBLUE, ms=9, zorder=5)
    ax.annotate("Witness point\n$x_1=x_2=0.2941$",
                xy=(px, py), xytext=(px-0.05, py+0.02),
                fontsize=14, color=C_NAVYBLUE)

    # ── PASS square: half-side = 0.285, full side = 0.570 ───────────────────
    hs_pass = 0.285
    ax.add_patch(mpatches.Rectangle(
        (px - hs_pass, py - hs_pass), 2 * hs_pass, 2 * hs_pass,
        facecolor=C_MAGENTA, alpha=0.14, linewidth=0, zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (px - hs_pass, py - hs_pass), 2 * hs_pass, 2 * hs_pass,
        facecolor="none", edgecolor=C_GREEN, lw=1.8, zorder=3))
    ax.text(px + hs_pass - 0.14, py + 0.11,
            f"side={2*hs_pass:.3f}\n  PASS ✓  →", fontsize=12,
            color=C_GREEN, va="center")

    # ── FAIL square: half-side = 0.286, full side = 0.572 ───────────────────
    hs_fail = 0.286
    ax.add_patch(mpatches.Rectangle(
        (px - hs_fail, py - hs_fail), 2 * hs_fail, 2 * hs_fail,
        facecolor="none", edgecolor=C_RED, lw=1.8, ls="--", zorder=3))
    ax.text(px + hs_fail + 0.01, py - 0.08,
            f"side={2*hs_fail:.3f}\n← FAIL ✗", fontsize=12,
            color=C_RED, va="center")

    # ── half-side annotation ──────────────────────────────────────────────────
    bd = R_constraint - px
    ax.annotate("", xy=(px + hs_fail + 0.005, py), xytext=(px, py),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(px + hs_fail / 2 - 0.03, py - 0.05, f"rad-abs = {hs_fail}",
            fontsize=14, color=C_RED, ha="center")

    handles = [
        Line2D([0], [0], color=C_BLUE, lw=1.5, ls="--",
               label=r"$|x|= √(x_1^2+x_2^2)$" + f"={R_constraint} - constraint boundary"),
        mpatches.Patch(facecolor=C_GREEN, alpha=0.65,
                       label=f"query == TRUE"),
        mpatches.Patch(facecolor=C_MAGENTA, alpha=0.35,
                       label=f"PASS square side=2*{hs_pass}={2*hs_pass:.3f}"),
        mpatches.Patch(facecolor="none", edgecolor=C_RED, lw=1.5,
                       linestyle="--", label=f"FAIL  square side=2*{hs_fail}={2*hs_fail:.3f}"),
        Line2D([0], [0], marker="o", color=C_NAVYBLUE, lw=0,
               markersize=8, label="Witness (0.2941, 0.2941)"),
    ]
    ax.legend(handles=handles, fontsize=11, loc="upper right", framealpha=0.85)
    ax.set_facecolor("#FAFAF8")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main(timeout: float = inf):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.subplots(1, 1)

    plot_geometry(ax1)
    #plot_pipeline(ax2)
    fig.tight_layout()

    fig.savefig("witness_certify.png",
                dpi=150, bbox_inches="tight")
    print("Saved witness_certify.png")
    
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=int(timeout)*1000, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()

if __name__ == "__main__":
    timeout = inf
    if len(argv) > 2:
        if '-timeout' == argv[1]:
            timeout = float(argv[2])
    main(timeout)
