#!/usr/bin/env python3.11
"""
BNH Certification Visualizations
Produces two output files:
  1. bnh_geometry.png  – matplotlib geometry plot
  2. bnh_pipeline.svg  – pure-SVG flowchart (no external deps)
"""

import math
import textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sys import argv
from math import inf

# ── shared palette ───────────────────────────────────────────────────────────
C_BLUE   = "#378ADD"
C_GREEN  = "#1D9E75"
C_RED    = "#E24B4A"
C_AMBER  = "#BA7517"
C_GRAY   = "#888780"

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
    ax.set_xlabel("x₁", fontsize=11)
    ax.set_ylabel("x₂", fontsize=11)
    ax.set_title("BNH certification geometry\n"
                 r"$f_1(x)=4x_1^2+4x_2^2$,  query: $(f_1-0.692)^2<4$",
                 fontsize=11, pad=10)

    R_constraint = 0.580091
    theta = np.linspace(0, math.pi / 2, 300)
    ax.fill(R_constraint * np.cos(theta), R_constraint * np.sin(theta),
            color=C_BLUE, alpha=0.10, zorder=1)
    ax.plot(R_constraint * np.cos(theta), R_constraint * np.sin(theta),
            color=C_BLUE, lw=1.5, ls="--",
            label=f"|x| = {R_constraint}  (constraint boundary)", zorder=2)

    ang = math.radians(48)
    ax.annotate(f"|x| = {R_constraint}",
                xy=(R_constraint * math.cos(ang), R_constraint * math.sin(ang)),
                xytext=(0.62, 0.50), fontsize=8, color=C_BLUE,
                arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=0.7))

    px = py = 0.294118
    ax.plot(px, py, "o", color=C_AMBER, ms=8, zorder=5)
    ax.add_patch(plt.Circle((px, py), 0.012, fill=False,
                             edgecolor=C_AMBER, lw=1.2, zorder=4))
    ax.annotate("Pareto point\n$x_1=x_2=0.2941$",
                xy=(px, py), xytext=(px + 0.08, py + 0.12),
                fontsize=8, color=C_AMBER,
                arrowprops=dict(arrowstyle="-", color=C_AMBER, lw=0.7))

    ax.plot([0, px], [0, py], color=C_GRAY, lw=0.8, ls=":", zorder=3)
    ax.text(0.10, 0.05, "0.2941×√2≈0.416", fontsize=7.5, color=C_GRAY,
            rotation=45)

    R_pass = 0.285
    ax.add_patch(plt.Circle((px, py), R_pass, color=C_GREEN,
                             fill=True, alpha=0.14, linewidth=0, zorder=2))
    ax.add_patch(plt.Circle((px, py), R_pass, fill=False,
                             edgecolor=C_GREEN, lw=1.5, zorder=3))
    ax.text(px + R_pass + 0.01, py,
            "rad=0.285\n→ PASS ✓", fontsize=8, color=C_GREEN, va="center")

    R_fail = 0.286
    ax.add_patch(plt.Circle((px, py), R_fail, fill=False,
                             edgecolor=C_RED, lw=1.5, ls="--", zorder=3))
    ax.text(px + R_fail + 0.01, py - 0.06,
            "rad=0.286\n→ FAIL ✗", fontsize=8, color=C_RED, va="center")

    bd = R_constraint - px
    ax.annotate("", xy=(px + R_fail, py), xytext=(px, py),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1))
    ax.text(px + R_fail / 2, py - 0.03, "0.286",
            fontsize=7.5, color=C_RED, ha="center")

    ax.annotate("", xy=(R_constraint, 0), xytext=(px, 0),
                arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=0.8))
    ax.text((px + R_constraint) / 2, -0.035, f"Δ={bd:.4f}",
            fontsize=7.5, color=C_BLUE, ha="center")

    handles = [
        Line2D([0], [0], color=C_BLUE, lw=1.5, ls="--",
               label=f"|x|<sqrt((2+0.692042)/8)={R_constraint}  constraint boundary"),
        mpatches.Patch(facecolor=C_GREEN, alpha=0.35,
                       label=f"rad={R_pass}  PASS"),
        Line2D([0], [0], color=C_RED, lw=1.5, ls="--",
               label=f"rad={R_fail}  FAIL"),
        Line2D([0], [0], marker="o", color=C_AMBER, lw=0,
               markersize=7, label="Pareto point (0.2941, 0.2941)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.85)
    ax.set_facecolor("#FAFAF8")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Pipeline flowchart (pure SVG)
# ════════════════════════════════════════════════════════════════════════════

# ── SVG primitives ───────────────────────────────────────────────────────────
def _box(x, y, w, h, fill, stroke, title, subtitle=None,
         title_col="#ffffff", sub_col=None, rx=10):
    """Rounded rect with 1-2 lines of text, centred."""
    if sub_col is None:
        sub_col = title_col
    ty = y + h / 2 + (0 if subtitle is None else -9)
    sy = y + h / 2 + 11
    sub_el = (f'<text x="{x + w/2:.1f}" y="{sy:.1f}" '
              f'text-anchor="middle" font-size="11" fill="{sub_col}" '
              f'font-family="sans-serif">{subtitle}</text>'
              if subtitle else "")
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>\n'
        f'<text x="{x + w/2:.1f}" y="{ty:.1f}" text-anchor="middle" '
        f'font-size="13" font-weight="600" fill="{title_col}" '
        f'font-family="sans-serif">{title}</text>\n'
        + sub_el
    )


def _arrow(x1, y1, x2, y2, color="#5F5E5A"):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="1.4" '
            f'marker-end="url(#arr)"/>\n')


def _line(x1, y1, x2, y2, color="#888780"):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="1"/>\n')


def _text(x, y, msg, size=11, color="#5F5E5A", anchor="middle",
          weight="normal", style="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" font-family="sans-serif" fill="{color}" '
            f'font-weight="{weight}" font-style="{style}">{msg}</text>\n')


def _multiline_box(x, y, w, h, fill, stroke, lines,
                   title_col="#ffffff", sub_col=None, rx=10):
    """Box with arbitrary number of text lines, evenly spaced."""
    if sub_col is None:
        sub_col = title_col
    els = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>\n')
    n = len(lines)
    step = h / (n + 1)
    for i, (txt, sz, bold) in enumerate(lines):
        ty = y + step * (i + 1)
        col = title_col if i == 0 else (sub_col or title_col)
        fw = "600" if bold else "400"
        els += (f'<text x="{x + w/2:.1f}" y="{ty:.1f}" text-anchor="middle" '
                f'font-size="{sz}" font-weight="{fw}" fill="{col}" '
                f'font-family="sans-serif">{txt}</text>\n')
    return els


def build_pipeline_svg():
    W = 680          # canvas width
    BW = 340         # wide box
    MW = 260         # medium box
    BH = 52          # normal box height
    SH = 48          # slim box height
    TH = 78          # tall box (smlp)
    cx = W // 2      # 340
    xL = 155         # left branch centre
    xR = 525         # right branch centre
    bw2 = 240        # branch box width

    # vertical positions (top-left y of each row)
    y0  = 30         # title
    y1  = 55         # bnh_dataset
    y2  = 145        # gunzip
    y3  = 230        # fork label
    y4  = 310        # PASS / FAIL json boxes
    y5  = 410        # smlp box
    y6  = 545        # result boxes
    y7  = 640        # result file label

    # total canvas height
    H = 700

    arrow_color = "#5F5E5A"
    line_color  = "#888780"

    svg = textwrap.dedent(f"""\
        <svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
             viewBox="0 0 {W} {H}" font-family="sans-serif">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
                  stroke-width="1.5" stroke-linecap="round"
                  stroke-linejoin="round"/>
          </marker>
        </defs>
        <rect width="{W}" height="{H}" fill="#FAFAF8"/>
    """)

    # title
    svg += _text(cx, y0 + 16, "run_certify – script pipeline",
                 size=20, color="#2C2C2A", weight="600")

    # ── row 1: bnh_dataset.py ────────────────────────────────────────────
    svg += _box(cx - BW//2, y1, BW, BH,
                fill="#D3D1C7", stroke="#B4B2A9",
                title="bnh_dataset.py",
                subtitle="generate BNH training data → bnh.csv.gz",
                title_col="#2C2C2A", sub_col="#5F5E5A")
    svg += _arrow(cx, y1 + BH, cx, y2 - 4, arrow_color)

    # ── row 2: gunzip ────────────────────────────────────────────────────
    svg += _box(cx - MW//2, y2, MW, SH,
                fill="#D3D1C7", stroke="#B4B2A9",
                title="gunzip  →  BNH.csv",
                title_col="#2C2C2A")
    svg += _arrow(cx, y2 + SH, cx, y3 + 2, arrow_color)

    # ── fork label ───────────────────────────────────────────────────────
    svg += _text(cx, y3 + 18, "for each json spec",
                 size=11, color=line_color, style="italic")

    # fork lines
    fork_y = y3 + 30
    svg += _line(xL, fork_y, xR, fork_y, line_color)
    svg += _line(xL, fork_y, xL, y4 - 4, line_color)
    svg += _line(xR, fork_y, xR, y4 - 4, line_color)
    svg += _arrow(xL, y4 - 4, xL, y4, arrow_color)
    svg += _arrow(xR, y4 - 4, xR, y4, arrow_color)

    # ── row 3: PASS / FAIL json boxes ───────────────────────────────────
    svg += _box(xL - bw2//2, y4, bw2, BH,
                fill="#9FE1CB", stroke="#5DCAA5",
                title="bnh_certify_pass.json",
                subtitle="rad-abs = 0.285",
                title_col="#085041", sub_col="#0F6E56")
    svg += _box(xR - bw2//2, y4, bw2, BH,
                fill="#F5C4B3", stroke="#F0997B",
                title="bnh_certify_fail.json",
                subtitle="rad-abs = 0.286",
                title_col="#4A1B0C", sub_col="#712B13")

    # join lines into smlp
    join_y = y5 - 14
    svg += _line(xL, y4 + BH, xL, join_y, line_color)
    svg += _line(xR, y4 + BH, xR, join_y, line_color)
    svg += _line(xL, join_y, xR, join_y, line_color)
    svg += _arrow(cx, join_y, cx, y5, arrow_color)

    # ── row 4: smlp certify ──────────────────────────────────────────────
    svg += _multiline_box(
        cx - BW//2 - 50, y5, BW + 100, TH,
        fill="#AFA9EC", stroke="#7F77DD",
        lines=[
            ("smlp  -mode certify",                              13, True),
            ("-model poly_sklearn   -resp F1,F2   -feat X1,X2", 11, False),
            ("-quer_exprs \"(F1−0.692)²&lt;4\"   -epsilon 0.000005   -delta_rel 0.05",
             10, False),
        ],
        title_col="#26215C", sub_col="#3C3489",
    )
    svg += _arrow(cx, y5 + TH, cx, y6 - 4, arrow_color)

    # ── row 5: result boxes ──────────────────────────────────────────────
    svg += _box(xL - bw2//2, y6, bw2, BH,
                fill="#9FE1CB", stroke="#5DCAA5",
                title="rad = 0.285",
                subtitle="witness_status  →  PASS ✓",
                title_col="#085041", sub_col="#0F6E56")
    svg += _box(xR - bw2//2, y6, bw2, BH,
                fill="#F5C4B3", stroke="#F0997B",
                title="rad = 0.286",
                subtitle="witness_status  →  FAIL ✗",
                title_col="#4A1B0C", sub_col="#712B13")
    svg += _text(cx, y6 + BH//2 + 4, "vs",
                 size=11, color=line_color, style="italic")

    # ── row 6: result file ───────────────────────────────────────────────
    mid_y = max(y6 + BH, y6 + BH)
    svg += _arrow(cx, mid_y, cx, y7 - 4, arrow_color)
    svg += _text(cx, y7 + 4,
                 "BNH_BNH_certify_results.json  →  .query1.witness_status  (jq)",
                 size=12, color=line_color, style="italic")
    svg += "</svg>\n"
    return svg

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Pipeline flowchart (matplotlib)
# ════════════════════════════════════════════════════════════════════════════
def _mpl_box(ax, cx, cy, w, h, title, subtitle=None,
             facecolor="#D3D1C7", textcolor="#2C2C2A", subcolor=None, rx=0.04):
    from matplotlib.patches import FancyBboxPatch
    if subcolor is None:
        subcolor = textcolor
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle=f"round,pad={rx}",
                         facecolor=facecolor, edgecolor="white",
                         linewidth=0.8, zorder=3)
    ax.add_patch(box)
    ty = cy if subtitle is None else cy + h * 0.13
    ax.text(cx, ty, title, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=textcolor, zorder=4)
    if subtitle:
        ax.text(cx, cy - h * 0.18, subtitle, ha="center", va="center",
                fontsize=7.5, color=subcolor, zorder=4)


def _mpl_arrow(ax, x1, y1, x2, y2, color="#5F5E5A"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.2, mutation_scale=10), zorder=2)


def plot_pipeline(ax):
    ax.set_xlim(0, 10)
    # ylim set after drawing to align with geometry axes
    ax.axis("off")
    ax.set_facecolor("#FAFAF8")
    ax.set_title("run_certify – script pipeline", fontsize=11, pad=10)

    cx   = 5.0
    BW, BH = 4.2, 0.55
    MW, MH = 3.0, 0.50
    TW, TH = 8.5, 0.85
    bw,  bh = 3.8, 0.55
    xL, xR  = 2.55, 7.45

    # 1 – bnh_dataset.py
    y1 = 10.8
    y_top = y1 + BH/2 + 0.3  # top of content + small margin
    _mpl_box(ax, cx, y1, BW, BH,
             "bnh_dataset.py",
             "generate BNH training data -> bnh.csv.gz",
             facecolor="#D3D1C7", textcolor="#2C2C2A", subcolor="#5F5E5A")
    _mpl_arrow(ax, cx, y1 - BH/2, cx, y1 - BH/2 - 0.28)

    # 2 – gunzip
    y2 = y1 - BH/2 - 0.28 - MH/2 - 0.02
    _mpl_box(ax, cx, y2, MW, MH,
             "gunzip  ->  BNH.csv",
             facecolor="#D3D1C7", textcolor="#2C2C2A")
    _mpl_arrow(ax, cx, y2 - MH/2, cx, y2 - MH/2 - 0.15)

    # fork label
    y_label = y2 - MH/2 - 0.15 - 0.18
    ax.text(cx, y_label, "for each json spec",
            ha="center", va="center", fontsize=8,
            color=C_GRAY, style="italic")

    # fork lines
    y_fork = y_label - 0.22
    ax.plot([xL, xR], [y_fork, y_fork], color=C_GRAY, lw=0.9)
    ax.plot([xL, xL], [y_fork, y_fork - 0.04], color=C_GRAY, lw=0.9)
    ax.plot([xR, xR], [y_fork, y_fork - 0.04], color=C_GRAY, lw=0.9)

    # 3 – PASS / FAIL json boxes
    y3 = y_fork - 0.04 - bh/2 - 0.28
    _mpl_arrow(ax, xL, y_fork - 0.04, xL, y3 + bh/2)
    _mpl_arrow(ax, xR, y_fork - 0.04, xR, y3 + bh/2)
    _mpl_box(ax, xL, y3, bw, bh,
             "bnh_certify_pass.json", "rad-abs = 0.285",
             facecolor="#9FE1CB", textcolor="#085041", subcolor="#0F6E56")
    _mpl_box(ax, xR, y3, bw, bh,
             "bnh_certify_fail.json", "rad-abs = 0.286",
             facecolor="#F5C4B3", textcolor="#4A1B0C", subcolor="#712B13")

    # join lines into smlp
    y_join = y3 - bh/2 - 0.22
    ax.plot([xL, xL], [y3 - bh/2, y_join], color=C_GRAY, lw=0.9)
    ax.plot([xR, xR], [y3 - bh/2, y_join], color=C_GRAY, lw=0.9)
    ax.plot([xL, xR], [y_join,    y_join],  color=C_GRAY, lw=0.9)
    _mpl_arrow(ax, cx, y_join, cx, y_join - 0.04)

    # 4 – smlp certify (tall, 3-line)
    y4 = y_join - 0.04 - TH/2 - 0.28
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((cx - TW/2, y4 - TH/2), TW, TH,
                         boxstyle="round,pad=0.04",
                         facecolor="#AFA9EC", edgecolor="white",
                         linewidth=0.8, zorder=3)
    ax.add_patch(box)
    ax.text(cx, y4 + TH*0.22, "smlp  -mode certify",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#26215C", zorder=4)
    ax.text(cx, y4 - TH*0.05,
            "-model poly_sklearn   -resp F1,F2   -feat X1,X2",
            ha="center", va="center", fontsize=7.5, color="#3C3489", zorder=4)
    ax.text(cx, y4 - TH*0.30,
            "-quer_exprs \"(F1-0.692)^2<4\"   -epsilon 0.000005   -delta_rel 0.05",
            ha="center", va="center", fontsize=7, color="#3C3489", zorder=4)
    _mpl_arrow(ax, cx, y4 - TH/2, cx, y4 - TH/2 - 0.28)

    # 5 – result boxes
    y5 = y4 - TH/2 - 0.28 - bh/2 - 0.02
    _mpl_box(ax, xL, y5, bw, bh,
             "rad = 0.285", "witness_status  ->  PASS",
             facecolor="#9FE1CB", textcolor="#085041", subcolor="#0F6E56")
    _mpl_box(ax, xR, y5, bw, bh,
             "rad = 0.286", "witness_status  ->  FAIL",
             facecolor="#F5C4B3", textcolor="#4A1B0C", subcolor="#712B13")
    ax.text(cx, y5, "vs", ha="center", va="center",
            fontsize=8, color=C_GRAY, style="italic")

    # 6 – result file
    y6 = y5 - bh/2 - 0.32
    _mpl_arrow(ax, cx, y5 - bh/2, cx, y6 + 0.12)
    ax.text(cx, y6,
            "BNH_BNH_certify_results.json  ->  .query1.witness_status  (jq)",
            ha="center", va="center", fontsize=7.5,
            color=C_GRAY, style="italic")


    ax.set_ylim(0, y_top)

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(16, 10))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.set_position([0.1, 0, 0.8, 0]) # [left, bottom, width, height]
    ax2.set_position([0.1, 0, 0.8, 0]) # [left, bottom, width, height]

    plot_geometry(ax1)
    plot_pipeline(ax2)
    fig.tight_layout()

    fig.savefig("bnh_certify.png", dpi=600, bbox_inches="tight")
    print("Saved bnh_certify.png")

    svg_text = build_pipeline_svg()
    with open("bnh_pipeline.svg", "w", encoding="utf-8") as f:
        f.write(svg_text)
    print("Saved bnh_pipeline.svg")

    timeout = inf
    if len(argv) > 2:
        if '-timeout' == argv[1]:
            timeout = int(argv[2]) 
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout*1000, callbacks=[(plt.close, [], {})])
        timer.start()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
