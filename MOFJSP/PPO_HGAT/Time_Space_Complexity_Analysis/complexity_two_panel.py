# -*- coding: utf-8 -*-
"""Two-panel complexity figure: growth versus the cubic bound, and per-step cost.

The Origin automation of this build does not deliver a plot into a second layer (the command
reports success but the layer stays empty), so this composite is produced directly as a vector
figure.  The style matches the single-panel drafts: medium blue, Times New Roman, no chart
junk, legends without frames.

Usage:  python PPO_HGAT/Time_Space_Complexity_Analysis/complexity_two_panel.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Figure_And_File", "PPO_HGAT", "Time_Space_Complexity_Analysis")
VAR = OUT
SRC = os.path.join(OUT, "runtime_scaling_data.csv")
BLUE = "#2E6FBA"
GRAY = "#8C8C8C"


def load():
    rows = []
    with open(SRC, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n = float(row["N"])
            total = float(row["time_ms"]) / 1000.0
            rows.append((row["scale"], n, total, float(row["time_ms"]) / n))
    return rows


def main():
    rows = load()
    scales = [row[0] for row in rows]
    n = np.array([row[1] for row in rows])
    total = np.array([row[2] for row in rows])
    step = np.array([row[3] for row in rows])
    rel = total / total[0]
    cubic = (n / n[0]) ** 3

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.15), dpi=300)

    ax1.loglog(n, rel, "-o", color=BLUE, lw=2.0, ms=5.5, mfc=BLUE, mec=BLUE,
               label="Measured (fit $N^{1.36}$)")
    ax1.loglog(n, cubic, "--", color=GRAY, lw=1.6,
               label="Cubic reference $(N/N_0)^3$")
    ax1.set_xlabel("Total number of operations $N$")
    ax1.set_ylabel("Relative inference time")
    ax1.set_title("(a) Growth versus the cubic bound")
    ax1.set_xlim(50, 700)
    ax1.set_ylim(0.8, 3500)
    ax1.legend(loc="upper left", frameon=False)
    ax1.annotate("%.1f$\\times$" % rel[-1], xy=(n[-1], rel[-1]), xytext=(-8, 8),
                 textcoords="offset points", ha="right", color=BLUE, fontsize=9)
    ax1.annotate("%.0f$\\times$" % cubic[-1], xy=(n[-1], cubic[-1]), xytext=(-8, 8),
                 textcoords="offset points", ha="right", va="bottom",
                 color=GRAY, fontsize=9)
    ax1.grid(True, which="major", ls=":", lw=0.5, color="#DDDDDD")

    ax2.semilogx(n, step, "-o", color=BLUE, lw=2.0, ms=5.5, mfc=BLUE, mec=BLUE)
    ax2.set_xlabel("Total number of operations $N$")
    ax2.set_ylabel("Time per decision step (ms)")
    ax2.set_title("(b) Cost of one decision step")
    ax2.set_xlim(50, 700)
    ax2.set_ylim(6, 20)
    ax2.annotate("%.1f $\\rightarrow$ %.1f ms" % (step[0], step[-1]),
                 xy=(n[0], step[0]), xytext=(10, 22), textcoords="offset points",
                 color=BLUE, fontsize=9)
    # Bottleneck onset: at the 50x25 scale the per-step cost starts to climb steeply
    # (12.3 ms here, 18.7 ms at the largest scale, i.e. +52%).
    knee = scales.index("50x25")
    ax2.axvline(n[knee], color="#C44E52", ls=(0, (4, 3)), lw=1.1, zorder=3)
    ax2.plot([n[knee]], [step[knee]], marker="o", ms=6.5, mfc="white", mec="#C44E52",
             mew=1.5, zorder=5)
    ax2.annotate("Bottleneck onset\n(+52% per step ahead)",
                 xy=(n[knee], step[knee]), xytext=(-8, 12), textcoords="offset points",
                 ha="right", va="bottom", color="#C44E52", fontsize=8)
    ax2.grid(True, which="major", ls=":", lw=0.5, color="#DDDDDD")

    for ax in (ax1, ax2):
        ax.tick_params(direction="in", top=True, right=True, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    fig.tight_layout()

    os.makedirs(VAR, exist_ok=True)
    for suffix in ("png", "pdf", "eps", "svg"):
        path = os.path.join(VAR, "Time_Complexity_Analysis." + suffix)
        if suffix == "pdf":
            # A preview process often holds the PDF; write a sibling first and swap.
            pending = os.path.splitext(path)[0] + "_pending.pdf"
            fig.savefig(pending, dpi=300, bbox_inches="tight")
            try:
                if os.path.exists(path):
                    os.remove(path)
                os.replace(pending, path)
                print("pdf: replaced")
            except OSError as error:
                print("pdf locked (%s); kept %s" % (error, pending))
            continue
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("%s: %s" % (suffix, "written" if os.path.exists(path) else "missing"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())


