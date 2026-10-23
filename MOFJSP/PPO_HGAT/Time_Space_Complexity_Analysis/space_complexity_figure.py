# -*- coding: utf-8 -*-
"""Space-complexity figure: peak inference memory and edge count.

Panel (a): peak tensor memory attributable to the graph (measured peak minus the smallest
scale) against the number of edges E, log-log.  Panel (b): measured edge count E against the
problem size N, together with the normalised N^2 A^2 / M reference from the derivation.

Usage:  python PPO_HGAT/Time_Space_Complexity_Analysis/space_complexity_figure.py
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Figure_And_File", "PPO_HGAT", "Time_Space_Complexity_Analysis")
SRC = os.path.join(OUT, "space_scaling_data.csv")
BLUE = "#2E6FBA"
GRAY = "#8C8C8C"


def load():
    rows = []
    with open(SRC, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) if key != "scale" else value for key, value in row.items()})
    # Average eligible machines per operation of the same instance, for the reference term.
    for row in rows:
        path = sorted(glob.glob(os.path.join(ROOT, "DataSet", "comparison_instances", "*%s*.txt"
                                             % row["scale"])))[0]
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        jobs, m = map(int, lines[0].split())
        eligible = []
        for line in lines[1:1 + jobs]:
            numbers = list(map(int, line.split()))
            idx = 1
            for _ in range(numbers[0]):
                k = numbers[idx]
                idx += 1 + 2 * k
                eligible.append(k)
        row["A"] = float(np.mean(eligible))
    return rows


def main():
    rows = load()
    n = np.array([row["N"] for row in rows])
    e = np.array([row["E"] for row in rows])
    peak = np.array([row["peak_MiB"] for row in rows])
    a = np.array([row["A"] for row in rows])
    m = np.array([row["M"] for row in rows])

    incremental = peak - peak[0]
    keep = incremental > 0
    slope, intercept = np.polyfit(np.log(e[keep]), np.log(incremental[keep]), 1)
    fit = np.exp(intercept) * e[keep] ** slope

    reference = n ** 2 * a ** 2 / m
    reference = reference / reference[0]

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

    ax1.loglog(e[keep], incremental[keep], "-o", color=BLUE, lw=2.0, ms=5.5)
    ax1.loglog(e[keep], fit, "--", color=GRAY, lw=1.6,
               label="Fit: $M_{peak}\\propto E^{%.2f}$" % slope)
    ax1.set_xlabel("Number of edges $E$")
    ax1.set_ylabel("Peak memory above smallest scale (MiB)")
    ax1.set_title("(a) Peak memory versus graph size")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, which="major", ls=":", lw=0.5, color="#DDDDDD")
    # Bottleneck: at the 75x25 scale the graph passes 10k edges and the peak memory starts
    # to climb steeply (35.6 -> 49.2 MiB, i.e. +38%, at the largest scale).
    knee = next(index for index, row in enumerate(rows) if row["scale"] == "75x25")
    ax1.axvline(e[knee], color="#C44E52", ls=(0, (4, 3)), lw=1.1, zorder=3)
    ax1.plot([e[knee]], [incremental[knee]], marker="o", ms=6.5, mfc="white",
             mec="#C44E52", mew=1.5, zorder=5)
    ax1.annotate("Bottleneck onset\n(+%.0f%% peak memory ahead)"
                 % (100.0 * (peak[-1] / peak[knee] - 1.0)),
                 xy=(e[knee], incremental[knee]), xytext=(-8, 12),
                 textcoords="offset points", ha="right", va="bottom",
                 color="#C44E52", fontsize=8)

    ax2.loglog(n, e, "-o", color=BLUE, lw=2.0, ms=5.5, label="Measured edges $E$")
    ax2.loglog(n, e[0] * reference, "--", color=GRAY, lw=1.6,
               label="Reference $N^2A^2/M$ (normalised)")
    ax2.set_xlabel("Total number of operations $N$")
    ax2.set_ylabel("Number of edges $E$")
    ax2.set_title("(b) Edge count versus problem size")
    ax2.legend(loc="upper left", frameon=False)
    ax2.grid(True, which="major", ls=":", lw=0.5, color="#DDDDDD")

    for ax in (ax1, ax2):
        ax.tick_params(direction="in", top=True, right=True, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.tight_layout()
    for suffix in ("png", "pdf", "eps", "svg"):
        path = os.path.join(OUT, "Space_Complexity_Analysis." + suffix)
        if suffix == "pdf":
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



