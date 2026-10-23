# -*- coding: utf-8 -*-
"""Split the public-benchmark results into the three comparison protocols.

Experiment 1 compares the five dispatching rules with the deployed hybrid
decoder.  Experiment 2 compares the two metaheuristics (GA and TS) with the
pure proposed policy at an equal evaluation budget.  Experiment 3 compares the
learning baselines (MLP+DQN, MLP+PPO, GAT+PPO, HGNN, DAN, RESCHED) with the
pure proposed policy.  The hybrid decoder and the pure policy are reported in
different protocols and are never mixed inside one comparison.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import public_benchmark as pb

METHODS_E1 = list(pb.RULES) + ["Proposed-hybrid"]
METHODS_E3 = ["MLP+DQN", "MLP+PPO", "GAT+PPO", "HGNN", "DAN", "RESCHED",
              "Proposed-policy", "Proposed-policy-refined"]
METHODS_E2_ALL = METHODS_E3


def _root() -> Path:
    return HERE.parent.parent


def _data_dir() -> Path:
    return _root() / "Public_DataSet" / "PublicBenchmarkValidation"


def _figure_dir() -> Path:
    return _root() / "Figure_And_File" / "PublicBenchmarkValidation"


def _num(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_rows(family: Optional[str] = None) -> List[dict]:
    result_root = _data_dir() / "results"
    if family:
        path = result_root / family / "per_instance.csv"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw = list(csv.DictReader(handle))
    else:
        paths = [result_root / name / "per_instance.csv"
                 for name in ("Brandimarte", "Lawrence")
                 if (result_root / name / "per_instance.csv").exists()]
        if not paths:
            return []
        raw = []
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                raw.extend(csv.DictReader(handle))
    rows = []
    for source in raw:
        row = dict(source)
        for key in ("jobs", "machines", "operations"):
            row[key] = int(float(row[key]))
        row["bks"] = _num(row["bks"]) if row.get("bks") not in (None, "", "nan") else None
        for method in METHODS_E2_ALL + ["Proposed-hybrid", "Proposed-policy-refined",
                                        "GA", "TS"] + list(pb.RULES):
            if method in row:
                row[method] = _num(row[method])
                row["gap_%s" % method] = _num(row.get("gap_%s" % method))
        for key in ("best_rule", "ratio_hybrid_best_rule", "ratio_policy_best_rule"):
            if key in row:
                row[key] = _num(row[key])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: Sequence[dict], proposed: str, competitors: Sequence[str]):
    out = {"Win": 0, "Tie": 0, "Loss": 0}
    detail = []
    for row in rows:
        best = min(row[method] for method in competitors)
        value = row[proposed]
        if value < best - 1e-9:
            label = "Win"
        elif value > best + 1e-9:
            label = "Loss"
        else:
            label = "Tie"
        out[label] += 1
        detail.append({
            "name": row["name"],
            "family": row["family"],
            "proposed": value,
            "best_competitor": best,
            "outcome": label,
            "relative": value / best if best else float("nan"),
        })
    return out, detail


def _stats(rows: Sequence[dict], methods: Sequence[str], reference: str):
    overall = pb.friedman_summary(rows, methods)
    pairwise = pb.pairwise_summary(rows, reference, methods)
    families = {}
    for family in ("Brandimarte", "Lawrence"):
        subset = [row for row in rows if row["family"] == family]
        if not subset:
            continue
        families[family] = {
            "n": len(subset),
            "friedman": pb.friedman_summary(subset, methods),
            "pairwise": pb.pairwise_summary(subset, reference, methods),
        }
    return overall, pairwise, families


def _summary(rows: Sequence[dict], methods: Sequence[str]) -> List[dict]:
    out = []
    for method in methods:
        values = np.asarray([row[method] for row in rows], dtype=float)
        gaps = np.asarray([row["gap_%s" % method] for row in rows], dtype=float)
        valid = gaps[~np.isnan(gaps)]
        out.append({
            "method": method,
            "n": int(values.size),
            "mean_makespan": float(values.mean()),
            "median_makespan": float(np.median(values)),
            "mean_gap_percent": float(valid.mean()) if valid.size else float("nan"),
            "median_gap_percent": float(np.median(valid)) if valid.size else float("nan"),
            "bks_count": int(valid.size),
        })
    return out


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _save(fig, folder: Path, stem: str):
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".eps", ".svg"):
        fig.savefig(folder / (stem + suffix), dpi=300, bbox_inches="tight",
                    facecolor="white")


def _plot_gap(rows, methods, folder, stem, title, single_panel=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "SPT": "#8c8c8c", "LBD": "#7f9c90",
        "FIFO": "#9a8fb5", "MOPNR": "#6f8faf", "MWKR": "#b07a7a",
        "MLP+DQN": "#b17f59", "MLP+PPO": "#7c9a92", "GAT+PPO": "#7b86a8",
        "HGNN": "#4c7a6f", "DAN": "#876f9f", "RESCHED": "#6b7f9e",
        "GA": "#8c6d4f", "TS": "#5f7d95",
        "Proposed-policy": "#1f4e79", "Proposed-policy-refined": "#c55a11",
        "Proposed-hybrid": "#c55a11",
    }
    labels = {"Proposed-policy": "HGAT+PPO (policy)",
              "Proposed-policy-refined": "HGAT+PPO (+refine)",
              "Proposed-hybrid": "HGAT+PPO (hybrid)"}
    families = [name for name in ("Brandimarte", "Lawrence")
                if any(row["family"] == name for row in rows)]
    if single_panel:
        families = ["All"]
    if not families:
        return
    if len(families) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5.0, 3.4))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, len(families), figsize=(4.3 * len(families), 3.4),
                                 sharey=True)
    for ax, family in zip(axes, families):
        subset = [row for row in rows
                  if (family == "All" or row["family"] == family) and row.get("bks")]
        data = [[row["gap_%s" % m] for row in subset
                 if not math.isnan(row["gap_%s" % m])] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False,
                        medianprops={"linewidth": 1.2, "color": "#333333"},
                        whiskerprops={"linewidth": 0.8},
                        capprops={"linewidth": 0.8})
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(colors.get(method, "#8c8c8c"))
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.78)
        ax.axhline(0, color="#c55a11", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([labels.get(m, m) for m in methods], rotation=35, ha="right")
        ax.set_title("%s (n=%d)" % ("All held-out instances" if family == "All" else family,
                                    len(subset)), pad=6)
        _style(ax)
    axes[0].set_ylabel("Gap to BKS (%)")
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, folder, stem)
    plt.close(fig)


def _plot_ranks(overall, methods, folder, stem, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "SPT": "#8c8c8c", "LBD": "#7f9c90",
        "FIFO": "#9a8fb5", "MOPNR": "#6f8faf", "MWKR": "#b07a7a",
        "MLP+DQN": "#b17f59", "MLP+PPO": "#7c9a92", "GAT+PPO": "#7b86a8",
        "HGNN": "#4c7a6f", "DAN": "#876f9f", "RESCHED": "#6b7f9e",
        "GA": "#8c6d4f", "TS": "#5f7d95",
        "Proposed-policy": "#1f4e79", "Proposed-policy-refined": "#c55a11",
        "Proposed-hybrid": "#c55a11",
    }
    labels = {"Proposed-policy": "HGAT+PPO (policy)",
              "Proposed-policy-refined": "HGAT+PPO (+refine)",
              "Proposed-hybrid": "HGAT+PPO (hybrid)"}
    if not overall.get("ranks"):
        return
    ordered = sorted(overall["ranks"], key=lambda m: overall["ranks"][m])
    values = [overall["ranks"][m] for m in ordered]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.barh(range(len(ordered)), values,
                   color=[colors.get(m, "#8c8c8c") for m in ordered],
                   edgecolor="#333333", linewidth=0.6, height=0.62)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([labels.get(m, m) for m in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Average Friedman rank (lower is better)")
    ax.set_title(title, pad=6)
    for bar, value in zip(bars, values):
        ax.text(value + 0.04, bar.get_y() + bar.get_height() / 2,
                "%.2f" % value, va="center", fontsize=7.5)
    _style(ax)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, folder, stem)
    plt.close(fig)


def _plot_wtl(counts, folder, stem, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    categories = ["Win", "Tie", "Loss"]
    values = [counts.get(c, 0) for c in categories]
    bars = ax.bar(categories, values,
                  color=["#2f6b5e", "#b9b9b9", "#b07a7a"],
                  edgecolor="#333333", linewidth=0.6, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1,
                str(value), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of instances")
    ax.set_title(title, pad=6)
    _style(ax)
    fig.tight_layout()
    _save(fig, folder, stem)
    plt.close(fig)


def _plot_ratio(detail, folder, stem, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detail = sorted(detail, key=lambda row: (row["family"], row["name"]))
    x = np.arange(len(detail))
    y = [row["relative"] for row in detail]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(x, y, color="#c55a11", marker="o", markersize=2.6, linewidth=1.1)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([row["name"] for row in detail], rotation=90, fontsize=6.5)
    ax.set_ylabel("Proposed / best competitor")
    ax.set_title(title, pad=6)
    _style(ax)
    fig.tight_layout()
    _save(fig, folder, stem)
    plt.close(fig)


def run_experiment(rows, methods, reference, competitors, name, title, folder,
                   figure_folder: Optional[Path] = None, single_panel: bool = False):
    overall, pairwise, families = _stats(rows, methods, reference)
    summary = _summary(rows, methods)
    counts, detail = _counts(rows, reference, competitors)
    out = folder / name
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "method_summary.csv", summary,
               ["method", "n", "mean_makespan", "median_makespan",
                "mean_gap_percent", "median_gap_percent", "bks_count"])
    pair_rows = [dict(row, comparison="overall") for row in pairwise]
    for family, blob in families.items():
        for row in blob["pairwise"]:
            pair_rows.append(dict(row, comparison=family))
    family_rows = []
    for family in ("Brandimarte", "Lawrence"):
        subset = [row for row in rows if row["family"] == family]
        if not subset:
            continue
        for entry in _summary(subset, methods):
            entry = dict(entry)
            entry["family"] = family
            family_rows.append(entry)
    if family_rows:
        _write_csv(out / "family_summary.csv", family_rows,
                   ["family", "method", "n", "mean_makespan", "median_makespan",
                    "mean_gap_percent", "median_gap_percent", "bks_count"])
    _write_csv(out / "pairwise_tests.csv", pair_rows,
               ["comparison", "reference", "method", "n", "wilcoxon_statistic",
                "p_value", "p_holm", "cliffs_delta", "effect"])
    _write_csv(out / "friedman_tests.csv", [{
        "scope": "overall",
        "n": overall.get("n"),
        "statistic": overall.get("statistic"),
        "p": overall.get("p"),
        "iman_davenport_F": overall.get("iman_davenport_F"),
        "iman_davenport_p": overall.get("iman_davenport_p"),
        "ranks": json.dumps(overall.get("ranks", {})),
    }] + [{
        "scope": family,
        "n": blob["friedman"].get("n"),
        "statistic": blob["friedman"].get("statistic"),
        "p": blob["friedman"].get("p"),
        "iman_davenport_F": blob["friedman"].get("iman_davenport_F"),
        "iman_davenport_p": blob["friedman"].get("iman_davenport_p"),
        "ranks": json.dumps(blob["friedman"].get("ranks", {})),
    } for family, blob in families.items()],
        ["scope", "n", "statistic", "p", "iman_davenport_F",
         "iman_davenport_p", "ranks"])
    _write_csv(out / "win_tie_loss.csv", [{"outcome": k, "count": v}
                                          for k, v in counts.items()],
               ["outcome", "count"])
    _write_csv(out / "per_instance_outcome.csv", detail,
               ["name", "family", "proposed", "best_competitor", "outcome", "relative"])
    (out / "summary.json").write_text(json.dumps({
        "experiment": name,
        "protocol": title,
        "methods": methods,
        "reference": reference,
        "overall": overall,
        "families": families,
        "win_tie_loss": counts,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    figure_dir = (figure_folder / name) if figure_folder is not None else (_figure_dir() / name)
    _plot_gap(rows, methods, figure_dir, "Fig_%s_GapByFamily" % name, title,
              single_panel=single_panel)
    _plot_ranks(overall, methods, figure_dir, "Fig_%s_AverageRanks" % name,
                "%s: average ranks" % name.replace("_", " "))
    _plot_wtl(counts, figure_dir, "Fig_%s_WinTieLoss" % name,
              "%s: proposed vs best competitor" % name.replace("_", " "))
    _plot_ratio(detail, figure_dir, "Fig_%s_PerInstanceRatio" % name,
                "%s: per-instance ratio" % name.replace("_", " "))
    return {"summary": summary, "overall": overall, "families": families,
            "win_tie_loss": counts}


def main():
    for family in ("Brandimarte", "Lawrence"):
        rows = load_rows(family)
        if not rows:
            print("skip %s: no per-instance results" % family)
            continue
        result_root = _data_dir() / "results" / family
        figure_root = _figure_dir() / family
        report = {}
        report["Experiment1"] = run_experiment(
            rows, METHODS_E1, "Proposed-hybrid", list(pb.RULES),
            "Experiment1", "Public Experiment 1: dispatching-rule comparison",
            result_root, figure_root)
        if all(method in rows[0] for method in METHODS_E3):
            report["Experiment3"] = run_experiment(
                rows, METHODS_E3, "Proposed-policy",
                ["MLP+DQN", "MLP+PPO", "GAT+PPO", "HGNN", "DAN", "RESCHED"],
                "Experiment3", "Public Experiment 3: learning-baseline comparison",
                result_root, figure_root)
        else:
            print("skip %s Experiment3: stale results lack the expanded baseline columns" % family)
        (result_root / "public_experiment_split.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("%s Experiment1 methods: %s" % (family, METHODS_E1))
        print("%s Experiment3 methods: %s" % (family, METHODS_E3))
        print("%s results: %s" % (family, result_root))
        print("%s figures: %s" % (family, figure_root))


if __name__ == "__main__":
    main()



