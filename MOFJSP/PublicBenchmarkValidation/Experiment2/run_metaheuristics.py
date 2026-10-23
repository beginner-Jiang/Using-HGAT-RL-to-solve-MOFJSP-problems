# -*- coding: utf-8 -*-
"""Run the public metaheuristic experiment (GA and TS).

This is public Experiment 2.  It uses the family-local test split prepared by
``common/public_benchmark.py`` and does not call the hybrid rule decoder of the
main rule-comparison protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

import public_benchmark as pb
import metaheuristics as mh


def _family_test_sets(instances: Sequence[pb.BenchInstance], family: str):
    if family == "Combined":
        return {"Combined": [item for item in instances if item.split == "test"]}
    if family in ("Brandimarte", "Lawrence"):
        return {family: [item for item in instances
                         if item.family == family and item.split == "test"]}
    return {
        name: [item for item in instances
               if item.family == name and item.split == "test"]
        for name in ("Brandimarte", "Lawrence")
    }


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: Sequence[dict], methods: Sequence[str]) -> List[dict]:
    out = []
    for family in sorted(set(row["family"] for row in rows)):
        subset = [row for row in rows if row["family"] == family]
        for method in methods:
            values = np.asarray([row[method] for row in subset], dtype=float)
            gaps = np.asarray([row["gap_%s" % method] for row in subset], dtype=float)
            gaps = gaps[~np.isnan(gaps)]
            out.append({
                "family": family,
                "method": method,
                "n": int(values.size),
                "mean_makespan": float(values.mean()),
                "median_makespan": float(np.median(values)),
                "mean_gap_percent": float(gaps.mean()) if gaps.size else float("nan"),
                "median_gap_percent": float(np.median(gaps)) if gaps.size else float("nan"),
                "bks_count": int(gaps.size),
            })
    return out


def _statistics(rows: Sequence[dict], methods: Sequence[str]) -> dict:
    result = {"methods": list(methods), "families": {}}
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception:
        result["note"] = "scipy unavailable; descriptive statistics only"
        return result
    for family in sorted(set(row["family"] for row in rows)):
        subset = [row for row in rows if row["family"] == family]
        arrays = [[float(row[method]) for row in subset] for method in methods]
        if len(subset) >= 3 and len(methods) >= 3:
            stat, p = friedmanchisquare(*arrays)
            result["families"][family] = {
                "n": len(subset),
                "friedman_statistic": float(stat),
                "friedman_p": float(p),
                "wilcoxon_ga_vs_ts": _wilcoxon_p(arrays[0], arrays[1]),
                "wilcoxon_ga_vs_proposed": _wilcoxon_p(arrays[0], arrays[2])
                if len(arrays) >= 3 else None,
                "wilcoxon_ts_vs_proposed": _wilcoxon_p(arrays[1], arrays[2])
                if len(arrays) >= 3 else None,
            }
        elif len(subset) >= 3:
            result["families"][family] = {
                "n": len(subset),
                "wilcoxon_ga_vs_ts": _wilcoxon_p(arrays[0], arrays[1]),
            }
    return result


def _wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    if len(a) < 3 or np.allclose(a, b):
        return None
    return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def _attach_proposed(rows: List[dict], code_data: Path, family: str):
    """Reuse the pure-policy column of the formal learning evaluation."""
    path = code_data / "results" / family / "per_instance.csv"
    if not path.exists():
        return rows, False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = {row["name"]: row.get("Proposed-policy-refined") for row in csv.DictReader(handle)}
    present = False
    for row in rows:
        raw = values.get(row["name"])
        value = float(raw) if raw not in (None, "", "nan") else float("nan")
        row["Proposed-policy-refined"] = value
        if row.get("bks") and math.isfinite(value):
            row["gap_Proposed-policy-refined"] = (value - row["bks"]) / row["bks"] * 100.0
        else:
            row["gap_Proposed-policy-refined"] = float("nan")
        present = present or math.isfinite(value)
    return rows, present




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=pb.project_root_from_script())
    parser.add_argument("--family", choices=("Combined", "Brandimarte", "Lawrence", "all"),
                        default="Combined")
    parser.add_argument("--evaluations", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    families = _family_test_sets(instances, args.family)
    all_rows: List[dict] = []
    for family, family_instances in families.items():
        if not family_instances:
            print("warning: no test instances for %s; run the prepare stage first" % family,
                  flush=True)
            continue
        print("== %s: %d test instances ==" % (family, len(family_instances)), flush=True)
        rows = mh.evaluate_instances(
            family_instances, evaluations=args.evaluations, seed=args.seed,
            methods=("GA", "TS"),
        )
        rows, has_proposed = _attach_proposed(rows, code_data, family)
        methods = ("GA", "TS", "Proposed-policy-refined") if has_proposed else ("GA", "TS")
        all_rows.extend(rows)
        fields = ["name", "family", "jobs", "machines", "operations", "bks",
                  "GA", "TS", "gap_GA", "gap_TS"]
        if has_proposed:
            fields += ["Proposed-policy-refined", "gap_Proposed-policy-refined"]
        out = code_data / "results" / family / "Experiment2"
        _write_csv(out / "per_instance.csv", rows, fields)
        summary = _summary(rows, methods)
        _write_csv(out / "method_summary.csv", summary,
                   ["family", "method", "n", "mean_makespan", "median_makespan",
                    "mean_gap_percent", "median_gap_percent", "bks_count"])
        stats = _statistics(rows, methods)
        (out / "statistics.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print("%s results written to %s" % (family, out), flush=True)
    if not all_rows:
        raise SystemExit("no public test instances found")


if __name__ == "__main__":
    main()
