# -*- coding: utf-8 -*-
"""Regenerate the public Experiment 2 (GA/TS) tables and figures."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
for path in (COMMON, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_public_experiments as analysis


def _rows(family: str):
    path = (analysis._data_dir() / "results" / family / "Experiment2"
            / "per_instance.csv")
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["jobs"] = int(float(row["jobs"]))
        row["machines"] = int(float(row["machines"]))
        row["operations"] = int(float(row["operations"]))
        row["bks"] = float(row["bks"]) if row.get("bks") not in (None, "", "nan") else None
        for method in ("GA", "TS", "Proposed-policy-refined"):
            if method in row:
                row[method] = float(row[method])
                row["gap_%s" % method] = float(row["gap_%s" % method])
    return rows


def main() -> None:
    for family in ("Combined",):
        rows = _rows(family)
        if not rows or "Proposed-policy-refined" not in rows[0]:
            print("skip %s: run the formal evaluation first" % family)
            continue
        result = analysis.run_experiment(
            rows, ("GA", "TS", "Proposed-policy-refined"), "Proposed-policy-refined",
            ("GA", "TS"), "Experiment2",
            "Public Experiment 2: metaheuristic comparison at an equal evaluation budget",
            analysis._data_dir() / "results" / family,
            analysis._figure_dir() / family,
            single_panel=False,
        )
        print(family, result["win_tie_loss"])


if __name__ == "__main__":
    main()
