# -*- coding: utf-8 -*-
"""Regenerate the public Experiment 3 tables and figures for both families."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

import analyze_public_experiments as analysis

for family in ("Combined",):
    rows = analysis.load_rows(family)
    if not rows or not all(m in rows[0] for m in analysis.METHODS_E3):
        print("skip %s: stale results are missing one or more Experiment3 methods" % family)
        continue
    result = analysis.run_experiment(
        rows, analysis.METHODS_E3, "Proposed-policy-refined",
        ["MLP+DQN", "MLP+PPO", "GAT+PPO", "HGNN", "DAN", "RESCHED"],
        "Experiment3", "Public Experiment 3: learning-baseline comparison",
        analysis._data_dir() / "results" / family,
        analysis._figure_dir() / family,
        single_panel=False,
    )
    print(family, result["win_tie_loss"])
