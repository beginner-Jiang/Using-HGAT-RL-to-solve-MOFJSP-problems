# -*- coding: utf-8 -*-
"""Regenerate the public Experiment 1 tables and figures for both families."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

import analyze_public_experiments as analysis
import public_benchmark as pb

for family in ("Combined",):
    rows = analysis.load_rows(family)
    if not rows:
        print("skip %s: no per-instance results" % family)
        continue
    result = analysis.run_experiment(
        rows, analysis.METHODS_E1, "Proposed-hybrid", list(pb.RULES),
        "Experiment1", "Public Experiment 1: dispatching-rule comparison",
        analysis._data_dir() / "results" / family,
        analysis._figure_dir() / family,
        single_panel=False,
    )
    print(family, result["win_tie_loss"])
