# -*- coding: utf-8 -*-
"""Evaluate one expanded learning baseline and cache its per-instance results.

The three flat/GAT baselines run one rollout at a time, so evaluating them in
parallel, one process per method, keeps the formal 51-rollout protocol without
changing any seed or budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
for path in (COMMON, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import public_benchmark as pb
import expanded_baselines as eb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=pb.project_root_from_script())
    parser.add_argument("--family", required=True)
    parser.add_argument("--method", required=True, choices=tuple(eb.EVALUATORS))
    parser.add_argument("--rollouts", type=int, default=51)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    if args.family == "Combined":
        test = [item for item in instances if item.split == "test"]
    else:
        test = [item for item in instances
                if item.family == args.family and item.split == "test"]
    geometry = pb.benchmark_geometry(instances)
    print("== %s %s: %d test instances ==" % (args.family, args.method, len(test)),
          flush=True)
    values = eb.EVALUATORS[args.method](
        args.project_root, code_data, test, geometry, args.rollouts, args.seed)
    path = eb.cache_path(code_data, args.family, args.method, args.rollouts, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=1), encoding="utf-8")
    print("cached", path, flush=True)


if __name__ == "__main__":
    main()
