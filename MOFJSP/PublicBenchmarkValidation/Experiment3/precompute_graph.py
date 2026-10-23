# -*- coding: utf-8 -*-
"""Cache one graph-neural-network baseline on one public protocol."""

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
import baseline_adapter as ba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=pb.project_root_from_script())
    parser.add_argument("--family", default="Combined")
    parser.add_argument("--method", required=True, choices=ba.TRAINABLE)
    parser.add_argument("--rollouts", type=int, default=51)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    if args.family == "Combined":
        test = [item for item in instances if item.split == "test"]
        checkpoint_family = "Lawrence"
    else:
        test = [item for item in instances
                if item.family == args.family and item.split == "test"]
        checkpoint_family = args.family
    geometry = pb.benchmark_geometry(instances)
    selection = ba.selection_map(code_data, checkpoint_family)
    checkpoint = Path(selection.get(args.method,
                                    ba.default_checkpoint(args.project_root, args.method)))
    print("== %s %s: %d test instances ==" % (args.family, args.method, len(test)), flush=True)
    print("    checkpoint: %s" % checkpoint, flush=True)
    internal = pb.materialize_internal_files(list(test), code_data / "internal")
    module = ba.import_baseline(args.project_root, args.method)
    objects = ba._instance_objects(module, test, internal, geometry)
    agent, _ = ba.load_agent(module, checkpoint, lr=2e-4)
    values = ba.evaluate_agent(module, agent, objects, geometry,
                               rollouts=args.rollouts, batch=min(8, args.rollouts))
    path = (code_data / "cache" / args.family
            / ("%s_r%d_s%d.json" % (args.method, args.rollouts, args.seed)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=1), encoding="utf-8")
    print("cached", path, flush=True)


if __name__ == "__main__":
    main()
