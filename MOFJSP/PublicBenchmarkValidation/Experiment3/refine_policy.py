# -*- coding: utf-8 -*-
"""Refine the best policy schedule of every test instance (no rule candidates).

The decoder is the deployed policy itself: one greedy and fifty sampled
rollouts.  The best makespan schedule is then improved by the same bounded
simulated-annealing / ruin-and-recreate search used by the main comparison.
The rule schedules are never used, so this is not the hybrid decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
for path in (COMMON, HERE, HERE.parent / "Experiment1"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import public_benchmark as pb
import baseline_adapter as ba
import refine_public_hybrid as rp
from Compare_Experiment import local_search


def capture_best_schedule(Param, agent, cfg, item, geometry, rollouts, batch):
    import torch

    pinst = pb.to_param_instance(Param, item, geometry[3], 1.0)
    best = None

    def run(width, deterministic):
        nonlocal best
        envs = [Param.HeteroGraphEnv(pinst, geometry[0], geometry[1], geometry[2],
                                     geometry[3], geometry[4], cfg)
                for _ in range(width)]
        states = [env.reset() for env in envs]
        orders = [[] for _ in range(width)]
        forced = [{} for _ in range(width)]
        done = [False] * width
        while not all(done):
            live = [index for index in range(width) if not done[index]]
            actions, _, _ = agent.get_action_batch(
                [states[index] for index in live], [pinst] * len(live),
                deterministic=deterministic, check_invalid=False)
            for slot, index in enumerate(live):
                action = actions[slot]
                job, op, machine = int(action[0]), int(action[1]), int(action[2])
                op_index = int(pinst.job_op_to_idx[job, op])
                orders[index].append(op_index)
                forced[index][op_index] = machine
                next_state, _, flag = envs[index].step(action)
                states[index] = next_state
                done[index] = bool(flag)
        for index, env in enumerate(envs):
            cmax = float(env.job_completion_time.max())
            if best is None or cmax < best[0]:
                best = (cmax, orders[index], forced[index])

    with torch.no_grad():
        run(1, True)
        done_count = 1
        while done_count < rollouts:
            width = min(batch, rollouts - done_count)
            run(width, False)
            done_count += width
    return best


def _refine_worker(payload):
    item, order, forced = payload
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    local_search.WEIGHT = (1.0, 0.0, 0.0)
    decoder = local_search.Decoder(rp._DecoderInstance(item))
    if item.n_jobs <= 40 and item.n_machines <= 20:
        budget, lns, stride = 4000, 8, 1
    else:
        budget, lns, stride = 12000, 10, (2 if item.total_ops > 450 else 1)
    objective, _, _ = local_search.refine(
        decoder, order, forced, budget=budget, lns_iterations=lns,
        stride=stride, seed=4242,
    )
    return item.name, float(objective[0]), budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=pb.project_root_from_script())
    parser.add_argument("--family", default="Combined")
    parser.add_argument("--rollouts", type=int, default=51)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    test = [item for item in instances if item.split == "test"]
    geometry = pb.benchmark_geometry(instances)
    selection = ba.selection_map(code_data, "Lawrence")
    checkpoint = Path(selection.get("Proposed-policy",
                                    args.project_root / "model" / "hghrl" / "hghrl_best.pth"))
    print("== checkpoint: %s ==" % checkpoint, flush=True)
    Param, agent, cfg, _ = pb.load_adopted_agent(args.project_root, checkpoint)
    agent.gat.eval(); agent.upper.eval(); agent.lower.eval()

    payloads = []
    raw = {}
    for item in test:
        best = capture_best_schedule(Param, agent, cfg, item, geometry,
                                     args.rollouts, 8)
        raw[item.name] = best[0]
        payloads.append((item, best[1], best[2]))
        print("%-6s policy=%8.1f" % (item.name, best[0]), flush=True)

    import concurrent.futures as futures
    if args.workers > 1:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            refined = list(pool.map(_refine_worker, payloads))
    else:
        refined = [_refine_worker(item) for item in payloads]

    values = {}
    detail = []
    for name, value, budget in refined:
        values[name] = value
        detail.append({"name": name, "policy": raw[name], "refined": value,
                       "budget": budget})
        print("%-6s policy=%8.1f refined=%8.1f" % (name, raw[name], value), flush=True)

    path = (code_data / "cache" / args.family
            / ("Proposed-policy-refined_r%d_s42.json" % args.rollouts))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=1), encoding="utf-8")
    (code_data / "results" / args.family).mkdir(parents=True, exist_ok=True)
    (code_data / "results" / args.family / "policy_refinement.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8")
    print("cached", path, flush=True)


if __name__ == "__main__":
    main()
