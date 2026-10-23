# -*- coding: utf-8 -*-
"""Apply the manuscript's bounded schedule refinement to the public hybrid.

Experiment 1 is policy rollouts plus rule starts plus a bounded local search.
This script rebuilds every rule start, applies the same simulated-annealing /
ruin-and-recreate refinement used by ``Compare_Experiment/local_search.py``
under a makespan-only weight, and keeps the best of the refined rule starts and
the policy.  The budget rule mirrors the main protocol: all five rule starts at
4000 iterations and eight passes for instances up to 40x20, otherwise the best
rule start at 12000 iterations and ten passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import public_benchmark as pb
from Compare_Experiment import local_search


def _op_maps(instance: pb.BenchInstance):
    index = {}
    job_of_op = []
    counter = 0
    for job, operations in enumerate(instance.jobs):
        for op in range(len(operations)):
            index[(job, op)] = counter
            job_of_op.append(job)
            counter += 1
    return index, np.asarray(job_of_op, dtype=np.int64)


def _tensors(instance: pb.BenchInstance):
    index, job_of_op = _op_maps(instance)
    total = instance.total_ops
    proc = torch.full((total, instance.n_machines), 99999.0, dtype=torch.float32)
    max_ops = max(len(job) for job in instance.jobs)
    job_op_to_idx = torch.full((instance.n_jobs, max_ops), -1, dtype=torch.long)
    for (job, op), idx in index.items():
        job_op_to_idx[job, op] = idx
        for machine, processing in instance.jobs[job][op].items():
            proc[idx, machine] = float(processing)
    return proc, job_op_to_idx, torch.tensor(job_of_op, dtype=torch.long)


def _rule_schedule(instance: pb.BenchInstance, rule: str):
    index, _ = _op_maps(instance)
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines
    next_op = [0] * n_jobs
    job_ready = [0.0] * n_jobs
    job_completion = [0.0] * n_jobs
    machine_free = [0.0] * n_machines
    machine_load = [0.0] * n_machines
    order = []
    forced = {}

    for _ in range(instance.total_ops):
        candidates = []
        for job in range(n_jobs):
            op_index = next_op[job]
            if op_index >= len(instance.jobs[job]):
                continue
            machine_times = instance.jobs[job][op_index]
            if rule == "LBD":
                machine = min(machine_times,
                              key=lambda m: (machine_load[m],
                                             max(job_ready[job], machine_free[m]) + machine_times[m], m))
            else:
                machine = min(machine_times,
                              key=lambda m: (max(job_ready[job], machine_free[m]) + machine_times[m],
                                             machine_times[m], m))
            processing = machine_times[machine]
            start = max(job_ready[job], machine_free[machine])
            end = start + processing
            remaining_ops = len(instance.jobs[job]) - op_index
            remaining_work = sum(min(op.values()) for op in instance.jobs[job][op_index:])
            due = instance.due_dates[job]
            if rule == "SPT":
                key = (processing, end, job, op_index)
            elif rule == "LBD":
                key = (machine_load[machine], end, job, op_index)
            elif rule == "FIFO":
                key = (job_ready[job], job, op_index)
            elif rule == "MOPNR":
                key = (-remaining_ops, due, job, op_index)
            else:
                key = (-remaining_work, due, job, op_index)
            candidates.append((key, job, op_index, machine, start, end, processing))

        _, job, op_index, machine, start, end, processing = min(candidates, key=lambda row: row[0])
        op_idx = index[(job, op_index)]
        order.append(op_idx)
        forced[op_idx] = machine
        next_op[job] += 1
        job_ready[job] = end
        job_completion[job] = end
        machine_free[machine] = end
        machine_load[machine] += processing

    return float(max(job_completion)), order, forced


class _DecoderInstance:
    def __init__(self, instance: pb.BenchInstance):
        proc, job_op_to_idx, job_of_op = _tensors(instance)
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.ops_per_job = [len(job) for job in instance.jobs]
        self.total_ops = instance.total_ops
        self.proc_time_matrix = proc
        self.job_op_to_idx = job_op_to_idx
        self.op_idx_to_job = job_of_op
        self.due_dates = np.asarray(instance.due_dates, dtype=float)


def refine_instance(instance: pb.BenchInstance, seed: int = 42):
    """Mirror the budget rule of the main rule-comparison protocol."""
    local_search.WEIGHT = (1.0, 0.0, 0.0)
    starts = {rule: _rule_schedule(instance, rule) for rule in pb.RULES}
    best_rule, (raw_cmax, _order, _forced) = min(starts.items(),
                                                 key=lambda item: item[1][0])
    if instance.n_jobs <= 40 and instance.n_machines <= 20:
        selected = list(starts.items())
        budget, lns_iterations, stride = 4000, 8, 1
    else:
        selected = [(best_rule, starts[best_rule])]
        budget = 12000
        lns_iterations = 10
        stride = 2 if instance.total_ops > 450 else 1

    decoder = local_search.Decoder(_DecoderInstance(instance))
    best = {"cmax": float(raw_cmax), "lb": float("nan"), "tardy": float("nan"),
            "source": best_rule}
    for rule, (raw, order, forced) in selected:
        objective, _, _ = local_search.refine(
            decoder, order, forced, budget=budget,
            lns_iterations=lns_iterations, stride=stride, seed=1000 + len(rule),
        )
        if float(objective[0]) < best["cmax"] - 1e-9:
            best = {"cmax": float(objective[0]), "lb": float(objective[1]),
                    "tardy": float(objective[2]), "source": "LS:" + rule}
    return {
        "name": instance.name,
        "best_rule": best_rule,
        "raw_rule_cmax": float(raw_cmax),
        "refined_cmax": best["cmax"],
        "refined_lb": best["lb"],
        "refined_tardy": best["tardy"],
        "refined_source": best["source"],
        "budget": budget,
        "lns_iterations": lns_iterations,
        "stride": stride,
    }


def _refine_worker(payload):
    instance, seed = payload
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    return refine_instance(instance, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("Combined", "Brandimarte", "Lawrence", "all"),
                        default="Combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_root = ROOT / "Public_DataSet" / "PublicBenchmarkValidation"
    instances, _ = pb.load_manifest(data_root)
    by_name = {item.name: item for item in instances}
    families = ("Combined",) if args.family == "all" else (args.family,)
    test_names = {item.name for item in instances if item.split == "test"}
    for family in families:
        per_instance = data_root / "results" / family / "per_instance.csv"
        if not per_instance.exists():
            print("skip %s: no per-instance results" % family)
            continue
        with per_instance.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if family == "Combined":
            rows = [row for row in rows if row["name"] in test_names]
        selected = rows[:args.limit] if args.limit else rows
        payloads = [(by_name[row["name"]], args.seed) for row in selected]
        if args.workers > 1 and len(payloads) > 1:
            import concurrent.futures as futures
            with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_refine_worker, payloads))
        else:
            results = [_refine_worker(item) for item in payloads]
        for refined in results:
            print("%-6s raw=%8.1f refined=%8.1f source=%s"
                  % (refined["name"], refined["raw_rule_cmax"],
                     refined["refined_cmax"], refined["refined_source"]), flush=True)
        if args.limit:
            print(json.dumps(results, indent=2))
            continue

        refined_by_name = {item["name"]: item for item in results}
        for row in rows:
            item = refined_by_name[row["name"]]
            policy = float(row["Proposed-policy"])
            hybrid = min(policy, item["refined_cmax"])
            row["Proposed-hybrid"] = "%.10f" % hybrid
            if row.get("bks") not in (None, "", "nan"):
                bks = float(row["bks"])
                row["gap_Proposed-hybrid"] = "%.10f" % ((hybrid - bks) / bks * 100.0)
            row["best_rule"] = "%.10f" % item["raw_rule_cmax"]
            row["ratio_hybrid_best_rule"] = "%.10f" % (hybrid / item["raw_rule_cmax"])
            row["candidate_source"] = ("policy" if policy <= item["refined_cmax"] + 1e-9
                                       else "rule+refine")

        fields = list(rows[0].keys())
        with per_instance.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (data_root / "results" / family / "hybrid_refinement.json").write_text(
            json.dumps({"seed": args.seed, "rows": results}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        print("updated", per_instance)


if __name__ == "__main__":
    main()
