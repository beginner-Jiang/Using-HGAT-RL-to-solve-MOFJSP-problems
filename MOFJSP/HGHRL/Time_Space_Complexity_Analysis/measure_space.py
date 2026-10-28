# -*- coding: utf-8 -*-
"""Measure graph size and peak inference memory for the nine scales.

For every scale one instance is solved greedily (one schedule, no sampling).  Recorded are the
node count V, the edge count E of the hetero-disjunctive graph at reset, the peak tensor memory
allocated during the rollout (torch.cuda.max_memory_allocated, which excludes the CUDA context
and therefore isolates the graph and activation footprint) and the wall-clock time.

Usage:  python HGHRL/Time_Space_Complexity_Analysis/measure_space.py
"""
import csv
import gc
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "HGHRL"))

from Gantt_Chart import (  # noqa: E402
    GreedyDecoder, HGAT_CFG, MOFJSPInstance, load_model, load_training_globals, read_fjsp_instance,
)
from HGHRL import HeteroGraphEnv  # noqa: E402

OUT = os.path.join(ROOT, "Figure_And_File", "HGHRL", "Time_Space_Complexity_Analysis")
MODEL = os.path.join(ROOT, "model", "hghrl", "hghrl_best.pth")
SCALES = ("10x5", "15x10", "20x10", "25x15", "30x20", "40x20", "50x25", "75x25", "100x30")


def instance_file(scale):
    directory = os.path.join(ROOT, "DataSet", "comparison_instances")
    for name in sorted(os.listdir(directory)):
        if scale in name and name.endswith(".txt"):
            return os.path.join(directory, name)
    raise FileNotFoundError(scale)


def count_edges(state):
    total = 0
    for value in state["edges"].values():
        if isinstance(value, (tuple, list)):
            total += int(value[0].shape[-1])
        else:
            total += int(value.shape[-1])
    return total


def main():
    maxima = load_training_globals()
    max_jobs, max_machines, max_ops, max_proc, max_due, _ = maxima
    gat, upper, lower, stats, mode, _dim_op, _dim_mac = load_model(MODEL)
    rows = []
    for scale in SCALES:
        jobs, caps, due, name, size = read_fjsp_instance(instance_file(scale))
        instance = MOFJSPInstance(jobs, caps, due, name, size, maxima[3], maxima[5])
        env = HeteroGraphEnv(instance, max_jobs, max(max_ops, instance.total_ops), max_machines,
                             max_proc, max_due, HGAT_CFG, feature_mode=mode)
        state = env.reset()
        nodes = instance.total_ops + instance.n_machines
        edges = count_edges(state)
        decoder = GreedyDecoder(instance, gat, upper, lower, stats)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        done = False
        steps = 0
        while not done and steps <= 20000:
            action = decoder.act(state, deterministic=True)
            state, _, done = env.step(action)
            steps += 1
        torch.cuda.synchronize()
        elapsed = time.time() - started
        peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        rows.append({
            "scale": scale, "N": instance.total_ops, "M": instance.n_machines,
            "V": nodes, "E": edges, "peak_MiB": round(peak, 3),
            "time_s": round(elapsed, 3), "steps": steps,
        })
        print("%-7s N=%4d M=%2d V=%4d E=%6d peak=%8.3f MiB  t=%7.3f s"
              % (scale, instance.total_ops, instance.n_machines, nodes, edges, peak, elapsed))
        del decoder, env, instance
        gc.collect()
        torch.cuda.empty_cache()

    path = os.path.join(OUT, "space_scaling_data.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())



