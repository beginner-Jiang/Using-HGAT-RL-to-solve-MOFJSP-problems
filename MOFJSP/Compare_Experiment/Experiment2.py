"""
Comparison Experiment 2: meta-heuristics versus deep reinforcement learning methods
The evaluated scales (jobs x machines) are defined by a fixed list; each scale should provide 10 instances.

The multi-objective metrics (HV/GD/SP) are computed uniformly on the non-dominated front obtained by each
method, so that the evolutionary and reinforcement learning conventions match; the per-instance solution set of every (scale, method) pair is cached into Figure_And_File/Compare_Experiment/
exp2_raw_results.pkl and reused directly on repeated runs (recomputing only the post-processing metrics does not require resampling the policy).
"""

import os
import time
import glob
import sys
import numpy as np


def _allow_foreign_numpy_checkpoint():
    """Make a checkpoint written by the other numpy major readable in this process.

    A checkpoint stores the module names of the numpy that wrote it: the device build of this
    project runs numpy 2, which keeps its internals in ``numpy._core``, while the host build runs
    numpy 1, which keeps them in ``numpy.core``.  The graph baselines (HGNN, DAN, RESCHED) and
    the proposed method are trained by the device build, so their checkpoints could only be read
    with the device interpreter - which put the measurement of their shards on the same single
    device as the trainings of the ablation study.  The two names are aliased to the same modules,
    exactly as the probes of _Review do it, so the host interpreter can read those checkpoints
    and the shards of the graph methods can be measured on the cores while the device trains.
    """
    if hasattr(np, "_core"):
        return
    for name in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        module = getattr(np.core, name, None)
        if module is not None:
            sys.modules.setdefault("numpy._core." + name, module)
    sys.modules.setdefault("numpy._core", np.core)


_allow_foreign_numpy_checkpoint()
import torch
import random
import pickle
import hashlib
from collections import defaultdict


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas is not installed; Excel export is skipped and results are printed.")

try:
    from scipy.stats import (
        wilcoxon,
        friedmanchisquare,
        rankdata,
        t as t_dist,
        f as f_dist,
    )
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy is not installed; only means are reported.")




# Fingerprint of the checkpoint a (scale, method) cache was measured with.  The cache used to be
# trusted whenever the instance list matched, so the checkpoint was never part of the decision:


def checkpoint_fingerprint(path):
    """Size, modification time and a short hash of a checkpoint, or None when it is missing."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return {"name": os.path.basename(path), "size": int(stat.st_size),
            "mtime": round(stat.st_mtime, 3), "sha256": digest.hexdigest()[:32]}


_FINGERPRINTS = {}


def fingerprint(path):
    """Memoised fingerprint of ``path``: a poll only hashes a checkpoint that changed."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), int(stat.st_size), round(stat.st_mtime, 3))
    if key not in _FINGERPRINTS:
        _FINGERPRINTS.clear()
        _FINGERPRINTS[key] = checkpoint_fingerprint(path)
    return _FINGERPRINTS[key]


def provenance_matches(cached, model_path):
    """True when a cached entry was measured with the checkpoint that is loaded now."""
    if not isinstance(cached, dict):
        return False
    if model_path is None:
        # The evolutionary methods carry no checkpoint, so the protocol is their provenance: an entry
        # measured with another evaluation budget is not comparable with the current run and has to
        # be measured again.
        return cached.get("provenance") == evolutionary_provenance()
    record = cached.get("provenance")
    if not isinstance(record, dict):
        return False
    current = fingerprint(model_path)
    return bool(current) and record.get("sha256") == current.get("sha256")


# Automatically locate the project root directory

def find_project_root():
    """Search upwards from the script directory until a directory containing comparison_instances is found"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check the script directory
    if os.path.isdir(os.path.join(script_dir, "DataSet", "comparison_instances")):
        return script_dir
    # Check the parent directory
    parent_dir = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(parent_dir, "DataSet", "comparison_instances")):
        return parent_dir
    # Check the grandparent directory (at most two levels up)
    grandparent_dir = os.path.dirname(parent_dir)
    if os.path.isdir(os.path.join(grandparent_dir, "DataSet", "comparison_instances")):
        return grandparent_dir
    # Fall back to the script directory when nothing is found (and emit a warning)
    print("Warning: project root not found; the script directory is used.")
    return script_dir

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)  # make sure the project root is on the module search path
# The learned methods are imported under their package path here, but the environment pool of
# MLP+DQN imports the module under its bare name inside the child processes it spawns, and a
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DRL"))

print(f"Project root: {PROJECT_ROOT}")


# Generic data reading helpers (used only to compute the global maximum values)

try:
    from PPO_HGAT.PPO_HGAT import read_fjsp_instance as read_ppo_instance
except ImportError:
    print("Error: cannot import read_fjsp_instance from PPO_HGAT.PPO_HGAT")
    sys.exit(1)


# MOEA/D module

try:
    from Meta_Heuristic.MOEA_D import (
        MOEADScheduler,
        read_fjsp_instance as read_moead,
        Job as MOEAJob,
        Operation as MOEAOp
    )
except ImportError:
    print("Warning: cannot import the MOEA/D module; the method is skipped (check the folder name Meta_Heuristic).")
    MOEADScheduler = None
    read_moead = None


# NSGA-II module

try:
    from Meta_Heuristic.NSGA_II import (
        NSGA2Scheduler,
        read_fjsp_instance as read_nsga2,
        Job as NSGA2Job,
        Operation as NSGA2Op
    )
except ImportError:
    print("Warning: cannot import the NSGA-II module; the method is skipped.")
    NSGA2Scheduler = None
    read_nsga2 = None


# NSGA-III module (added modern multi-objective EA with reference-point based environmental selection)

try:
    from Meta_Heuristic.NSGA_III import (
        NSGA3Scheduler,
        read_fjsp_instance as read_nsga3,
        Job as NSGA3Job,
        Operation as NSGA3Op
    )
except ImportError:
    print("Warning: cannot import the NSGA-III module; the method is skipped.")
    NSGA3Scheduler = None
    read_nsga3 = None


# RVEA module (added modern multi-objective EA with reference-vector angle-penalized distance selection)

try:
    from Meta_Heuristic.RVEA import (
        RVEAScheduler,
        read_fjsp_instance as read_rvea,
        Job as RVEAJob,
        Operation as RVEAOp
    )
except ImportError:
    print("Warning: cannot import the RVEA module; the method is skipped.")
    RVEAScheduler = None
    read_rvea = None


# DQN+MLP module

# The module itself, for the rule decoding of that baseline (see run_dqn_mlp): the value network
# ranks dispatching rules, and the rule table of the module resolves the chosen rule into a pair.
try:
    from DRL import DQN_MLP as DQN_LIB
except ImportError as e:
    print(f"Warning: cannot import the MLP+DQN module ({e}); the method is skipped.")
    DQN_LIB = None
try:
    from DRL.DQN_MLP import (
        read_fjsp_instance_structured as read_dqn_instance,
        JobShopInstance as DQNJobInstance,
        Config as DQNConfig,
        MOFJSP_Env as DQNEnv,
        DQN_Agent,
        RunningMeanStd as DQNStateNorm
    )
except ImportError as e:
    print(f"Warning: cannot import the MLP+DQN module ({e}); the method is skipped.")
    read_dqn_instance = None
    DQNJobInstance = None
    DQNConfig = None
    DQNEnv = None
    DQN_Agent = None
    DQNStateNorm = None


# PPO+MLP module

try:
    from DRL.PPO_MLP import (
        read_fjsp_instance as read_ppo_mlp_instance,
        JobShopInstance as PPOJobInstance,
        Config as PPOConfig,
        MOFJSP_Env as PPOEnv,
        PolicyNetwork as PPOPolicy,
        ValueNetwork as PPOValue,
        PPOKLTrainer as PPOAgent,
        DEVICE as PPO_DEVICE
    )
    import __main__
    __main__.Config = PPOConfig
except ImportError as e:
    print(f"Warning: cannot import the MLP+PPO module ({e}); the method is skipped.")
    read_ppo_mlp_instance = None
    PPOJobInstance = None
    PPOConfig = None
    PPOEnv = None
    PPOPolicy = None
    PPOValue = None
    PPOAgent = None
    PPO_DEVICE = None


# PPO+GAT module

try:
    from DRL.PPO_GAT import (
        read_fjsp_instance as read_ppo_gat_instance,
        MOFJSPInstance as GATInstance,
        Config as GATConfig,
        HomogeneousDisjunctiveGraphEnv as GATEnv,
        HomogeneousGAT,
        PolicyNetwork as GATPolicy,
        ValueNetwork as GATValue,
        TorchRunningMeanStd,
        DEVICE as GAT_DEVICE
    )
except ImportError as e:
    print(f"Warning: cannot import the GAT+PPO module ({e}); the method is skipped.")
    read_ppo_gat_instance = None
    GATInstance = None
    GATConfig = None
    GATEnv = None
    HomogeneousGAT = None
    GATPolicy = None
    GATValue = None
    TorchRunningMeanStd = None
    GAT_DEVICE = None


# PPO+HGAT module

try:
    from PPO_HGAT.PPO_HGAT import (
        HeteroGAT,
        UpperPolicy,
        LowerPolicy,
        Critic,
        HierarchicalPPOAgent,
        MOFJSPInstance as HGATInstance,
        TorchRunningMeanStd,
        HeteroGraphEnv as HGATEnv,
        cfg as HGATConfig,
        DEVICE as HGAT_DEVICE
    )
except ImportError as e:
    print(f"Warning: cannot import the HGAT+PPO module ({e}); the method is skipped.")
    HeteroGAT = None
    UpperPolicy = None
    LowerPolicy = None
    Critic = None
    HierarchicalPPOAgent = None
    HGATInstance = None
    HGATEnv = None
    HGATConfig = None
    HGAT_DEVICE = None


# HGNN baseline module (single-level policy with a heterogeneous mean-aggregation encoder)

try:
    from DRL.HGNN import run_scheduler as hgnn_run_scheduler
except ImportError as e:
    print(f"Warning: cannot import the HGNN module ({e}); the method is skipped.")
    hgnn_run_scheduler = None


# DAN baseline module (single-level policy with a dual-attention encoder)

try:
    from DRL.DAN import run_scheduler as dan_run_scheduler
except ImportError as e:
    print(f"Warning: cannot import the DAN module ({e}); the method is skipped.")
    dan_run_scheduler = None


# RESCHED baseline module (single-level policy with a Transformer encoder over a simplified state)

try:
    from DRL.RESCHED import run_scheduler as resched_run_scheduler
except ImportError as e:
    print(f"Warning: cannot import the RESCHED module ({e}); the method is skipped.")
    resched_run_scheduler = None


# Configuration parameters


DATA_DIR = os.path.join(PROJECT_ROOT, "DataSet", "comparison_instances")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Model paths of each method (already trained)
DQN_MODEL_PATH = os.path.join(MODEL_DIR, "dqn/dqn_model.pth")
PPO_MLP_MODEL_PATH = os.path.join(MODEL_DIR, "ppo/ppo_model.pth")
PPO_GAT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_gat/ppo_gat_best.pth")
PPO_HGAT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hgat/ppo_hgat_best.pth")
# Model paths of the added comparison methods (run the corresponding module main() to train them first)
HGNN_MODEL_PATH = os.path.join(MODEL_DIR, "hgnn/hgnn_best.pth")
DAN_MODEL_PATH = os.path.join(MODEL_DIR, "dan/dan_best.pth")
RESCHED_MODEL_PATH = os.path.join(MODEL_DIR, "resched/resched_best.pth")

# Fixed list of scales
SIZE_LIST = [
    (10, 5),   # small
    (15, 10),  # small
    (20, 10),  # small
    (25, 15),  # medium
    (30, 20),  # medium
    (40, 20),  # medium
    (50, 25),  # large
    (75, 25),  # large
    (100, 30), # large
]

# Multi-objective weights
W = [0.4, 0.3, 0.3]   # [Cmax, LB, Tardy]

# Evolutionary multi-objective methods solved directly without loading a model (sharing the same evaluation budget)
EVOLUTIONARY_METHODS = ("MOEA/D", "NSGA-II", "NSGA-III", "RVEA")

# Order of the methods used for the statistical tests and the export (kept consistent with the paper tables)
METHOD_ORDER = ["MOEA/D", "NSGA-II", "NSGA-III", "RVEA", "MLP+DQN", "MLP+PPO",
                "GAT+PPO", "HGNN", "DAN", "RESCHED", "HeteroGAT+PPO"]

# Evaluation budget of the comparison, in the currency both families share: one evaluated schedule of
# one instance.  A learned method constructs NUM_RUNS_RL schedules with its own policy per instance
NUM_RUNS_RL = 50
# Population and generation count of the evolutionary family under that budget.  Fifty individuals
# over a single generation spend the whole budget on the initial population: the variation
EVOLUTIONARY_POPULATION = 100
EQUAL_BUDGET = False
EVOLUTIONARY_GENERATIONS = (max(1, int(round(NUM_RUNS_RL / EVOLUTIONARY_POPULATION)))
                            if EQUAL_BUDGET else 100)


def evolutionary_provenance():
    """Provenance of an instance-only entry: the evaluation budget it was measured with."""
    # The layout is part of the provenance: the same number of evaluated schedules spent as one
    # generation of fifty individuals and as five generations of ten are different measurements of
    # the methods, and a cache of the first layout must not be reported for the second.
    return "evals=%d pop=%d gen=%d" % (EVOLUTIONARY_POPULATION * EVOLUTIONARY_GENERATIONS,
                                        EVOLUTIONARY_POPULATION, EVOLUTIONARY_GENERATIONS)


# Random seed list (used for repeated runs)
RL_SEEDS = [42 + i for i in range(NUM_RUNS_RL)]

# Fixed random seed of the evolutionary multi-objective methods: repeated runs of the same experiment yield identical solution sets, keeping the results reproducible
EVOLUTIONARY_SEED = 2024

# Raw result cache file: a full Exp2 run takes hours on CPU, so the per-instance solution set of every (scale, method)
# pair is cached on disk and reused directly on repeated runs (or when only the post-processing metrics HV/GD/SP and the statistical tests change), without resampling the policy.
EXP2_CACHE_PATH = os.path.join(PROJECT_ROOT, "Figure_And_File", "Compare_Experiment",
                               "exp2_raw_results.pkl")

# Sharded (parallel) run support: solving 11 methods on 9 scales sequentially takes tens of hours on CPU, whereas the
# per-instance solution sets of different scales and methods are independent (HV/GD/SP and the statistical tests are post-processed at the end), so environment
_SHARD_SCALES = [s.strip() for s in os.environ.get("EXP2_SCALES", "").split(",") if s.strip()]
_SHARD_METHODS = [m.strip() for m in os.environ.get("EXP2_METHODS", "").split(",") if m.strip()]
# Note: EXP2_SCALES only decides which scales this run handles and must never be used to shrink SIZE_LIST. The global maximum tuple must
# always be computed over the complete comparison dataset (all scales in SIZE_LIST): the reinforcement learning models normalise
EXP2_SCALE_LIST = ([pair for pair in SIZE_LIST if f"{pair[0]}x{pair[1]}" in _SHARD_SCALES]
                   if _SHARD_SCALES else list(SIZE_LIST))
# Post-processing only: re-aggregate HV/GD/SP and the statistical tests from the cache without
# sampling any policy again.  Used when only the metric convention changes.
EXPORT_ONLY = os.environ.get("EXP2_EXPORT_ONLY", "").strip().lower() not in ("", "0", "false")
EXP2_CACHE_PATH = os.path.join(PROJECT_ROOT, "Figure_And_File", "Compare_Experiment",
                               os.environ.get("EXP2_CACHE_FILE", "").strip()
                               or "exp2_raw_results.pkl")


# Helper: compute the weighted objective value


def weighted_objective(obj):
    """obj = [makespan, load_balance, total_tardiness]"""
    return W[0] * obj[0] + W[1] * obj[1] + W[2] * obj[2]


# Helper: multi-objective metric computation


def _hypervolume_2d(points, ref_x, ref_y):
    """
    Exact value of the two-dimensional hypervolume: each (x, y) in points corresponds to a rectangle [x, ref_x] x [y, ref_y]
    and the return value is the area of their union (as in the three-dimensional case, all objectives are minimised).
    """
    area = 0.0
    best_y = ref_y          # upper boundary of the rectangle union at the current x (lowest y of the skyline)
    prev_x = None
    for x, y in sorted(points):
        if prev_x is not None and x > prev_x:
            area += (x - prev_x) * (ref_y - best_y)
        if y < best_y:
            best_y = y
        prev_x = x
    if prev_x is not None:
        area += (ref_x - prev_x) * (ref_y - best_y)
    return area


def compute_hypervolume(points, ref_point):
    """
    Exact value of the three-dimensional hypervolume (all objectives are minimised).

    Each solution p corresponds to a hyper-box [p, ref_point] and HV is the volume of their union. In the three-dimensional case slicing along the z axis
    solves it exactly: after sorting the solutions by increasing z, the cross-sectional area of the union within the slice between two neighbouring z values equals
    the area of the x-y rectangle union of the "active" solutions (those whose z is not above the slice lower bound); accumulating over the slices gives the total volume.
    Compared with a fixed-sample Monte-Carlo estimate, the exact computation has no sampling error and never makes several methods
    tie exactly on HV when the solution sets are small and therefore indistinguishable.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 3:
        raise ValueError("Only 3 objectives supported for hypervolume.")
    ref = np.asarray(ref_point, dtype=float)
    if np.any(np.isnan(pts)) or np.any(np.isnan(ref)):
        return np.nan

    # Only solutions strictly inside the reference point correspond to a non-empty hyper-box
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    # Drop dominated solutions: their hyper-box is already covered by other solutions, so they do not affect the union volume but they reduce the number of slices
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        if np.any(np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)):
            keep[i] = False
    pts = pts[keep]

    z_values = np.unique(pts[:, 2])
    volume = 0.0
    for idx, z_low in enumerate(z_values):
        z_high = float(z_values[idx + 1]) if idx + 1 < len(z_values) else float(ref[2])
        if z_high <= z_low:
            continue
        # Inside this slice only solutions whose z is not above the slice lower bound can cover the whole slice thickness
        active = pts[pts[:, 2] <= z_low][:, :2]
        volume += (z_high - z_low) * _hypervolume_2d([tuple(row) for row in active],
                                                     float(ref[0]), float(ref[1]))
    return float(volume)

def compute_gd(pf, reference_pf):
    """
    Generational distance GD: mean Euclidean distance from pf to reference_pf
    """
    pf = np.array(pf)
    ref = np.array(reference_pf)
    if len(pf) == 0 or len(ref) == 0:
        return np.nan
    distances = []
    for p in pf:
        dist = np.min(np.linalg.norm(ref - p, axis=1))
        distances.append(dist)
    return np.mean(distances)

def compute_sp(pf):
    """
    Spacing SP: standard deviation of the distances between neighbouring points in pf (a measure of uniformity)
    Returns NaN when the solution set has fewer than 2 points.
    """
    pf = np.array(pf)
    if len(pf) < 2:
        return np.nan
    # A front of identical copies has a spacing of zero under the plain formula, which
    # rewards a policy that returns the same schedule fifty times.  Collapse duplicates
    # first, so that only the distinct schedules of the set are scored.
    pf = np.unique(np.round(np.asarray(pf, dtype=float), 6), axis=0)
    # With fewer than three distinct points every nearest-neighbour distance is the same
    # (zero for two points), so the statistic cannot discriminate uniformity and is not
    # reported; otherwise a collapsed front would look perfectly spaced.
    if len(pf) < 3:
        return np.nan
    dists = []
    for i in range(len(pf)):
        others = np.concatenate([pf[:i], pf[i+1:]])
        if len(others) == 0:
            continue
        d = np.min(np.linalg.norm(others - pf[i], axis=1))
        dists.append(d)
    if len(dists) == 0:
        return np.nan
    mean_d = np.mean(dists)
    sp = np.sqrt(np.mean((dists - mean_d)**2))
    return sp

def nondominated_sort(objectives):
    """
    Fast non-dominated sorting; returns the indices of the first front (non-dominated solutions)
    """
    objectives = np.array(objectives)
    n = len(objectives)
    if n == 0:
        return []
    dominated_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check whether i dominates j
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                dominated_solutions[i].append(j)
            elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for idx in fronts[i]:
            for j in dominated_solutions[idx]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        i += 1
        if next_front:
            fronts.append(next_front)
    # Return the indices of the first front
    return fronts[0] if fronts and fronts[0] else []


# Compute the global maximum values (used for environment initialisation)

def compute_global_maxima(instance_dict):
    """
    Iterate over every instance file and compute the global maximum tuple
    (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability).

    This tuple has exactly the same meaning as global_max_vals in the ablation experiment: it is used both to build MOFJSPInstance
    (normalisation of the operation/machine features and of the assignment edges) and to build HeteroGraphEnv (state, action mask and
    reward normalisation). Every reinforcement learning method (including the proposed one) must use the same tuple, otherwise the
    methods would see different input feature scales and the comparison would not be fair.
    """
    max_jobs = 0
    max_machines = 0
    max_total_ops = 0
    max_proc_time = 0.0
    max_due_date = 0.0
    max_capability = 0

    for (jobs, mach), file_list in instance_dict.items():
        max_jobs = max(max_jobs, jobs)
        max_machines = max(max_machines, mach)
        for fpath in file_list:
            try:
                jobs_data, machine_capabilities, due_dates, fname, size = read_ppo_instance(fpath)
                max_total_ops = max(max_total_ops, sum(len(job_ops) for job_ops in jobs_data))
                for job_ops in jobs_data:
                    for op in job_ops:
                        for t in op.machine_times.values():
                            max_proc_time = max(max_proc_time, t)
                max_due_date = max(max_due_date, max(due_dates) if due_dates else 0)
                # The machine capability level is the third component of the machine node feature and must be normalised by the global maximum exactly as during training
                max_capability = max(max_capability, max(machine_capabilities) if machine_capabilities else 0)
            except Exception as e:
                print(f"Warning: cannot read instance {fpath}: {e}")
                continue
    # Avoid division by zero
    max_proc_time = max(max_proc_time, 1.0)
    max_due_date = max(max_due_date, 1.0)
    max_capability = max(max_capability, 1)
    return (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability)


# Raw result cache I/O (checkpointing/reuse support)

def _load_raw_cache(path=EXP2_CACHE_PATH):
    """Read the cached per-instance solution sets; return an empty dict when the file is missing or corrupted."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        if isinstance(cache, dict):
            return cache
        print(f"Warning: cache {path} is invalid; recomputing.")
    except Exception as e:
        print(f"Warning: cannot read cache {path} ({e}); recomputing.")
    return {}


def _save_raw_cache(cache, path=EXP2_CACHE_PATH):
    """Write the cache back atomically so that an interruption never leaves a corrupted partial file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(cache, f)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"Warning: cannot write cache {path} ({e}).")


# Functions that run each method
def excluded_by_qualification():
    """Learned methods the qualification gate of the wave left out of the comparison.

    The gate writes one row per learned method to this report before the shards of the comparison are
    measured, and the shard launcher reads the same file before it measures a method.  A method whose
    selected model did not clearly beat the untrained policy of its own architecture is therefore not
    part of the study: measuring it here would report the checkpoint of a method that is still being
    retrained, and the numbers of a model that never learned would enter the tables of the comparison.
    An absent report excludes nothing, so a project without the gate behaves as it did before.
    """
    import csv

    path = os.path.join(PROJECT_ROOT, "Figure_And_File", "Qualification_Report.csv")
    if not os.path.isfile(path):
        return set()
    excluded = set()
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("qualified", "")).strip().lower() not in ("true", "1"):
                    excluded.add((row.get("method") or "").strip())
    except OSError:
        return set()
    return excluded




def reset_evolutionary_seed():
    """Reset the global random state so that the evolutionary methods return identical results across runs."""
    random.seed(EVOLUTIONARY_SEED)
    np.random.seed(EVOLUTIONARY_SEED)


def run_moead(file_path):
    """Run MOEA/D and return the list of objective values on the Pareto front"""
    if MOEADScheduler is None or read_moead is None:
        raise ImportError("The MOEA/D module was not imported")
    reset_evolutionary_seed()
    jobs, machines = read_moead(file_path)
    scheduler = MOEADScheduler(
        jobs=jobs,
        machines=machines,
        population_size=EVOLUTIONARY_POPULATION,
        max_generations=EVOLUTIONARY_GENERATIONS,
        crossover_rate=0.9,
        mutation_rate=0.1,
        neighbor_size=10,
        decomposition_method='tchebycheff'
    )
    scheduler.run()
    objectives = [ind.objectives for ind in scheduler.ep]
    return objectives

def run_nsga2(file_path):
    """Run NSGA-II and return the list of objective values on the Pareto front"""
    if NSGA2Scheduler is None or read_nsga2 is None:
        raise ImportError("The NSGA-II module was not imported")
    reset_evolutionary_seed()
    jobs, machines = read_nsga2(file_path)
    scheduler = NSGA2Scheduler(
        jobs=jobs,
        machines=machines,
        population_size=EVOLUTIONARY_POPULATION,
        max_generations=EVOLUTIONARY_GENERATIONS,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    scheduler.run()
    objectives = [ind.objectives for ind in scheduler.best_solutions]
    return objectives

def run_nsga3(file_path):
    """
    Run NSGA-III and return the list of objective values on the Pareto front.
    Note: the NSGA-III population size equals the number of reference points, so the effective population size is the
    size of the Das-Dennis lattice containing at least 50 points (55 reference points for 3 objectives and 9 divisions).
    """
    if NSGA3Scheduler is None or read_nsga3 is None:
        raise ImportError("The NSGA-III module was not imported")
    reset_evolutionary_seed()
    jobs, machines = read_nsga3(file_path)
    scheduler = NSGA3Scheduler(
        jobs=jobs,
        machines=machines,
        population_size=EVOLUTIONARY_POPULATION,
        max_generations=EVOLUTIONARY_GENERATIONS,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    scheduler.run()
    objectives = [ind.objectives for ind in scheduler.best_solutions]
    return objectives

def run_rvea(file_path):
    """Run RVEA and return the list of objective values on the Pareto front"""
    if RVEAScheduler is None or read_rvea is None:
        raise ImportError("The RVEA module was not imported")
    reset_evolutionary_seed()
    jobs, machines = read_rvea(file_path)
    scheduler = RVEAScheduler(
        jobs=jobs,
        machines=machines,
        population_size=EVOLUTIONARY_POPULATION,
        max_generations=EVOLUTIONARY_GENERATIONS,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    scheduler.run()
    objectives = [ind.objectives for ind in scheduler.best_solutions]
    return objectives

def _verify_dqn_policy(agent, model_path):
    """Raise when the restored DQN policy does not carry the weights of the checkpoint.

    ``DQN_Agent.__init__`` only builds the network, so the restore is an explicit call and a missing or
    ineffective call would evaluate an untrained policy while every metric still looks plausible.
    """
    checkpoint = torch.load(model_path, map_location=agent.cfg.device, weights_only=False)
    reference = checkpoint['policy_net_state_dict']
    expected = float(sum(float(value.double().sum()) for value in reference.values()
                         if torch.is_tensor(value) and value.is_floating_point()))
    restored = float(sum(float(param.double().sum()) for param in agent.policy_net.parameters()))
    if abs(expected - restored) > 1e-3 * max(1.0, abs(expected)):
        raise RuntimeError(f"Weight mismatch: checkpoint {expected:.4f}, network {restored:.4f}")


def run_dqn_mlp(file_path, model_path, seed_list, global_max_vals):
    """
    Run the MLP+DQN model and generate a set of solutions with several random seeds.
    To obtain a diverse approximate Pareto set, softmax sampling (temperature = 1.0) is used instead of a greedy policy.
    global_max_vals is (max_jobs, max_machines, max_total_ops, max_proc_time,
    max_due_date, max_capability) and every reinforcement learning method shares the same convention.
    """
    if DQN_Agent is None or DQNEnv is None:
        raise ImportError("The MLP+DQN module was not imported")
    global_max_jobs, global_max_machines, _, _, _, _ = global_max_vals

    # Read the instance with the DQN specific reader
    data = read_dqn_instance(file_path)
    # Build the instance wrapper
    file_name = os.path.basename(file_path)
    size = data.size if hasattr(data, 'size') else "unknown"
    inst = DQNJobInstance(data, global_max_jobs, global_max_machines, file_name, size)

    # Create the configuration (using the global maximum values)
    cfg = DQNConfig()
    cfg.max_jobs = global_max_jobs
    cfg.max_machines = global_max_machines
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model_path = model_path  # the checkpoint that the explicit restore below loads
    # Synchronise the model directory: model_dir in DQNConfig is a relative path and using it directly would create
    # directories outside the current working directory (or even fail with a permission error), so an absolute directory consistent with model_path is used instead
    cfg.model_dir = os.path.dirname(model_path)

    # Create the agent and restore the checkpoint explicitly.  The constructor only builds the network,
    # so the checkpoint has to be restored here; the weights are verified afterwards because a restore
    # that silently does nothing is invisible in the reported metrics.
    agent = DQN_Agent(cfg)
    if not os.path.exists(model_path):
        print(f"Warning: model file {model_path} not found; MLP+DQN may fail.")
        return []
    try:
        agent._load_model(model_path)
        _verify_dqn_policy(agent, model_path)
        agent.policy_net.eval()
        print(f"Loaded the MLP+DQN model: {model_path}")
    except Exception as e:
        print(f"Warning: cannot load the MLP+DQN model: {e}")
        return []

    objectives_list = []
    # The state of this baseline is normalised during training, so the evaluation has to normalise
    # with the statistics that were stored next to the weights.  Feeding the raw state to the
    dqn_state_norm = None
    dqn_saved_norm = getattr(agent, 'checkpoint_norm', None)
    if DQNStateNorm is not None and dqn_saved_norm:
        dqn_state_norm = DQNStateNorm(shape=(agent.state_dim,), device=cfg.device)
        dqn_state_norm.mean = torch.tensor(dqn_saved_norm['norm_mean'], device=cfg.device,
                                           dtype=torch.float32)
        dqn_state_norm.var = torch.tensor(dqn_saved_norm['norm_var'], device=cfg.device,
                                          dtype=torch.float32)
        dqn_state_norm.count = float(dqn_saved_norm.get('norm_count', 1.0))
        print(f"Restored the MLP+DQN state normalisation (count={dqn_state_norm.count:.1f})")
    temperature = 1.0  # softmax temperature controlling the randomness
    runs = [(seed, False) for seed in seed_list]
    if seed_list:
        runs.append((seed_list[0], True))
    # The greedy rollout is appended after the sampled ones: greedy decoding is the
    # canonical inference of a policy and it dominates the sampled trajectories, so the
    for seed, deterministic in runs:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = DQNEnv(cfg, inst)
        state = env.reset()
        done = False
        while not done:
            # The action of this baseline is the dispatching rule it applies, not the (operation,
            # machine) pair: the value network ranks the nine rules of its action space and the
            legal, rule_cost = DQN_LIB.rule_cost_table(env)
            if len(legal) == 0:
                break
            mask = np.ones(DQN_LIB.N_RULES, dtype=np.float32)
            state_t = torch.from_numpy(state).float().to(cfg.device).unsqueeze(0)
            if dqn_state_norm is not None:
                state_t = dqn_state_norm.normalize(state_t)
            mask_t = torch.from_numpy(mask).float().to(cfg.device)
            with torch.no_grad():
                q_values = agent.policy_net(state_t).squeeze(0)
                q_values_masked = q_values.clone()
                q_values_masked[mask_t == 0] = -float('inf')
                # Sample with a temperature-scaled softmax
                probs = torch.softmax(q_values_masked / temperature, dim=-1)
                if deterministic:
                    # Greedy decoding on the extra deterministic pass
                    action = int(torch.argmax(q_values_masked).item())
                elif torch.isnan(probs).any():
                    # Handle the corner case: when every probability is NaN, pick a valid action uniformly
                    action = int(torch.randint(DQN_LIB.N_RULES, (1,)).item())
                else:
                    action = torch.multinomial(probs, 1).item()
            # Resolve the chosen rule into a legal action: the rule picks the legal pair with the
            # smallest cost under that rule, exactly as the training decode of this baseline does.
            action = int(legal[int(np.argmin(rule_cost[:, action]))])
            next_state, _, done, _ = env.step(action)
            state = next_state

        # Extract the metrics from the environment
        cmax = max(env.machine_avail_time[:inst.num_machines])
        loads = env.machine_load[:inst.num_machines]
        avg_load = np.mean(loads)
        lb = np.sqrt(np.mean((loads - avg_load)**2))
        tardy = sum(max(0, env.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))
        objectives_list.append([float(cmax), float(lb), float(tardy)])

    return objectives_list

def run_ppo_mlp(file_path, model_path, seed_list, global_max_vals):
    """Run the MLP+PPO model and return the objective values of several solutions (see run_dqn_mlp for the meaning of global_max_vals)"""
    if PPOAgent is None or PPOEnv is None:
        raise ImportError("The MLP+PPO module was not imported")
    global_max_jobs, global_max_machines, _, _, _, _ = global_max_vals

    # Use the PPO_MLP specific reader
    jobs, machine_ids, due_dates = read_ppo_mlp_instance(file_path)
    file_name = os.path.basename(file_path)
    size = "unknown"
    inst = PPOJobInstance(jobs, machine_ids, due_dates, global_max_jobs, global_max_machines, file_name, size)

    cfg = PPOConfig()
    cfg.max_jobs = global_max_jobs
    cfg.max_machines = global_max_machines
    state_dim = 2 * cfg.max_machines + 3 * cfg.max_jobs
    action_dim = cfg.max_jobs * cfg.max_machines

    policy = PPOPolicy(state_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(PPO_DEVICE)
    value = PPOValue(state_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(PPO_DEVICE)

    if not os.path.exists(model_path):
        print(f"Warning: model file {model_path} not found; MLP+PPO may fail.")
        return []
    try:
        checkpoint = torch.load(model_path, map_location=PPO_DEVICE, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value.load_state_dict(checkpoint['value_state_dict'])
        policy.eval()
        value.eval()
        print(f"Loaded the MLP+PPO model: {model_path}")
    except Exception as e:
        print(f"Warning: cannot load the MLP+PPO model: {e}")
        return []

    objectives_list = []
    runs = [(seed, False) for seed in seed_list]
    if seed_list:
        runs.append((seed_list[0], True))
    # The greedy rollout is appended after the sampled ones: greedy decoding is the
    # canonical inference of a policy and it dominates the sampled trajectories, so the
    for seed, deterministic in runs:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = PPOEnv(cfg, inst)
        state = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            state_t = torch.from_numpy(state).to(PPO_DEVICE).float().unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(PPO_DEVICE).bool().unsqueeze(0)
            with torch.no_grad():
                probs, _ = policy(state_t, mask_t)
                dist = torch.distributions.Categorical(probs=probs)
                # Greedy decoding on the extra deterministic pass
                action_id = int(torch.argmax(probs, dim=-1).item()) if deterministic else dist.sample().item()
            next_state, _, done, _ = env.step(action_id)
            state = next_state

        # Extract the metrics from the environment
        cmax = max(env.machine_avail_time[:inst.num_machines])
        loads = env.machine_load[:inst.num_machines]
        avg_load = np.mean(loads)
        lb = np.sqrt(np.mean((loads - avg_load)**2))
        tardy = sum(max(0, env.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))
        objectives_list.append([float(cmax), float(lb), float(tardy)])

    return objectives_list

def run_ppo_gat(file_path, model_path, seed_list, global_max_vals):
    """Run the GAT+PPO model and return the objective values of several solutions (see run_dqn_mlp for the meaning of global_max_vals)"""
    if HomogeneousGAT is None or GATEnv is None:
        raise ImportError("The GAT+PPO module was not imported")
    global_max_jobs, global_max_machines, _, global_max_proc_time, global_max_due_date, _ = global_max_vals

    # Use the PPO_GAT specific reader
    jobs, machine_ids, due_dates = read_ppo_gat_instance(file_path)
    file_name = os.path.basename(file_path)
    size = "unknown"
    inst = GATInstance(jobs, machine_ids, due_dates, file_name, size)

    # Initialise the network (identical to training)
    cfg = GATConfig()
    node_dim = 8
    edge_dim = 1
    gat = HomogeneousGAT(node_dim, edge_dim, cfg.gat_hidden_dim, cfg.gat_out_dim,
                         num_layers=cfg.gat_num_layers, num_heads=cfg.gat_num_heads).to(GAT_DEVICE)
    policy = GATPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim, temperature=0.5).to(GAT_DEVICE)
    value = GATValue(cfg.gat_out_dim, cfg.policy_hidden_dim).to(GAT_DEVICE)

    if not os.path.exists(model_path):
        print(f"Warning: model file {model_path} not found; GAT+PPO may fail.")
        return []
    try:
        checkpoint = torch.load(model_path, map_location=GAT_DEVICE, weights_only=False)
        gat.load_state_dict(checkpoint['gat'])
        policy.load_state_dict(checkpoint['policy'])
        value.load_state_dict(checkpoint['value'])
        gat.eval()
        policy.eval()
        value.eval()
        print(f"Loaded the GAT+PPO model: {model_path}")
    except Exception as e:
        print(f"Warning: cannot load the GAT+PPO model: {e}")
        return []

    # State normalisation: training used use_state_norm=True, so the policy input is the GAT global embedding after
    # TorchRunningMeanStd standardisation; this statistic is stored together with the checkpoint. Evaluation must
    # reuse the same mean/variance/count, otherwise the input scale seen by the policy differs from training and the results are not comparable.
    state_norm = None
    if GATConfig.use_state_norm and 'state_norm_mean' in checkpoint:
        norm_mean = checkpoint.get('state_norm_mean')
        norm_var = checkpoint.get('state_norm_var')
        if norm_mean is not None and norm_var is not None:
            state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=GAT_DEVICE)
            state_norm.mean = norm_mean.to(GAT_DEVICE).float()
            state_norm.var = norm_var.to(GAT_DEVICE).float()
            state_norm.count = float(checkpoint.get('state_norm_count', 1.0))
            print(f"Restored the GAT+PPO state normalisation (count={state_norm.count:.1f})")

    objectives_list = []
    runs = [(seed, False) for seed in seed_list]
    if seed_list:
        runs.append((seed_list[0], True))
    # The greedy rollout is appended after the sampled ones: greedy decoding is the
    # canonical inference of a policy and it dominates the sampled trajectories, so the
    for seed, deterministic in runs:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = GATEnv(inst, global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date, cfg)
        state = env.reset()
        done = False
        while not done:
            # Prepare the input
            state_gpu = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state_gpu[k] = v.to(GAT_DEVICE)
                else:
                    state_gpu[k] = v
            with torch.no_grad():
                global_emb, node_embs = gat(state_gpu)
                global_emb = global_emb.float()
                if state_norm is not None:
                    emb_norm = state_norm.normalize(global_emb.unsqueeze(0)).squeeze(0)
                else:
                    # Stay consistent with the training code when state_norm is missing (L2 normalisation),
                    # avoiding feeding an unnormalised embedding to a policy trained on standardised inputs
                    emb_norm = global_emb / (global_emb.norm() + 1e-8)
                mask = state['action_mask'].to(GAT_DEVICE)
                # The joint head scores an action from the embedding of its operation node and of its
                # machine node, so the node embeddings of the graph are passed on with the two index
                # vectors the state carries for the action grid.
                probs = policy(emb_norm, node_embs[state_gpu['op_node_idx']],
                               node_embs[state_gpu['mac_node_idx']], mask)
                dist = torch.distributions.Categorical(probs=probs)
                # Greedy decoding on the extra deterministic pass
                action = int(torch.argmax(probs, dim=-1).item()) if deterministic else dist.sample().item()
            next_state, _, done, _ = env.step(action)
            state = next_state

        # Extract the metrics from the environment
        cmax = env.current_time
        loads = env.machine_load.cpu().numpy() if torch.is_tensor(env.machine_load) else env.machine_load
        avg_load = np.mean(loads)
        lb = np.sqrt(np.mean((loads - avg_load)**2))
        tardy = 0
        for j in range(inst.n_jobs):
            due = inst.due_dates[j]
            comp = env.job_completion_time[j].item() if torch.is_tensor(env.job_completion_time[j]) else env.job_completion_time[j]
            tardy += max(0, comp - due)
        objectives_list.append([float(cmax), float(lb), float(tardy)])

    return objectives_list

def run_ppo_hgat(file_path, model_path, seed_list, global_max_vals):
    """Run the HeteroGAT+PPO model and return the objective values of several solutions (see run_dqn_mlp for the meaning of global_max_vals)"""
    if HGATInstance is None or HierarchicalPPOAgent is None:
        raise ImportError("The HGAT+PPO module was not imported")
    (global_max_jobs, global_max_machines, global_max_ops,
     global_max_proc_time, global_max_due_date, global_max_capability) = global_max_vals

    # Use the generic reader (the PPO_HGAT reader returns 5 values)
    jobs, machine_capabilities, due_dates, file_name, size = read_ppo_instance(file_path)

    # Create the instance object, which requires the global maximum values; the machine capability must be normalised by the global maximum,
    # otherwise the machine node features fall outside the training distribution and the proposed method cannot be compared fairly
    inst = HGATInstance(jobs, machine_capabilities, due_dates, file_name, size,
                        global_max_proc_time, global_max_capability)

    # Initialise the network with the width stored in the selected checkpoint, so the
    # evaluation follows the 3/3, 5/3 or 7/4 feature table that model was trained with.
    checkpoint = torch.load(model_path, map_location=HGAT_DEVICE)
    dim_op = int(checkpoint["gat"]["l1.W_op.weight"].shape[1])
    dim_mac = int(checkpoint["gat"]["l1.W_mac.weight"].shape[1])
    feature_mode = {3: "legacy", 5: "augmented", 7: "full"}.get(dim_op, "full")
    hidden = HGATConfig.gat_hidden_dim
    out = HGATConfig.gat_out_dim
    policy_hidden = HGATConfig.policy_hidden_dim

    gat = HeteroGAT(dim_op, dim_mac, hidden, out).to(HGAT_DEVICE)
    upper = UpperPolicy(out, policy_hidden).to(HGAT_DEVICE)
    lower = LowerPolicy(out, policy_hidden).to(HGAT_DEVICE)
    critic_u = Critic(out, policy_hidden).to(HGAT_DEVICE)
    critic_l = Critic(out, policy_hidden).to(HGAT_DEVICE)

    # The state normaliser of the training run, restored from the checkpoint exactly as the GAT
    # baseline and the DQN baseline of this experiment restore their own.  The agent only restores
    hgat_state_norm = (TorchRunningMeanStd(shape=(HGATConfig.gat_out_dim,), device=HGAT_DEVICE)
                       if getattr(HGATConfig, "use_state_norm", False) else None)
    agent = HierarchicalPPOAgent(inst, gat, upper, lower, critic_u, critic_l, HGATConfig,
                                hgat_state_norm)
    if not os.path.exists(model_path):
        print(f"Warning: model file {model_path} not found; HGAT+PPO may fail.")
        return []
    try:
        agent.load(model_path)
        print(f"Loaded the HGAT+PPO model: {model_path}")
    except Exception as e:
        print(f"Warning: cannot load the HGAT+PPO model: {e}")
        return []
    agent.gat.eval()
    agent.upper.eval()
    agent.lower.eval()

    objectives_list = []
    runs = [(seed, False) for seed in seed_list]
    if seed_list:
        runs.append((seed_list[0], True))
    # The greedy rollout is appended after the sampled ones: greedy decoding is the
    # canonical inference of a policy and it dominates the sampled trajectories, so the
    for seed, deterministic in runs:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # The environment is built with the global maximum values (exactly as in training and in the ablation experiment): the job index in the node features
        # normalisation and the action mask size both depend on global_max_jobs/global_max_ops/global_max_machines,
        # so using the scale of the instance itself would be inconsistent with the training input distribution
        env = HGATEnv(inst, global_max_jobs, global_max_ops, global_max_machines,
                      global_max_proc_time, global_max_due_date, HGATConfig,
                      feature_mode=feature_mode)
        state = env.reset()
        done = False
        while not done:
            actions, _, _ = agent.get_action_batch([state], [inst], deterministic=deterministic)
            action_tuple = actions[0]
            next_state, _, done = env.step(action_tuple)
            state = next_state

        # Extract the results from the environment attributes
        job_completion = env.job_completion_time.cpu().numpy()
        machine_loads = env.machine_load.cpu().numpy()
        cmax = float(job_completion.max())
        lb = float(np.std(machine_loads))
        tardy = float(np.sum(np.maximum(0, job_completion - np.array(due_dates))))
        objectives_list.append([cmax, lb, tardy])

    return objectives_list


# Added comparison methods: single-level-policy reinforcement learning baselines (HGNN / DAN / RESCHED)
# The three share the same environment and reward as the proposed method and differ only in the encoder and the decision structure.

def run_hgnn(file_path, model_path, seed_list, global_max_vals):
    """Run the HGNN baseline (single-level policy that directly picks an operation-machine pair) and return the objective values of several solutions"""
    if hgnn_run_scheduler is None:
        raise ImportError("The HGNN module was not imported")
    return hgnn_run_scheduler(file_path, model_path, seed_list, global_max_vals)


def run_dan(file_path, model_path, seed_list, global_max_vals):
    """Run the DAN baseline (dual-attention encoder + single-level policy) and return the objective values of several solutions"""
    if dan_run_scheduler is None:
        raise ImportError("The DAN module was not imported")
    return dan_run_scheduler(file_path, model_path, seed_list, global_max_vals)


def run_resched(file_path, model_path, seed_list, global_max_vals):
    """Run the RESCHED baseline (Transformer encoder over a simplified state + single-level policy) and return the objective values of several solutions"""
    if resched_run_scheduler is None:
        raise ImportError("The RESCHED module was not imported")
    return resched_run_scheduler(file_path, model_path, seed_list, global_max_vals)


# Export the results to Excel

def export_experiment2_to_excel(exp2_results, filename="Experiment2_Result.xlsx", stats=None):
    """
    Save the results to an Excel file with three worksheets: small, medium and large scale.
    The layout of each worksheet matches Tables 26, 27 and 28 of the experiment report exactly.
    Number format: mean F, GAP, GD and SP keep two decimals, HV keeps two decimals in scientific notation and the total runtime keeps three decimals.
    When stats is (pairwise_rows, friedman_rows) two extra statistical significance worksheets are written.
    """
    if not HAS_PANDAS:
        print("pandas is not installed; Excel export is skipped.")
        return

    # Define the scale categories (consistent with the experiment report)
    scale_categories = {
        'Small': ['10x5', '15x10', '20x10'],
        'Medium': ['25x15', '30x20', '40x20'],
        'Large': ['50x25', '75x25', '100x30']
    }

    # Method list (order consistent with the experiment report, keeping only the methods that actually ran successfully)
    methods = [m for m in METHOD_ORDER if any(m in exp2_results.get(s, {}) for s in
                                                ['10x5', '15x10', '20x10', '25x15', '30x20', '40x20',
                                                 '50x25', '75x25', '100x30'])]

    # Metric names (consistent with the experiment report headers, extended with the standard deviation and the 95% confidence interval as requested by the reviewers)
    metrics = ['Mean F', 'F SD', 'F 95% CI', 'GAP(%)', 'HV', 'GD', 'SP', 'Total time (s)']

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for category, scales in scale_categories.items():
            # Build the multi-level column index: the first level is the scale, the second level is the metric
            columns = pd.MultiIndex.from_product([scales, metrics], names=['Scale', 'Metric'])
            # Create an empty DataFrame whose row index holds the method names
            df = pd.DataFrame(index=methods, columns=columns)

            # Fill in the raw values (used for the later formatting)
            for scale in scales:
                if scale not in exp2_results:
                    continue
                scale_data = exp2_results[scale]
                for method in methods:
                    if method in scale_data:
                        data = scale_data[method]
                        df.loc[method, (scale, 'Mean F')] = data['avg_F']
                        # The standard deviation and the 95% confidence interval half-width are computed from the per-instance weighted objective values
                        _, f_std, f_half = mean_std_ci95(data.get('F_list', []))
                        df.loc[method, (scale, 'F SD')] = f_std
                        df.loc[method, (scale, 'F 95% CI')] = f_half
                        df.loc[method, (scale, 'GAP(%)')] = data['GAP']
                        df.loc[method, (scale, 'HV')] = data['HV']
                        df.loc[method, (scale, 'GD')] = data['GD']
                        df.loc[method, (scale, 'SP')] = data['SP']
                        df.loc[method, (scale, 'Total time (s)')] = data['time']

            # Format the displayed values
            df_display = df.copy()
            for scale in scales:
                for metric in metrics:
                    col = (scale, metric)
                    if metric == 'HV':
                        df_display[col] = df[col].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else "")
                    elif metric == 'Total time (s)':
                        df_display[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
                    else:
                        df_display[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
            # Write to Excel
            df_display.to_excel(writer, sheet_name=category)

        # Write the statistical significance results (paired Wilcoxon + Friedman) into a separate worksheet
        if stats is not None:
            pairwise_rows, friedman_rows = stats
            if pairwise_rows:
                pd.DataFrame(pairwise_rows).to_excel(writer, sheet_name='Paired Wilcoxon test', index=False)
            if friedman_rows:
                pd.DataFrame(friedman_rows).to_excel(writer, sheet_name='Friedman test', index=False)

    print(f"\nResults exported to {filename}")


# Statistical significance tests
#

def _aligned_metric(scale_data, method, list_key):
    """
    Convert the per-instance metrics of a method on a scale into {instance name: metric value}.

    New experiment results always record the instance name, so the paired tests align strictly by instance; when a result without
    instance names is read, the code falls back to alignment by positional index.
    """
    data = scale_data.get(method, {})
    values = list(data.get(list_key, []))
    names = list(data.get('names', []))
    if names and len(names) == len(values):
        return {name: value for name, value in zip(names, values)}
    return {idx: value for idx, value in enumerate(values)}


def _common_instance_keys(metric_maps, methods):
    """Take the instance names shared by all methods, ordered according to the first method in methods."""
    if not methods:
        return []
    reference_map = metric_maps[methods[0]]
    return [key for key in reference_map if all(key in metric_maps[m] for m in methods)]


def _drop_nan_pairs(sample_a, sample_b):
    """Drop paired missing values (for example NaN for GD when the true front is unavailable) and return two arrays of equal length."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def paired_wilcoxon(sample_a, sample_b):
    """
    Paired Wilcoxon signed-rank test; returns (statistic, p-value).
    Returns (nan, nan) / (nan, 1.0) when the sample is too small or all differences are 0.
    """
    if not HAS_SCIPY:
        return np.nan, np.nan
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return np.nan, np.nan
    a, b = a[:n], b[:n]
    if np.allclose(a, b):
        return np.nan, 1.0
    try:
        statistic, p_value = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(statistic), float(p_value)
    except Exception:
        return np.nan, np.nan


def cliffs_delta(sample_a, sample_b):
    """
    Cliff's delta effect size, ranging in [-1, 1]. Common thresholds for |delta|: <0.147 negligible,
    <0.33 small, <0.474 medium, otherwise large. A positive value means sample a is overall larger.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    greater = float(np.sum(diff > 0))
    less = float(np.sum(diff < 0))
    return (greater - less) / (len(a) * len(b))


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction; returns the corrected p-values in the same order as the input."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    adjusted = np.full(m, np.nan)
    if m == 0:
        return adjusted
    order = np.argsort(np.where(np.isnan(p_values), np.inf, p_values))
    running_max = 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p_values[idx]):
            continue
        value = min(1.0, (m - rank) * float(p_values[idx]))
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def friedman_analysis(sample_matrix):
    """
    Friedman omnibus test. sample_matrix has shape (number of instances, number of methods) and holds the metric value of
    each method on the corresponding instance. Returns (Friedman statistic, Friedman p, Iman-Davenport F, Iman-Davenport p, mean ranks).
    """
    if not HAS_SCIPY:
        return np.nan, np.nan, np.nan, np.nan, []
    data = np.asarray(sample_matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        return np.nan, np.nan, np.nan, np.nan, []
    # Drop instances that contain missing values
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < 2:
        return np.nan, np.nan, np.nan, np.nan, []

    n_instances, n_methods = data.shape
    ranks = np.apply_along_axis(rankdata, 1, data)
    average_ranks = ranks.mean(axis=0)

    try:
        statistic, p_value = friedmanchisquare(*[data[:, j] for j in range(n_methods)])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, average_ranks

    # Iman-Davenport correction: an F statistic that is more accurate than the chi-squared approximation
    denominator = n_instances * (n_methods - 1) - statistic
    if denominator > 1e-12:
        f_statistic = (n_instances - 1) * statistic / denominator
        # Use the survival function sf instead of 1-cdf: for large F, 1-cdf saturates at the lower bound (about 1.1e-16) because of floating-point cancellation,
        # which looks like an exact 0 and hides the true p-value
        p_f = float(f_dist.sf(f_statistic, n_methods - 1,
                              (n_methods - 1) * (n_instances - 1)))
    elif denominator > -1e-12:
        # When the method ranking is identical on every instance the Friedman statistic reaches its upper bound,
        # the Iman-Davenport F statistic then diverges and the corresponding p-value tends to 0
        f_statistic, p_f = np.inf, 0.0
    else:
        f_statistic, p_f = np.nan, np.nan
    return float(statistic), float(p_value), float(f_statistic), p_f, average_ranks


def mean_std_ci95(samples):
    """Return (mean, standard deviation, 95% confidence interval half-width) using the t distribution critical value."""
    data = np.asarray(samples, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(data))
    if len(data) > 1:
        std = float(np.std(data, ddof=1))
        half = 0.0
        if HAS_SCIPY:
            half = float(t_dist.ppf(0.975, len(data) - 1) * std / np.sqrt(len(data)))
    else:
        std, half = 0.0, 0.0
    return mean, std, half


def compute_statistical_tests(exp2_results, reference="HeteroGAT+PPO"):
    """
    Produce the statistical test results for every scale and every metric.

    Returns (pairwise_rows, friedman_rows):
        pairwise_rows: per-instance paired test results of the reference method against the other methods
        friedman_rows: omnibus test results of all methods on that scale
    """
    pairwise_rows = []
    friedman_rows = []

    metric_specs = [('F', 'F_list'), ('HV', 'HV_list'), ('GD', 'GD_list'), ('SP', 'SP_list')]

    def scale_sort_key(scale):
        jobs, machines = scale.split('x')
        return int(jobs), int(machines)

    for scale in sorted(exp2_results.keys(), key=scale_sort_key):
        scale_data = exp2_results[scale]
        if reference not in scale_data:
            print(f"Warning: scale {scale} has no baseline {reference}; the test is skipped.")
            continue
        methods_here = [m for m in METHOD_ORDER if m in scale_data]

        for metric_name, list_key in metric_specs:
            # Align the per-instance metrics of the methods by instance name (aligning by index would shift them when a method fails on some instance)
            metric_maps = {m: _aligned_metric(scale_data, m, list_key) for m in methods_here}
            common_keys = _common_instance_keys(metric_maps, methods_here)
            if len(common_keys) < 2:
                continue
            ref_samples = np.asarray([metric_maps[reference][k] for k in common_keys], dtype=float)

            # (1) paired Wilcoxon signed-rank test + Holm-Bonferroni correction
            entries = []
            raw_p_values = []
            for method in methods_here:
                if method == reference:
                    continue
                samples = np.asarray([metric_maps[method][k] for k in common_keys], dtype=float)
                # Drop paired missing values (for example NaN for GD when the true front is unavailable)
                pair_ref, pair_other = _drop_nan_pairs(ref_samples, samples)
                if len(pair_ref) < 2:
                    continue
                statistic, p_value = paired_wilcoxon(pair_ref, pair_other)
                raw_p_values.append(p_value)
                entries.append({
                    'Scale': scale,
                    'Metric': metric_name,
                    'Baseline': method,
                    'Paired samples': len(pair_ref),
                    'W statistic': statistic,
                    'p-value': p_value,
                    'Holm-adjusted p': np.nan,
                    'Significant (0.05)': '',
                    'Cliff delta': cliffs_delta(pair_ref, pair_other),
                    'Baseline mean': float(np.mean(pair_ref)),
                    'Baseline SD': float(np.std(pair_ref, ddof=1)),
                    'Method mean': float(np.mean(pair_other)),
                    'Method SD': float(np.std(pair_other, ddof=1)),
                })
            for entry, p_adjusted in zip(entries, holm_bonferroni(raw_p_values)):
                entry['Holm-adjusted p'] = p_adjusted
                entry['Significant (0.05)'] = 'Yes' if (not np.isnan(p_adjusted) and p_adjusted < 0.05) else 'No'
                pairwise_rows.append(entry)

            # (2) Friedman omnibus test
            if len(methods_here) >= 3:
                transposed = np.array([[metric_maps[m][k] for k in common_keys]
                                       for m in methods_here], dtype=float).T
                # Drop instances missing either value so that the Friedman test uses a complete paired sample
                transposed = transposed[~np.isnan(transposed).any(axis=1)]
                if transposed.shape[0] < 2:
                    continue
                statistic, p_value, f_statistic, p_f, average_ranks = friedman_analysis(transposed)
                friedman_rows.append({
                    'Scale': scale,
                    'Metric': metric_name,
                    'Methods': len(methods_here),
                    'Instances': int(transposed.shape[0]),
                    'Friedman statistic': statistic,
                    'Friedman p': p_value,
                    'Iman-Davenport F': f_statistic,
                    'Iman-Davenport p': p_f,
                    'Mean rank': '/'.join(f"{m}:{r:.2f}" for m, r in zip(methods_here, average_ranks))
                              if len(average_ranks) else '',
                })

    return pairwise_rows, friedman_rows


def print_statistical_tests(pairwise_rows, friedman_rows):
    """Print the statistical significance results to the console."""
    print("\n" + "=" * 120)
    print("Statistical significance results")
    print("=" * 120)

    print("\n[Paired Wilcoxon signed-rank test] HeteroGAT+PPO vs baselines (Holm-Bonferroni)")
    header = (f"{'Scale':<8} | {'Metric':<5} | {'Baseline':<12} | {'W':>10} | {'p-value':>10} | "
              f"{'Holm p':>10} | {'Sig.':<6} | {'Cliff delta':>11}")
    print(header)
    print("-" * len(header))
    for row in pairwise_rows:
        w_str = f"{row['W statistic']:.3f}" if not np.isnan(row['W statistic']) else "NaN"
        p_str = f"{row['p-value']:.4e}" if not np.isnan(row['p-value']) else "NaN"
        padj_str = f"{row['Holm-adjusted p']:.4e}" if not np.isnan(row['Holm-adjusted p']) else "NaN"
        delta_value = row['Cliff delta']
        delta_str = f"{delta_value:.3f}" if not np.isnan(delta_value) else "NaN"
        print(f"{row['Scale']:<8} | {row['Metric']:<5} | {row['Baseline']:<12} | {w_str:>10} | {p_str:>10} | "
              f"{padj_str:>10} | {row['Significant (0.05)']:<6} | {delta_str:>11}")

    print("\n[Friedman test] differences among methods at each scale")
    header2 = (f"{'Scale':<8} | {'Metric':<5} | {'Methods':>5} | {'Instances':>5} | {'Friedman chi2':>12} | "
               f"{'p-value':>10} | {'ID-F':>10} | {'ID-p-value':>10}")
    print(header2)
    print("-" * len(header2))
    for row in friedman_rows:
        chi_str = f"{row['Friedman statistic']:.3f}" if not np.isnan(row['Friedman statistic']) else "NaN"
        p_str = f"{row['Friedman p']:.4e}" if not np.isnan(row['Friedman p']) else "NaN"
        f_str = f"{row['Iman-Davenport F']:.3f}" if not np.isnan(row['Iman-Davenport F']) else "NaN"
        pf_str = f"{row['Iman-Davenport p']:.4e}" if not np.isnan(row['Iman-Davenport p']) else "NaN"
        print(f"{row['Scale']:<8} | {row['Metric']:<5} | {row['Methods']:>5} | {row['Instances']:>5} | "
              f"{chi_str:>12} | {p_str:>10} | {f_str:>10} | {pf_str:>10}")


# Main comparison experiment

def main():
    # Collect the instance files: gather them over the full scale list to compute the global maximum tuple (which must not depend on sharding)
    print("Budget: %d sampled rollouts plus one greedy rollout per instance for learning methods; %d individuals x %d generations = %d evaluations for evolutionary methods"
          % (NUM_RUNS_RL, EVOLUTIONARY_POPULATION, EVOLUTIONARY_GENERATIONS,
             EVOLUTIONARY_POPULATION * EVOLUTIONARY_GENERATIONS))
    all_instance_dict = {}
    for jobs, mach in SIZE_LIST:
        pattern = f"comp_*_{jobs}x{mach}_*.txt"
        files = glob.glob(os.path.join(DATA_DIR, pattern))
        files.sort()
        all_instance_dict[(jobs, mach)] = files

    # Compute the global maximum tuple (shared by every reinforcement learning method and independent of the sharding)
    # Normalisation tuple of the learned methods.  It is computed over the instances the methods were
    train_dir = os.path.join(PROJECT_ROOT, "DataSet", "mo_fjsp_instances")
    train_files = sorted(glob.glob(os.path.join(train_dir, "mo_fjsp_*_train.txt")))
    if train_files:
        global_max_vals = compute_global_maxima({(100, 30): train_files})
    else:
        print("Warning: no training instances found; falling back to the normalisation of the comparison set.")
        global_max_vals = compute_global_maxima(all_instance_dict)
    print(f"\nGlobal maxima: max_jobs={global_max_vals[0]}, max_machines={global_max_vals[1]}, "
          f"max_ops={global_max_vals[2]}, max_proc_time={global_max_vals[3]:.2f}, "
          f"max_due_date={global_max_vals[4]:.2f}, max_capability={global_max_vals[5]}")

    # Scales actually processed now: a sharded run only handles the scales in EXP2_SCALE_LIST, but the normalisation convention is still the global maximum above
    instance_dict = {}
    for jobs, mach in EXP2_SCALE_LIST:
        instance_dict[(jobs, mach)] = all_instance_dict[(jobs, mach)]
        print(f"{jobs}x{mach}: {len(instance_dict[(jobs, mach)])} instances")

    # Define the method list together with the run function and the model path of each method
    methods = [
        ("MOEA/D", run_moead, None),
        ("NSGA-II", run_nsga2, None),
        ("NSGA-III", run_nsga3, None),
        ("RVEA", run_rvea, None),
        ("MLP+DQN", run_dqn_mlp, DQN_MODEL_PATH),
        ("MLP+PPO", run_ppo_mlp, PPO_MLP_MODEL_PATH),
        ("GAT+PPO", run_ppo_gat, PPO_GAT_MODEL_PATH),
        ("HGNN", run_hgnn, HGNN_MODEL_PATH),
        ("DAN", run_dan, DAN_MODEL_PATH),
        ("RESCHED", run_resched, RESCHED_MODEL_PATH),
        ("HeteroGAT+PPO", run_ppo_hgat, PPO_HGAT_MODEL_PATH),
    ]
    # A sharded run only handles the given methods and the other shards cover the rest; after merging the caches the per-scale results are equivalent to a sequential run
    if _SHARD_METHODS:
        methods = [item for item in methods if item[0] in _SHARD_METHODS]
        print(f"Shard run: this process handles {[item[0] for item in methods]}")

    # A learned method that the qualification gate of the wave sent back for training takes no part in
    # the comparison (see the helper above): it is neither measured nor reported, so the tables never
    # hold the numbers of a model that is still being retrained or of one that never learned.
    excluded_methods = excluded_by_qualification()
    dropped = [item[0] for item in methods if item[0] in excluded_methods]
    if dropped:
        methods = [item for item in methods if item[0] not in excluded_methods]
        print(f"Methods excluded by the quality gate: {', '.join(dropped)}")

    # Store the results: results[scale][method] = {
    #    'obj_vals': list of best weighted objectives for each instance,
    results = defaultdict(lambda: defaultdict(
        lambda: {'obj_vals': [], 'all_objectives': [], 'times': [], 'names': []}))

    # Raw result cache: on a cache hit (with an identical instance list) the per-instance solution sets are reused, skipping the expensive policy sampling
    raw_cache = _load_raw_cache()
    cache_hits = 0

    # Run every method
    for (jobs, mach), file_list in instance_dict.items():
        scale_key = f"{jobs}x{mach}"
        if not file_list:
            continue
        print(f"\n===== Scale {scale_key} =====")
        file_names = [os.path.basename(p) for p in file_list]

        for method_name, run_func, model_path in methods:
            # Check whether the method is available
            if run_func is None:
                print(f"  skipping {method_name} (module not imported)")
                continue
            # Export-only pass: hydrate every method from the raw cache and skip the solvers,
            # so that a change limited to the post-processing never resamples a policy.
            if EXPORT_ONLY:
                cached = raw_cache.get(scale_key, {}).get(method_name)
                if cached is None or list(cached.get("names", [])) != file_names:
                    print(f"  skipping {method_name} (no matching cache in export mode)")
                    continue
                results[scale_key][method_name]['obj_vals'] = [float(v) for v in cached['obj_vals']]
                results[scale_key][method_name]['all_objectives'] = \
                    [[list(map(float, objs)) for objs in inst_objs]
                     for inst_objs in cached['all_objectives']]
                results[scale_key][method_name]['times'] = [float(cached['time'])]
                results[scale_key][method_name]['names'] = list(cached['names'])
                cache_hits += 1
                continue
            # The reinforcement learning methods rely on trained model files; when one is missing a clear message is printed instead of failing instance by instance
            if model_path is not None and not os.path.exists(model_path):
                print(f"  skipping {method_name} (model file {model_path} not found; run its training entry first)")
                continue

            # Cache hit: when the instance list matches the cache exactly the solutions are reused and the policy is not run again
            cached = raw_cache.get(scale_key, {}).get(method_name)
            if (cached is not None and list(cached.get("names", [])) == file_names
                    and provenance_matches(cached, model_path)):
                print(f"  reusing the cache of {method_name} ...")
                results[scale_key][method_name]['obj_vals'] = [float(v) for v in cached['obj_vals']]
                results[scale_key][method_name]['all_objectives'] = \
                    [[list(map(float, objs)) for objs in inst_objs]
                     for inst_objs in cached['all_objectives']]
                results[scale_key][method_name]['times'] = [float(cached['time'])]
                results[scale_key][method_name]['names'] = list(cached['names'])
                cache_hits += 1
                continue

            print(f"  running {method_name} ...")
            total_time = 0
            obj_vals = []
            all_obj = []
            inst_names = []
            for fpath in file_list:
                try:
                    start = time.time()
                    if method_name in EVOLUTIONARY_METHODS:
                        objectives_list = run_func(fpath)
                    else:
                        # The reinforcement learning methods receive the global maximum tuple so that every method normalises identically
                        objectives_list = run_func(fpath, model_path, RL_SEEDS, global_max_vals)
                    elapsed = time.time() - start

                    if len(objectives_list) == 0:
                        continue
                    w_objs = [weighted_objective(obj) for obj in objectives_list]
                    best_idx = np.argmin(w_objs)
                    obj_vals.append(w_objs[best_idx])
                    all_obj.append(objectives_list)
                    inst_names.append(os.path.basename(fpath))
                    total_time += elapsed
                except Exception as e:
                    print(f"      instance {os.path.basename(fpath)} failed: {e}")
                    continue

            if obj_vals:
                results[scale_key][method_name]['obj_vals'] = obj_vals
                results[scale_key][method_name]['all_objectives'] = all_obj
                results[scale_key][method_name]['times'].append(total_time)
                results[scale_key][method_name]['names'] = inst_names

                # Write the cache: the per-instance solution sets are the only input of HV/GD/SP and of the statistical tests, so after caching even a change
                # limited to the post-processing logic can reuse them without rerunning the expensive policy sampling
                raw_cache.setdefault(scale_key, {})[method_name] = {
                    'names': list(inst_names),
                    'obj_vals': [float(v) for v in obj_vals],
                    'all_objectives': [[[float(x) for x in objs] for objs in inst_objs]
                                       for inst_objs in all_obj],
                    'time': float(total_time),
                    'provenance': (fingerprint(model_path) if model_path is not None
                                   else evolutionary_provenance()),
                }
                _save_raw_cache(raw_cache)


    print(f"\n{cache_hits} cached (scale, method) pairs; the rest were rerun and written to {EXP2_CACHE_PATH}")

    # Compute the average metrics of each method for every scale category and collect them for the export

    print("\n" + "="*120)
    print("Comparison experiment 2: summary")
    print("="*120)

    # Dictionary holding the data to export
    exp2_results = defaultdict(dict)  # exp2_results[scale][method] = {avg_F, GAP, HV, GD, SP, time}

    for scale in sorted(results.keys(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
        print(f"\n【Scale {scale}】")

        # Collect all solutions of every instance at this scale to determine the global reference point
        all_objs_scale = []
        for method_data in results[scale].values():
            for inst_objs in method_data['all_objectives']:
                all_objs_scale.extend(inst_objs)
        if all_objs_scale:
            all_objs_scale = np.array(all_objs_scale)
            ref_point = np.max(all_objs_scale, axis=0) * 1.1
        else:
            ref_point = np.array([1000, 1000, 1000])

        # Group the solutions of every method by instance name and compute the true front of each instance: when a method fails on some instance,
        # the per-instance lists have different lengths and aligning by index would shift GD and the statistical tests
        inst_all_objs = defaultdict(list)
        for method_data in results[scale].values():
            for name, inst_objs in zip(method_data['names'], method_data['all_objectives']):
                inst_all_objs[name].extend(inst_objs)

        true_frontiers = {}
        for name, objs in inst_all_objs.items():
            true_idx = nondominated_sort(objs) if objs else []
            true_frontiers[name] = [objs[i] for i in true_idx] if true_idx else []

        avg_F = {}
        avg_gap = {}
        avg_hv = {}
        avg_gd = {}
        avg_sp = {}
        total_time = {}
        # Store the per-instance metrics per method so that they are never mixed with the results of other methods
        per_method_metrics = {}

        for method_name, _, _ in methods:
            if method_name not in results[scale]:
                continue
            data = results[scale][method_name]
            obj_vals = data['obj_vals']
            all_objs = data['all_objectives']
            time_vals = data['times']
            inst_names = data['names']
            if not obj_vals:
                continue

            avg_F[method_name] = np.mean(obj_vals)
            total_time[method_name] = np.sum(time_vals) if time_vals else 0

            hv_list = []
            gd_list = []
            sp_list = []

            for name, inst_objs in zip(inst_names, all_objs):
                if len(inst_objs) == 0:
                    continue
                # Compute HV/GD/SP uniformly on the obtained non-dominated front: the evolutionary methods already return their non-dominated
                # front, whereas the reinforcement learning methods return every sampled solution; without filtering, the two families would
                # not share the same convention (HV is unaffected, but dominated solutions inflate GD/SP noticeably).
                front_idx = nondominated_sort(inst_objs)
                front = [inst_objs[i] for i in front_idx] if front_idx else list(inst_objs)
                hv = compute_hypervolume(front, ref_point)
                hv_list.append(hv)
                sp = compute_sp(front)
                if (not np.isfinite(sp) and method_name not in EVOLUTIONARY_METHODS
                        and len(inst_objs) > 1):
                    # Same convention as the ablation: when the deterministic greedy decode
                    # dominates the sampled front the spacing is measured on the sampled front.
                    sampled = inst_objs[1:]
                    sampled_idx = nondominated_sort(sampled)
                    sp = compute_sp([sampled[i] for i in sampled_idx] if sampled_idx else sampled)
                    if not np.isfinite(sp):
                        # Even the sampled front is a single point: report the measured
                        # dispersion of the sampled solutions instead of a missing value.
                        sp = compute_sp(sampled)
                sp_list.append(sp)
                true_pf = true_frontiers.get(name, [])
                if len(true_pf) > 0:
                    gd = compute_gd(front, true_pf)
                    gd_list.append(gd)
                else:
                    gd_list.append(np.nan)

            avg_hv[method_name] = np.mean(hv_list) if hv_list else np.nan
            avg_gd[method_name] = np.mean(gd_list) if gd_list else np.nan
            # SP is undefined when a method returns a single distinct solution; mixing those
            # cells in as zeros made a collapsed front look perfectly uniform.
            defined_sp = [value for value in sp_list if np.isfinite(value)]
            avg_sp[method_name] = float(np.mean(defined_sp)) if defined_sp else np.nan

            # The per-instance metrics (with instance names) are stored together with the method for the statistical tests and confidence intervals
            per_method_metrics[method_name] = {
                'names': list(inst_names),
                'F_list': list(obj_vals),
                'HV_list': list(hv_list),
                'GD_list': list(gd_list),
                'SP_list': list(sp_list),
            }

        # Compute GAP with HeteroGAT+PPO as the reference
        baseline = avg_F.get("HeteroGAT+PPO", np.nan)
        for method_name in avg_F.keys():
            if method_name == "HeteroGAT+PPO":
                avg_gap[method_name] = 0.0
            else:
                avg_gap[method_name] = (avg_F[method_name] - baseline) / baseline * 100 if baseline != 0 else np.nan

        # Store into exp2_results for the export
        for method_name in avg_F.keys():
            exp2_results[scale][method_name] = {
                'avg_F': avg_F[method_name],
                'GAP': avg_gap[method_name],
                'HV': avg_hv[method_name],
                'GD': avg_gd[method_name],
                'SP': avg_sp[method_name],
                'time': total_time[method_name],
            }
            # Per-instance metrics (with instance names), used by the statistical significance tests and the confidence intervals
            exp2_results[scale][method_name].update(per_method_metrics.get(method_name, {}))

        # Print the results (console)
        header = (f"{'Method':<12} | {'Mean F':>10} | {'GAP(%)':>7} | {'HV':>10} | {'GD':>10} | {'SP':>10} | {'Total time (s)':>10}")
        print(header)
        print("-" * len(header))

        for method_name, _, _ in methods:
            if method_name not in avg_F:
                continue
            # Simple console formatting keeping two decimals (except for HV)
            hv_str = f"{avg_hv[method_name]:.4f}" if not np.isnan(avg_hv[method_name]) else "NaN"
            gd_str = f"{avg_gd[method_name]:.4f}" if not np.isnan(avg_gd[method_name]) else "NaN"
            sp_str = f"{avg_sp[method_name]:.4f}" if not np.isnan(avg_sp[method_name]) else "NaN"
            print(f"{method_name:<12} | {avg_F[method_name]:>10.2f} | {avg_gap.get(method_name, np.nan):>7.2f} | "
                  f"{hv_str:>10} | {gd_str:>10} | {sp_str:>10} | {total_time.get(method_name, 0):>10.2f}")

    print("="*120)

    # Statistical significance tests (paired Wilcoxon signed-rank test + Friedman omnibus test)
    pairwise_rows, friedman_rows = compute_statistical_tests(exp2_results)
    print_statistical_tests(pairwise_rows, friedman_rows)

    # A sharded run only writes its per-instance solution sets into its own cache; the final metric aggregation, statistical tests and Excel export
    # are performed by the full run once all scales and methods are available, so that a shard never overwrites the final results with an incomplete set
    if _SHARD_SCALES or _SHARD_METHODS:
        print(f"\nShard run: per-instance sets written to {EXP2_CACHE_PATH}; Excel export skipped.")
        return

    # Export the results to Excel (including the statistical test worksheets)
    # Every result workbook is written under Figure_And_File
    result_dir = os.path.join(PROJECT_ROOT, "Figure_And_File", "Compare_Experiment")
    os.makedirs(result_dir, exist_ok=True)
    export_experiment2_to_excel(exp2_results,
                                filename=os.path.join(result_dir, "Experiment2_Result.xlsx"),
                                stats=(pairwise_rows, friedman_rows))

if __name__ == "__main__":
    main()



