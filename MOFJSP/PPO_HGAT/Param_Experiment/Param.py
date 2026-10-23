"""
Parameter Sensitivity Detection Script (Optimized Version)
Based on the trained best model, scans the following parameters:
- Multi-objective weights w (w1, w2, w3)
- KL penalty coefficient beta
- Discount factor gamma
- Number of attention heads K

For each parameter value, retrains the model (with settable number of training epochs), evaluates the average weighted objective value and sub-objectives on the test set,
and records the number of training epochs required for convergence. Finally, plots curves and outputs a table.

The baseline of every scan is the recipe this project trains the method with, and the grid is
anchored on the values the earlier waves measured on this host (the discount of one, the KL
penalty band of the current configuration, K=4), not on the values of the manuscript: the
parameter selection and the reading of these curves follow the experiments of this round.
The graph representation and the assignment score of this script have to be the ones of
PPO_HGAT.PPO_HGAT (disjunctive edges with the per-machine cap, and the fixed-preference term of
the lower policy), otherwise the curves measure a method that is not the one of the comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
import gc
try:
    import msvcrt
except ImportError:  # non-Windows fallback used only by the probes
    msvcrt = None
try:
    import fcntl
except ImportError:  # Windows fallback used only by the probes
    fcntl = None
import json
import glob
import warnings
import matplotlib
# Without a display session (background run, stdout redirected to a file, headless server)
# the Qt backend cannot be initialised and the process would crash; fall back to the file-only
# Agg backend and keep the default backend for interactive runs.
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def release_device():
    """Drop the references of the finished scan value and release its CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Every artefact of this scan is written under Figure_And_File, so that the
# output location does not depend on the working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The figures and the tables of this scan belong to their own folder, named after this experiment
# and sitting next to the one of the method whose parameters are scanned, so that neither the
# comparison experiments nor the method folder can overwrite them.
FIGURE_DIR = os.path.join(_PROJECT_ROOT, "Figure_And_File", "PPO_HGAT", "Param_Experiment")
os.makedirs(FIGURE_DIR, exist_ok=True)

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import re

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10
# Overflow guard of the main run (PPO_HGAT.Config.logits_clip).  The policy logits of the
# manuscript are clamped to this range before the mask is written, so a scan of the
LOGITS_CLIP = 50.0

W1_DEFAULT = 0.4
W2_DEFAULT = 0.3
W3_DEFAULT = 0.3

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

INSTANCE_DIR = os.path.join(_PROJECT_ROOT, "DataSet", "mo_fjsp_instances")
FILE_PATTERN = "mo_fjsp_*_train.txt"
MODEL_DIR = os.path.join(_PROJECT_ROOT, "model", "ppo_hgat")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hgat_best.pth")

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Episodes every configuration of the sweep is trained for.  The value was ten, which leaves every
# configuration essentially untrained and turns the curves into a reading of the initialisation
NUM_EPOCHS = 800
# Episode budget of one configuration of this scan, and of one point of the weight study that reuses
# it (the line above is the old, arbitrary value; this is the one that applies).  It is set from the
NUM_EPOCHS = 2500
NUM_EPOCHS = int(os.environ.get("PARAM_EPOCHS", NUM_EPOCHS))

# Instances collected in lockstep by the training loop of one configuration.  The loop of this
# study used to carry a single environment: every decision of an episode was one forward pass of
NUM_ENVS = int(os.environ.get("PARAM_NUM_ENVS", "12"))

# Rollouts of one instance decoded together while the test set of a configuration is evaluated.
# The protocol keeps the best of one greedy pass and fifty samples, and the fifty were decoded one
EVAL_BATCH = int(os.environ.get("PARAM_EVAL_BATCH", "8"))

# Frame of the measurements of this scan.  A value of the sweep writes its row into a small cache
# as soon as it is measured, so that an interruption - the machine of this project is shut down on
PARAM_CACHE_FRAME = os.environ.get("PARAM_CACHE_FRAME", "curriculum-2026-10-18")
# Probe flags used by the throughput benchmark.  A probe forces large instances and does not
# write the value cache, so it can measure chunk sizes without touching the scan frame.
PARAM_PROBE = os.environ.get("PARAM_PROBE", "") not in ("", "0")
FORCE_LARGE = os.environ.get("PARAM_FORCE_LARGE", "") not in ("", "0")
PARAM_VALUE_ONLY = os.environ.get("PARAM_VALUE_ONLY", "") not in ("", "0")
PARAM_ASSEMBLE_FROM_CACHE = os.environ.get("PARAM_ASSEMBLE_FROM_CACHE", "") not in ("", "0")


def _value_cache_path(param):
    """Path of the value cache of one parameter of the scan."""
    return os.path.join(FIGURE_DIR, "values_%s.json" % param)


def _load_value_cache(param):
    """Rows of the value cache of ``param`` that belong to the current frame."""
    try:
        with open(_value_cache_path(param), "r", encoding="utf-8") as handle:
            blob = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(blob, dict) or blob.get("frame") != PARAM_CACHE_FRAME:
        return {}
    rows = blob.get("rows")
    return rows if isinstance(rows, dict) else {}


def reuse_cached_row(entry, param, value):
    """Append the cached row of ``value`` to the results of ``param``; True when it existed."""
    row = _load_value_cache(param).get(str(value))
    if not row:
        return False
    print("    %s=%s was measured already (value cache of frame %s); the training is skipped"
          % (param, value, PARAM_CACHE_FRAME), flush=True)
    entry['values'].append(row['value'])
    for key in ('F', 'Cmax', 'LB', 'Tardy', 'epochs', 'sd', 'F_list'):
        entry.setdefault(key, []).append(row[key])
    return True


def remember_row(param, value, entry):
    """Write the row that was appended last for ``value`` into the value cache of ``param``."""
    index = len(entry['values']) - 1
    if index < 0:
        return
    if PARAM_PROBE:
        return
    path = _value_cache_path(param)
    lock_path = path + ".lock"
    with open(lock_path, "a+b") as lock:
        if msvcrt is not None:
            lock.seek(0)
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        elif fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = _load_value_cache(param)
            rows[str(value)] = {
                'value': value,
                'F': entry['F'][index],
                'Cmax': entry['Cmax'][index],
                'LB': entry['LB'][index],
                'Tardy': entry['Tardy'][index],
                'epochs': entry['epochs'][index],
                'sd': entry.get('sd', [0.0] * (index + 1))[index],
                'F_list': entry.get('F_list', [[]] * (index + 1))[index],
            }
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump({'frame': PARAM_CACHE_FRAME, 'rows': rows}, handle,
                          ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if msvcrt is not None:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

# Rollouts per instance of one configuration: one deterministic pass and fifty samples, which is
# the protocol the comparison experiment reports and therefore the one on which the parameters are
# selected.  A configuration is scored by the best of them, exactly as in that protocol.

# Update passes over one collected trajectory.  The trainer of the method walks a collected
# episode exactly once: update() of PPO_HGAT has a single loop over the chunks of the trajectory
PPO_EPOCHS = 1
EVAL_TEMPERATURE = 1.0
EVAL_ROLLOUTS = 51
# Update structure of the method the sweep measures.  The recipe of the proposed method walks the
# chunks of a collected episode once, clips the gradient to the value of the family and takes one
# optimiser step per chunk; the same structure is used here with one pass (see PPO_EPOCHS).
UPDATE_EPOCHS = 1
UPDATE_CHUNK = 32
# States per chunk of the value estimate of one collected episode.  The update already walks the
# trajectory in chunks, but the estimate of the values walked it as one graph, and the packed
VALUE_CHUNK = int(os.environ.get("PARAM_VALUE_CHUNK", "8"))
# States per group of one update chunk, which is the second half of the same measurement: two
# collectors share this card and the whole chunk of both of them does not fit in its memory, so
UPDATE_SUBCHUNK = int(os.environ.get("PARAM_UPDATE_SUBCHUNK", "8"))
MAX_GRAD_NORM = 50.0

REWARD_SCALING = 5.0
REWARD_CLIP = 2.0

# GPU-based normalization tool
class TorchRunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4, device=DEVICE):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.device = device

    def update(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.square() * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

# Data reading and instance definition
def parse_size_from_filename(filename: str) -> str:
    match = re.search(r'(small|medium|large)', filename)
    return match.group(1) if match else "unknown"

@dataclass
class Operation:
    job_id: int
    op_id: int
    machine_times: Dict[int, float]

def read_fjsp_instance(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    num_jobs, num_machines = map(int, lines[0].split())
    job_lines = lines[1:1 + num_jobs]
    due_dates_line = lines[-2]
    capabilities_line = lines[-1]
    due_dates = list(map(float, due_dates_line.split()))
    machine_capabilities = list(map(int, capabilities_line.split()))
    jobs = []
    for job_idx in range(num_jobs):
        nums = list(map(int, job_lines[job_idx].split()))
        num_ops = nums[0]
        idx = 1
        job_ops = []
        for op_idx in range(num_ops):
            k = nums[idx]
            idx += 1
            machine_times = {}
            for _ in range(k):
                m_id = nums[idx] - 1
                p_time = float(nums[idx + 1])
                machine_times[m_id] = p_time
                idx += 2
            job_ops.append(Operation(job_idx, op_idx, machine_times))
        jobs.append(job_ops)
    file_name = os.path.basename(file_path)
    size = parse_size_from_filename(file_name)
    return jobs, machine_capabilities, due_dates, file_name, size

def load_all_instances(instance_dir: str, pattern: str):
    file_paths = glob.glob(os.path.join(instance_dir, pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files matching {pattern} found in directory {instance_dir}")
    instance_list = []
    max_jobs = 0
    max_machines = 0
    max_proc_time = 0
    max_due_date = 0
    max_total_ops = 0
    max_capability = 0
    for fp in file_paths:
        jobs, caps, due_dates, fname, size = read_fjsp_instance(fp)
        total_ops = sum(len(j) for j in jobs)
        instance_list.append((jobs, caps, due_dates, fname, size))
        max_jobs = max(max_jobs, len(jobs))
        max_machines = max(max_machines, len(caps))
        for job_ops in jobs:
            for op in job_ops:
                for t in op.machine_times.values():
                    max_proc_time = max(max_proc_time, t)
        max_due_date = max(max_due_date, max(due_dates) if due_dates else 0)
        max_total_ops = max(max_total_ops, total_ops)
        max_capability = max(max_capability, max(caps) if caps else 0)
    return instance_list, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability

class MOFJSPInstance:
    def __init__(self, jobs, machine_capabilities, due_dates, file_name, size,
                 global_max_proc_time, global_max_capability):
        self.n_jobs = len(jobs)
        self.n_machines = len(machine_capabilities)
        self.due_dates = due_dates
        self.machine_capabilities = machine_capabilities
        self.file_name = file_name
        self.size = size
        self.ops_per_job = [len(j) for j in jobs]
        self.total_ops = sum(self.ops_per_job)

        self.processing_times = {}
        self.available_machines = {}

        self.proc_time_matrix = torch.full((self.total_ops, self.n_machines), 99999.0, dtype=torch.float32, device=DEVICE)

        total_assignments = 0
        for j_idx, job in enumerate(jobs):
            for op in job:
                key = (j_idx, op.op_id)
                self.available_machines[key] = []
                for m, t in op.machine_times.items():
                    self.processing_times[(j_idx, op.op_id, m)] = t
                    self.available_machines[key].append(m)
                    total_assignments += 1

        all_times = list(self.processing_times.values())
        self.max_proc_time = max(all_times) if all_times else 1.0
        self.mean_proc_time = np.mean(all_times) if all_times else 1.0

        self.avg_flex = total_assignments / (self.total_ops * self.n_machines) if self.total_ops > 0 else 0.0

        self.op_index_map = {}
        op_idx = 0
        for job in range(self.n_jobs):
            for op in range(self.ops_per_job[job]):
                self.op_index_map[(job, op)] = op_idx
                op_idx += 1
        self.mac_index_map = {m: m for m in range(self.n_machines)}

        self.op_idx_to_job = torch.zeros(self.total_ops, dtype=torch.long, device=DEVICE)
        self.op_idx_to_op = torch.zeros(self.total_ops, dtype=torch.long, device=DEVICE)
        for (job, op), idx in self.op_index_map.items():
            self.op_idx_to_job[idx] = job
            self.op_idx_to_op[idx] = op

        max_ops = max(self.ops_per_job)
        self.job_op_to_idx = torch.full((self.n_jobs, max_ops), -1, dtype=torch.long, device=DEVICE)
        for (job, op), idx in self.op_index_map.items():
            self.job_op_to_idx[job, op] = idx

        self.op_predecessor = torch.full((self.total_ops,), -1, dtype=torch.long, device=DEVICE)
        for job in range(self.n_jobs):
            for op in range(1, self.ops_per_job[job]):
                cur_idx = self.op_index_map[(job, op)]
                prev_idx = self.op_index_map[(job, op-1)]
                self.op_predecessor[cur_idx] = prev_idx

        self.op_mac_mask = torch.zeros((self.total_ops, self.n_machines), dtype=torch.float32, device=DEVICE)
        for (job, op), idx in self.op_index_map.items():
            for m in self.available_machines[(job, op)]:
                self.op_mac_mask[idx, m] = 1.0

        for (job, op, m), t in self.processing_times.items():
            idx = self.op_index_map[(job, op)]
            self.proc_time_matrix[idx, m] = t

        self.due_dates_tensor = torch.tensor(due_dates, dtype=torch.float32, device=DEVICE)
        self.ops_per_job_tensor = torch.tensor(self.ops_per_job, dtype=torch.long, device=DEVICE)

        self.capability_tensor = torch.tensor(machine_capabilities, dtype=torch.float32, device=DEVICE)
        if global_max_capability > 0:
            self.capability_tensor = self.capability_tensor / global_max_capability
        else:
            self.capability_tensor = self.capability_tensor * 0.0

        # Precompute edges
        seq_src = []
        seq_dst = []
        for job in range(self.n_jobs):
            for op in range(self.ops_per_job[job] - 1):
                u = self.op_index_map[(job, op)]
                v = self.op_index_map[(job, op+1)]
                seq_src.append(u)
                seq_dst.append(v)
        self.seq_edges_src = torch.tensor(seq_src, dtype=torch.long, device=DEVICE)
        self.seq_edges_dst = torch.tensor(seq_dst, dtype=torch.long, device=DEVICE)

        alloc_src = []
        alloc_dst = []
        alloc_feat = []
        for (job, op), idx in self.op_index_map.items():
            for m in self.available_machines[(job, op)]:
                alloc_src.append(idx)
                alloc_dst.append(m)
                pt = self.processing_times[(job, op, m)]
                norm_pt = pt / (global_max_proc_time + EPS)
                alloc_feat.append(norm_pt)
        self.alloc_edges_src = torch.tensor(alloc_src, dtype=torch.long, device=DEVICE)
        self.alloc_edges_dst = torch.tensor(alloc_dst, dtype=torch.long, device=DEVICE)
        self.alloc_edges_feat = torch.tensor(alloc_feat, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

        self.alloc_rev_edges_src = self.alloc_edges_dst.clone()
        self.alloc_rev_edges_dst = self.alloc_edges_src.clone()
        self.alloc_rev_edges_feat = self.alloc_edges_feat.clone()

        self.machine_ops_list = []
        for m in range(self.n_machines):
            ops = torch.nonzero(self.op_mac_mask[:, m] > 0.5).squeeze(1).cpu()
            self.machine_ops_list.append(ops)

        # Disjunctive edges of the manuscript: two operations that compete for the same machine.
        # The set is capped per machine (the graph sampling the manuscript proposes for this edge
        disj_src = []
        disj_dst = []
        for machine in range(self.n_machines):
            ops = self.machine_ops_list[machine].tolist()
            pairs = 0
            for first in range(len(ops)):
                for second in range(first + 1, len(ops)):
                    if pairs >= 50:
                        break
                    disj_src.append(ops[first])
                    disj_dst.append(ops[second])
                    disj_src.append(ops[second])
                    disj_dst.append(ops[first])
                    pairs += 1
                if pairs >= 50:
                    break
        self.disj_edges_src = torch.tensor(disj_src, dtype=torch.long, device=DEVICE)
        self.disj_edges_dst = torch.tensor(disj_dst, dtype=torch.long, device=DEVICE)
        self.disj_edges_feat = torch.full((len(disj_src), 1), 2.0, dtype=torch.float32,
                                          device=DEVICE)

        # Normalised processing time of every operation on every machine, the first term of the
        # fixed-preference heuristic of the lower policy (Eq. 10).  The infeasible pairs are zeroed
        self.pt_rows_norm = torch.where(self.proc_time_matrix > 99998.0,
                                        torch.zeros_like(self.proc_time_matrix),
                                        self.proc_time_matrix / (self.max_proc_time + EPS))

    def get_pt(self, job, op, machine):
        return self.proc_time_matrix[self.op_index_map[(job, op)], machine].item()

# Heterogeneous graph environment
class HeteroGraphEnv:
    def __init__(self, instance: MOFJSPInstance, global_max_jobs, global_max_ops, global_max_machines,
                 global_max_proc_time, global_max_due_date, cfg):
        self.inst = instance
        self.global_max_jobs = global_max_jobs
        # The operation-feature table is sized by the global maximum of the training family, but a
        # comparison instance may carry more operations than anything the policies were trained on
        self.global_max_ops = max(global_max_ops, instance.total_ops)
        self.global_max_machines = global_max_machines
        self.global_max_proc_time = global_max_proc_time
        self.global_max_due_date = global_max_due_date
        self.cfg = cfg

        self.global_max_proc_time_tensor = torch.tensor(global_max_proc_time, device=DEVICE)
        self.global_max_due_date_tensor = torch.tensor(global_max_due_date, device=DEVICE)

        self.total_op_nodes = instance.total_ops
        self.total_machine_nodes = instance.n_machines

        # Original table: position, due date, job id; load, availability and capability.
        self.op_feat_dim = 3
        self.mac_feat_dim = 3

        self.op_predecessor = instance.op_predecessor
        self.job_op_to_idx = instance.job_op_to_idx
        self.op_mac_mask = instance.op_mac_mask
        self.proc_time_matrix = instance.proc_time_matrix
        self.disj_edges_src = instance.disj_edges_src
        self.disj_edges_dst = instance.disj_edges_dst
        self.disj_edges_feat = instance.disj_edges_feat
        self.due_dates_tensor = instance.due_dates_tensor
        self.ops_per_job_tensor = instance.ops_per_job_tensor
        self.op_idx_to_job = instance.op_idx_to_job
        self.op_idx_to_op = instance.op_idx_to_op

        self.seq_edges_src = instance.seq_edges_src
        self.seq_edges_dst = instance.seq_edges_dst
        self._seq_ones = torch.ones((self.seq_edges_src.numel(), 1),
                                    dtype=torch.float32, device=DEVICE)
        self.alloc_edges_src = instance.alloc_edges_src
        self.alloc_edges_dst = instance.alloc_edges_dst
        self.alloc_edges_feat = instance.alloc_edges_feat
        self.alloc_rev_edges_src = instance.alloc_rev_edges_src
        self.alloc_rev_edges_dst = instance.alloc_rev_edges_dst
        self.alloc_rev_edges_feat = instance.alloc_rev_edges_feat

        self.machine_ops_list = instance.machine_ops_list

        cache_key = (global_max_jobs, self.global_max_ops, global_max_due_date)
        cached = getattr(instance, "_static_feature_cache", None)
        if cached is not None and cached[0] == cache_key:
            self.static_op_feats, self.static_mac_feats = cached[1], cached[2]
        else:
            self.static_op_feats = torch.zeros((self.global_max_ops, self.op_feat_dim),
                                             dtype=torch.float32, device=DEVICE)
            for job in range(instance.n_jobs):
                ops_in_job = instance.ops_per_job[job]
                due_time = instance.due_dates[job] / (global_max_due_date + EPS)
                job_id_norm = job / max(1, global_max_jobs - 1)
                for op in range(ops_in_job):
                    node_idx = instance.op_index_map[(job, op)]
                    self.static_op_feats[node_idx, 0] = op / max(1, ops_in_job - 1)
                    self.static_op_feats[node_idx, 1] = due_time
                    self.static_op_feats[node_idx, 2] = job_id_norm

            self.static_mac_feats = torch.zeros(
                (self.global_max_machines, self.mac_feat_dim), dtype=torch.float32,
                device=DEVICE)
            for m in range(instance.n_machines):
                self.static_mac_feats[m, 2] = instance.capability_tensor[m]
            instance._static_feature_cache = (
                cache_key, self.static_op_feats, self.static_mac_feats)

        self.reset()

    def reset(self):
        self.machine_available_time = torch.zeros(self.inst.n_machines, dtype=torch.float32, device=DEVICE)
        self.machine_load = torch.zeros(self.inst.n_machines, dtype=torch.float32, device=DEVICE)
        # Live mirrors of the two machine states on the instance.  The lower policy of the method
        # scores a machine from its processing time, its load and its waiting time (Eq. 10), and
        self.inst.machine_load = self.machine_load
        self.inst.machine_available_time = self.machine_available_time
        self.job_next_op = torch.zeros(self.inst.n_jobs, dtype=torch.long, device=DEVICE)
        self.job_completion_time = torch.zeros(self.inst.n_jobs, dtype=torch.float32, device=DEVICE)
        self.current_time = torch.tensor(0.0, device=DEVICE)

        self.op_finish_time = torch.zeros(self.total_op_nodes, dtype=torch.float32, device=DEVICE)
        self.op_scheduled = torch.zeros(self.total_op_nodes, dtype=torch.bool, device=DEVICE)
        self.scheduled_count = 0
        self.zero_scalar = torch.zeros((), dtype=torch.float32, device=DEVICE)

        self.op_feats_t = self.static_op_feats.clone()
        self.mac_feats_t = self.static_mac_feats.clone()

        return self._get_state()

    def step(self, action):
        job, op, machine = action
        op_idx = self.job_op_to_idx[job.long(), op.long()]
        p_time = self.proc_time_matrix[op_idx, machine.long()]

        pre_idx = self.op_predecessor[op_idx]
        job_ready = torch.where(pre_idx != -1, self.op_finish_time[pre_idx], torch.tensor(0.0, device=DEVICE))
        machine_ready = self.machine_available_time[machine.long()]
        start = torch.maximum(job_ready, machine_ready)
        end = start + p_time

        # The load-balance variation ΔLB_t must be the difference of the load standard deviation
        # between "before and after this assignment": record the standard deviation before the assignment, update the machine loads and take the standard deviation afterwards; the original implementation accumulated
        # p_time and then accumulated p_time again on the same machine, which is not the load change caused by this assignment (fixed to a single-step difference).
        old_LB = torch.std(self.machine_load)
        self.machine_available_time[machine.long()] = end
        self.machine_load[machine.long()] += p_time
        new_LB = torch.std(self.machine_load)
        delta_LB = new_LB - old_LB
        self.job_next_op[job.long()] = self.job_next_op[job.long()] + 1
        self.job_completion_time[job.long()] = torch.maximum(self.job_completion_time[job.long()], end)
        self.current_time = torch.maximum(self.current_time, end)
        self.op_finish_time[op_idx] = end
        self.op_scheduled[op_idx] = True
        self.scheduled_count += 1

        norm_p = p_time / self.global_max_proc_time_tensor
        due = self.due_dates_tensor[job.long()]
        t_c = self.current_time
        n_i = self.ops_per_job_tensor[job.long()]
        # Operation urgency: zero while the operation finishes before its due date and
        # proportional to the lateness once the due date is exceeded. Using the lateness
        u_ij = torch.clamp(end - due, min=0) / (self.global_max_due_date_tensor * n_i + EPS)

        reward = - (self.cfg.w1 * norm_p + self.cfg.w2 * delta_LB / self.global_max_proc_time_tensor + self.cfg.w3 * u_ij)
        reward /= self.cfg.reward_scaling
        reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        done = self.scheduled_count >= self.total_op_nodes
        if done:
            c_max = self.current_time
            avg_load = torch.mean(self.machine_load)
            lb = torch.sqrt(torch.mean((self.machine_load - avg_load) ** 2))
            t_tardy = torch.sum(torch.clamp(self.job_completion_time - self.due_dates_tensor, min=0))

            # The terminal reward is normalised by the size and the scale of the instance, exactly
            # as in PPO_HGAT.PPO_HGAT: the fixed divisors this script carried (c_max/100, lb/50,
            n_ops = float(self.op_scheduled.numel())
            global_penalty = (
                self.cfg.w1 * c_max / (self.global_max_proc_time_tensor * n_ops)
                + self.cfg.w2 * lb / (avg_load + EPS)
                + self.cfg.w3 * t_tardy / (self.global_max_due_date_tensor * n_ops))
            global_penalty /= self.cfg.reward_scaling
            reward -= global_penalty
            reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        next_state = self._get_state() if not done else None
        return next_state, reward, done

    def _get_state(self):
        total_load = self.machine_load.sum()
        load_norm = self.machine_load / (total_load + EPS)
        avail_norm = self.machine_available_time / (self.current_time + 1.0 + EPS)
        self.mac_feats_t[:self.inst.n_machines, 0] = load_norm
        self.mac_feats_t[:self.inst.n_machines, 1] = avail_norm


        unsched_mask = ~self.op_scheduled

        seq_valid = unsched_mask[self.seq_edges_src] & unsched_mask[self.seq_edges_dst]
        seq_src_final = self.seq_edges_src[seq_valid]
        seq_dst_final = self.seq_edges_dst[seq_valid]
        seq_feat_final = self._seq_ones[:seq_src_final.shape[0]]

        alloc_valid = unsched_mask[self.alloc_edges_src]
        alloc_src_final = self.alloc_edges_src[alloc_valid]
        alloc_dst_final = self.alloc_edges_dst[alloc_valid]
        alloc_feat_final = self.alloc_edges_feat[alloc_valid]

        alloc_rev_valid = unsched_mask[self.alloc_rev_edges_dst]
        alloc_rev_src_final = self.alloc_rev_edges_src[alloc_rev_valid]
        alloc_rev_dst_final = self.alloc_rev_edges_dst[alloc_rev_valid]
        alloc_rev_feat_final = self.alloc_rev_edges_feat[alloc_rev_valid]

        # Competition stays visible while both of its operations are still unscheduled, so the
        # encoder only sees the pairs that still compete for a machine.
        unscheduled = ~self.op_scheduled
        disj_valid = unscheduled[self.disj_edges_src] & unscheduled[self.disj_edges_dst]
        disj_src_final = self.disj_edges_src[disj_valid]
        disj_dst_final = self.disj_edges_dst[disj_valid]
        disj_feat_final = self.disj_edges_feat[disj_valid]

        edges_t = {
            'seq': (seq_src_final, seq_dst_final),
            'disj': (disj_src_final, disj_dst_final),
            'op_mac': (alloc_src_final, alloc_dst_final),
            'mac_op': (alloc_rev_src_final, alloc_rev_dst_final)
        }
        edge_feats_t = {
            'seq': seq_feat_final,
            'disj': disj_feat_final,
            'op_mac': alloc_feat_final,
            'mac_op': alloc_rev_feat_final
        }

        op_mask = torch.zeros(self.global_max_ops, dtype=torch.float32, device=DEVICE)
        job_ids = torch.arange(self.inst.n_jobs, device=DEVICE)
        next_op = self.job_next_op
        max_ops_per_job = self.ops_per_job_tensor
        valid_jobs = job_ids[next_op < max_ops_per_job]
        if len(valid_jobs) > 0:
            avail_op_indices = self.job_op_to_idx[valid_jobs, next_op[valid_jobs]]
            op_mask[avail_op_indices] = 1.0

        return {
            'op_feats': self.op_feats_t.clone(),  # in-place updated tensors must be copied, otherwise later steps overwrite the trajectory state
            'mac_feats': self.mac_feats_t.clone(),  # in-place updated tensors must be copied, otherwise later steps overwrite the trajectory state
            'edges': edges_t,
            'edge_feats': edge_feats_t,
            'op_mask': op_mask,
            # The two live mirrors are stored with the state so that the batched collector can
            # rebuild the same heuristic term of the lower policy (Eq. 10) that the single
            'machine_load': self.machine_load.clone(),
            'machine_available_time': self.machine_available_time.clone(),
        }

# Heterogeneous graph neural network
def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer

class HeteroGATLayer(nn.Module):
    def __init__(self, in_dim_op, in_dim_mac, out_dim, num_heads=4):
        super().__init__()
        self.heads = num_heads
        self.d_k = out_dim // num_heads
        self.out_dim = out_dim

        self.W_op = nn.Linear(in_dim_op, out_dim)
        self.W_mac = nn.Linear(in_dim_mac, out_dim)
        orthogonal_init(self.W_op, gain=nn.init.calculate_gain('relu'))
        orthogonal_init(self.W_mac, gain=nn.init.calculate_gain('relu'))

        self.att_seq = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))
        self.att_op_mac = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))
        self.att_mac_op = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))
        # Disjunctive (machine competition) channel, identical to the one of the trained proposed
        # method (PPO_HGAT.py).  This study re-implements the pipeline so that no checkpoint is
        self.att_disj = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))

        self.leaky = nn.LeakyReLU(0.2)
        self.max_edges = 300000

    def forward(self, h_op, h_mac, edges, edge_feats):
        dtype = h_op.dtype
        device = h_op.device

        proj_op = self.W_op(h_op)
        proj_mac = self.W_mac(h_mac)
        z_op = proj_op.view(-1, self.heads, self.d_k)
        z_mac = proj_mac.view(-1, self.heads, self.d_k)

        out_op = torch.zeros_like(z_op)
        out_mac = torch.zeros_like(z_mac)

        edge_feats_proc = {}
        for key, feat in edge_feats.items():
            if feat.numel() > 0:
                edge_feats_proc[key] = feat.to(dtype)
            else:
                edge_feats_proc[key] = feat

        def apply_attn(src_emb, dst_emb, src_idx, dst_idx, feat_e, att_param, out_buffer,
                      gate=False):
            if src_idx.numel() == 0:
                return
            if src_idx.numel() > self.max_edges:
                return
            h_s = src_emb[src_idx]
            h_d = dst_emb[dst_idx]
            att_param = att_param.to(dtype)
            d_k = self.d_k
            scores = (h_s * att_param[:, :d_k]).sum(dim=-1)
            scores = scores + (h_d * att_param[:, d_k:2*d_k]).sum(dim=-1)
            scores = scores + feat_e * att_param[:, 2*d_k]
            scores = self.leaky(scores)
            if gate:
                # Gate of the conjunctive channel, applied to the raw score exactly as the trained
                # method applies it (PPO_HGAT.HeteroGATLayer.apply_attn places the stability
                weight = torch.sigmoid(scores).to(dtype)
                weighted = (h_s * weight.unsqueeze(-1)).to(dtype)
                out_buffer.index_add_(0, dst_idx, weighted)
                return

            # Stability transform of the trained method: the largest score is subtracted before
            # the exponential, because the exponential of a raw score overflows once the score
            scores = scores - scores.detach().max()
            alpha = torch.exp(scores)
            denom = torch.zeros(out_buffer.shape[0], self.heads, device=device, dtype=dtype)
            denom.index_add_(0, dst_idx, alpha)
            denom = denom[dst_idx] + torch.tensor(EPS, dtype=dtype, device=device)
            norm_alpha = alpha / denom
            weighted = h_s * norm_alpha.unsqueeze(-1)
            weighted = weighted.to(dtype)
            out_buffer.index_add_(0, dst_idx, weighted)

        if edges['seq'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['seq'][0], edges['seq'][1], edge_feats_proc['seq'], self.att_seq,
                       out_op, gate=True)
        if edges['op_mac'][0].numel() > 0:
            apply_attn(z_op, z_mac, edges['op_mac'][0], edges['op_mac'][1], edge_feats_proc['op_mac'], self.att_op_mac, out_mac)
        if edges['mac_op'][0].numel() > 0:
            apply_attn(z_mac, z_op, edges['mac_op'][0], edges['mac_op'][1], edge_feats_proc['mac_op'], self.att_mac_op, out_op)

        # Same disjunctive channel as the trained method: this study must scan the model with the
        # representation it was trained with, otherwise the ranking of the configurations
        # describes another encoder.
        if edges.get('disj') is not None and edges['disj'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['disj'][0], edges['disj'][1], edge_feats_proc['disj'],
                       self.att_disj, out_op)

        res_op = out_op.flatten(1) + proj_op
        res_mac = out_mac.flatten(1) + proj_mac
        return F.elu(res_op), F.elu(res_mac)

class HeteroGAT(nn.Module):
    def __init__(self, dim_op, dim_mac, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        self.l1 = HeteroGATLayer(dim_op, dim_mac, hidden_dim, num_heads)
        self.l2 = HeteroGATLayer(hidden_dim, hidden_dim, out_dim, num_heads)
        self.out_dim = out_dim

    def forward(self, state_dict):
        op_feats = state_dict['op_feats']
        mac_feats = state_dict['mac_feats']
        edges = state_dict['edges']
        edge_feats = state_dict['edge_feats']

        h_op1, h_mac1 = self.l1(op_feats, mac_feats, edges, edge_feats)
        h_op2, h_mac2 = self.l2(h_op1, h_mac1, edges, edge_feats)

        return h_op2, h_mac2

class UpperPolicy(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 2, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1)
        )

    def forward(self, g_emb, op_embs, mask=None):
        if g_emb.dim() == 1:
            N = op_embs.shape[0]
            g_rep = g_emb.unsqueeze(0).expand(N, -1)
            cat = torch.cat([g_rep, op_embs], dim=1)
            logits = self.net(cat).squeeze(-1)
            logits = torch.clamp(logits, min=-LOGITS_CLIP, max=LOGITS_CLIP)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e10)
            probs = F.softmax(logits, dim=0)
        else:
            batch_size, N, _ = op_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, N, -1)
            cat = torch.cat([g_rep, op_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            logits = torch.clamp(logits, min=-LOGITS_CLIP, max=LOGITS_CLIP)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e10)
            probs = F.softmax(logits, dim=1)
        return probs

class LowerPolicy(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 3, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1)
        )

    def forward(self, g_emb, selected_op_emb, mac_embs, mask=None, heuristic=None):
        if g_emb.dim() == 1:
            M = mac_embs.shape[0]
            g_rep = g_emb.unsqueeze(0).expand(M, -1)
            op_rep = selected_op_emb.unsqueeze(0).expand(M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=1)
            logits = self.net(cat).squeeze(-1)
            if heuristic is not None:
                # Fixed-preference heuristic term of the manuscript's lower policy (Eq. 10).
                logits = logits + heuristic
            logits = torch.clamp(logits, min=-LOGITS_CLIP, max=LOGITS_CLIP)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e10)
            probs = F.softmax(logits, dim=0)
        else:
            batch_size, M, _ = mac_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, M, -1)
            op_rep = selected_op_emb.unsqueeze(1).expand(-1, M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            if heuristic is not None:
                logits = logits + heuristic
            logits = torch.clamp(logits, min=-LOGITS_CLIP, max=LOGITS_CLIP)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e10)
            probs = F.softmax(logits, dim=1)
        return probs

class Critic(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)
        )
    def forward(self, g_emb):
        return self.net(g_emb)

# PPO Agent
# Batched graph layout of the vectorised collector
def padded_op_mac_mask(inst, n_op: int, n_mac: int):
    """Machine mask of every operation in the padded layout of a packed state.

    The packer lays the states out as consecutive blocks of ``n_op`` operation nodes, so the mask
    of a whole chunk is one gather instead of one row-wise copy per state.  The tensor depends on
    the padding of the instance alone and is therefore cached on it.
    """
    mask = getattr(inst, '_op_mac_mask_padded', None)
    if mask is None or mask.shape[0] != n_op or mask.shape[1] != n_mac:
        mask = torch.zeros((n_op, n_mac), dtype=torch.float32, device=DEVICE)
        mask[:inst.total_ops, :inst.n_machines] = inst.op_mac_mask
        inst._op_mac_mask_padded = mask
    return mask


def pack_states(states: List[Dict], n_op: int, n_mac: int):
    """Pack several states into the one heterogeneous graph the encoders of this study read.

    Every state contributes one block of ``n_op`` operation nodes, one block of ``n_mac`` machine
    nodes and its own edges, whose endpoints are shifted into the block of their own state.  The
    state of this study carries the edges that the current visibility mask keeps (the filtering is
    done in the environment), so the packer only shifts and concatenates and no validity channel
    is needed.  The packed graph is the disjoint union of the graphs the single environment loop
    handed to the encoder one at a time: the attention of a layer normalises over the edges that
    end in a node, and every node of a packed batch is reached only by the edges of its own state,
    so the batches are independent and one packed forward reproduces one forward per state.  The
    order of the edges of a state is preserved, which keeps the numerical result of the layer
    comparable with the one of the single environment path.

    Returns the packed state dictionary and the concatenated operation mask.
    """
    batch_size = len(states)
    op_offsets = torch.arange(batch_size, device=DEVICE) * n_op
    mac_offsets = torch.arange(batch_size, device=DEVICE) * n_mac
    empty_edges = (torch.empty(0, dtype=torch.long, device=DEVICE),
                   torch.empty(0, dtype=torch.long, device=DEVICE))
    empty_feat = torch.empty((0, 1), dtype=torch.float32, device=DEVICE)

    batch_edges = {}
    batch_edge_feats = {}
    for key in ('seq', 'disj', 'op_mac', 'mac_op'):
        lengths = [s['edges'][key][0].numel() for s in states]
        if not any(lengths):
            batch_edges[key] = empty_edges
            batch_edge_feats[key] = empty_feat
            continue

        src_all = torch.cat([s['edges'][key][0] for s in states]).long()
        dst_all = torch.cat([s['edges'][key][1] for s in states]).long()
        feat_all = torch.cat([s['edge_feats'][key] for s in states], dim=0)
        owner = torch.repeat_interleave(
            torch.arange(batch_size, device=DEVICE),
            torch.tensor(lengths, dtype=torch.long, device=DEVICE))

        if key in ('seq', 'disj'):
            # Sequential precedence and machine competition both run between two operations.
            src = src_all + op_offsets[owner]
            dst = dst_all + op_offsets[owner]
        elif key == 'op_mac':
            src = src_all + op_offsets[owner]
            dst = dst_all + mac_offsets[owner]
        else:
            src = src_all + mac_offsets[owner]
            dst = dst_all + op_offsets[owner]

        batch_edges[key] = (src, dst)
        batch_edge_feats[key] = feat_all

    packed = {
        'op_feats': torch.cat([s['op_feats'] for s in states], dim=0),
        'mac_feats': torch.cat([s['mac_feats'] for s in states], dim=0),
        'edges': batch_edges,
        'edge_feats': batch_edge_feats,
    }
    op_mask = torch.cat([s['op_mask'] for s in states], dim=0)
    return packed, op_mask


class HierarchicalPPOAgent:
    def __init__(self, inst, gat, upper, lower, critic_u, critic_l, gamma=0.99, lam=0.95, clip=0.2, beta=0.01, lr=3e-4, state_norm=None):
        self.inst = inst
        self.gat = gat
        self.upper = upper
        self.lower = lower
        self.critic_u = critic_u
        self.critic_l = critic_l
        self.state_norm = state_norm
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.beta = beta
        self.lr = lr
        self.params = list(gat.parameters()) + list(upper.parameters()) + \
                      list(lower.parameters()) + list(critic_u.parameters()) + list(critic_l.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)

    def save(self, path):
        torch.save({
            'gat': self.gat.state_dict(),
            'upper': self.upper.state_dict(),
            'lower': self.lower.state_dict(),
            'critic_u': self.critic_u.state_dict(),
            'critic_l': self.critic_l.state_dict(),
            'optimizer': self.opt.state_dict(),
            'gamma': self.gamma,
            'beta': self.beta
        }, path)

    def load(self, path):
        if os.path.exists(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(path, map_location=DEVICE)
            self.gat.load_state_dict(checkpoint['gat'])
            self.upper.load_state_dict(checkpoint['upper'])
            self.lower.load_state_dict(checkpoint['lower'])
            self.critic_u.load_state_dict(checkpoint['critic_u'])
            self.critic_l.load_state_dict(checkpoint['critic_l'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            self.gamma = checkpoint.get('gamma', 0.99)
            self.beta = checkpoint.get('beta', 0.01)
            print(f"Model loaded from {path}, gamma={self.gamma:.3f}, beta={self.beta:.3f}")
            return True
        return False

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            h_op, h_mac = self.gat(state)
            g_emb = h_op.mean(dim=0)
            if self.state_norm is not None:
                g_emb = self.state_norm.normalize(g_emb)

        op_mask = state['op_mask']
        u_probs = self.upper(g_emb, h_op, op_mask)

        if deterministic:
            op_idx = torch.argmax(u_probs)
        else:
            dist = torch.distributions.Categorical(u_probs)
            op_idx = dist.sample()

        job = self.inst.op_idx_to_job[op_idx]
        op = self.inst.op_idx_to_op[op_idx]

        mac_mask = torch.zeros(h_mac.shape[0], device=DEVICE)
        if op_idx < self.inst.total_ops:
            mac_mask[:self.inst.n_machines] = self.inst.op_mac_mask[op_idx]

        selected_op_emb = h_op[op_idx]
        # Fixed-preference heuristic term of the manuscript's lower policy (Eq. 10): the
        # processing time of the chosen operation on every machine, the normalised load and the
        n_mac = h_mac.shape[0]
        pt_row = torch.zeros(n_mac, device=DEVICE)
        load_row = torch.zeros(n_mac, device=DEVICE)
        wait_row = torch.zeros(n_mac, device=DEVICE)
        pt_row[:self.inst.n_machines] = self.inst.pt_rows_norm[op_idx]
        live_load = self.inst.machine_load
        live_avail = self.inst.machine_available_time
        load_row[:self.inst.n_machines] = live_load / (live_load.sum() + EPS)
        wait_row[:self.inst.n_machines] = live_avail / (live_avail.max() + 1.0 + EPS)
        lower_heuristic = -(W1_DEFAULT * pt_row + W2_DEFAULT * load_row
                            + W3_DEFAULT * wait_row)
        l_probs = self.lower(g_emb, selected_op_emb, h_mac, mac_mask, lower_heuristic)

        if deterministic:
            mac_idx = torch.argmax(l_probs)
        else:
            dist_l = torch.distributions.Categorical(l_probs)
            mac_idx = dist_l.sample()

        machine = mac_idx

        return (job, op, machine), (u_probs[op_idx], l_probs[mac_idx]), (op_idx, mac_idx)

    def get_action_batch(self, states, insts, deterministic=False, check_invalid=True):
        """Decode a batch of states, one packed forward pass per block of equal geometry.

        Returns three lists aligned with ``states``: the action tuples, the probabilities of the
        chosen operation and machine, and their indices.  The result of an entry is the one
        ``get_action`` returns for that state (see ``_decode_group``).
        """
        actions = [None] * len(states)
        probs_list = [None] * len(states)
        indices_list = [None] * len(states)
        # The packed layout pads every state to the maximum operation count of the training
        # family, so the states of one batch share their geometry and form one block.  The grouping
        groups = {}
        for index, state in enumerate(states):
            key = (state['op_feats'].shape[0], state['mac_feats'].shape[0])
            groups.setdefault(key, []).append(index)

        for (n_op, n_mac), members in groups.items():
            group_actions, group_probs, group_indices = self._decode_group(
                [states[i] for i in members], [insts[i] for i in members],
                n_op, n_mac, deterministic, check_invalid)
            for slot, index in enumerate(members):
                actions[index] = group_actions[slot]
                probs_list[index] = group_probs[slot]
                indices_list[index] = group_indices[slot]
        return actions, probs_list, indices_list

    def _decode_group(self, states, insts, n_op, n_mac, deterministic, check_invalid=True):
        """One packed decode of a block of states of the same padded geometry."""
        batch_size = len(states)
        if check_invalid:
            op_mask_all = torch.stack([s['op_mask'] for s in states], dim=0)
            invalid = torch.nonzero(op_mask_all.sum(dim=1) <= 0).squeeze(1).tolist()
        else:
            invalid = []
        if invalid:
            # A state whose mask is empty has no operation left to schedule.  The single
            # environment loop never met it (it stopped on the terminal flag of the episode) and
            actions = [None] * batch_size
            probs_list = [None] * batch_size
            indices_list = [None] * batch_size
            for index in invalid:
                actions[index] = (0, 0, 0)
                probs_list[index] = (1.0, 1.0)
                indices_list[index] = (0, 0)
            valid = [i for i in range(batch_size) if i not in invalid]
            if valid:
                group = self._decode_group([states[i] for i in valid],
                                           [insts[i] for i in valid], n_op, n_mac,
                                           deterministic, check_invalid)
                for slot, index in enumerate(valid):
                    actions[index] = group[0][slot]
                    probs_list[index] = group[1][slot]
                    indices_list[index] = group[2][slot]
            return actions, probs_list, indices_list

        packed, batch_op_mask = pack_states(states, n_op, n_mac)
        rows = torch.arange(batch_size, device=DEVICE)

        with torch.no_grad():
            h_op_all, h_mac_all = self.gat(packed)
            batch_h_op = h_op_all.reshape(batch_size, n_op, -1)
            batch_h_mac = h_mac_all.reshape(batch_size, n_mac, -1)
            g_embs = batch_h_op.mean(dim=1)
            if self.state_norm is not None:
                g_embs = self.state_norm.normalize(g_embs)

            u_probs_batch = self.upper(g_embs, batch_h_op, batch_op_mask.view(batch_size, n_op))
            if deterministic:
                op_indices = torch.argmax(u_probs_batch, dim=1)
            else:
                if EVAL_TEMPERATURE == 1.0:
                    op_indices = torch.distributions.Categorical(u_probs_batch).sample()
                else:
                    mask = u_probs_batch > 0
                    tempered = torch.where(mask,
                                           torch.pow(u_probs_batch.clamp_min(1e-12), 1.0 / EVAL_TEMPERATURE),
                                           torch.zeros_like(u_probs_batch))
                    tempered = tempered / tempered.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    op_indices = torch.distributions.Categorical(tempered).sample()

            # Machine mask and the fixed-preference heuristic term of the lower policy (Eq. 10).
            # The two rows are read out of the state of the decision and not out of the live
            batch_mac_masks = torch.zeros((batch_size, n_mac), device=DEVICE)
            batch_pt_rows = torch.zeros((batch_size, n_mac), device=DEVICE)
            batch_load_rows = torch.zeros((batch_size, n_mac), device=DEVICE)
            batch_wait_rows = torch.zeros((batch_size, n_mac), device=DEVICE)
            for i, inst in enumerate(insts):
                op_index = op_indices[i]
                valid = (op_index < inst.total_ops).to(batch_mac_masks.dtype)
                safe_index = op_index.clamp(max=inst.total_ops - 1)
                batch_mac_masks[i, :inst.n_machines] = inst.op_mac_mask[safe_index] * valid
                batch_pt_rows[i, :inst.n_machines] = inst.pt_rows_norm[safe_index] * valid
                live_load = states[i]['machine_load']
                live_avail = states[i]['machine_available_time']
                batch_load_rows[i, :inst.n_machines] = (live_load / (live_load.sum() + EPS)) * valid
                batch_wait_rows[i, :inst.n_machines] = (live_avail / (live_avail.max() + 1.0 + EPS)) * valid
            lower_heuristic = -(W1_DEFAULT * batch_pt_rows + W2_DEFAULT * batch_load_rows
                                + W3_DEFAULT * batch_wait_rows)

            selected_op_emb = batch_h_op[rows, op_indices]
            l_probs_batch = self.lower(g_embs, selected_op_emb, batch_h_mac,
                                      batch_mac_masks, lower_heuristic)
            if deterministic:
                mac_indices = torch.argmax(l_probs_batch, dim=1)
            else:
                if EVAL_TEMPERATURE == 1.0:
                    mac_indices = torch.distributions.Categorical(l_probs_batch).sample()
                else:
                    mask = l_probs_batch > 0
                    tempered = torch.where(mask,
                                           torch.pow(l_probs_batch.clamp_min(1e-12), 1.0 / EVAL_TEMPERATURE),
                                           torch.zeros_like(l_probs_batch))
                    tempered = tempered / tempered.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    mac_indices = torch.distributions.Categorical(tempered).sample()

            actions, probs_list, indices_list = [], [], []
            for i, inst in enumerate(insts):
                actions.append((inst.op_idx_to_job[op_indices[i]],
                                inst.op_idx_to_op[op_indices[i]], mac_indices[i]))
                probs_list.append((u_probs_batch[i, op_indices[i]],
                                   l_probs_batch[i, mac_indices[i]]))
                indices_list.append((op_indices[i], mac_indices[i]))
        return actions, probs_list, indices_list

    def update(self, trajectory, inst=None):
        # ``inst`` is the instance the collected episode belongs to.  The vectorised collector
        # passes it explicitly, because the environments of a batch do not share the instance of
        inst = self.inst if inst is None else inst
        T = len(trajectory)
        if T == 0:
            return

        states = [t['state'] for t in trajectory]
        rewards = torch.tensor([t['reward'] for t in trajectory], dtype=torch.float32, device=DEVICE)
        old_u_probs = torch.stack([t['old_probs'][0] for t in trajectory]).detach()
        old_l_probs = torch.stack([t['old_probs'][1] for t in trajectory]).detach()
        op_indices = torch.tensor([t['indices'][0] for t in trajectory], device=DEVICE)
        mac_indices = torch.tensor([t['indices'][1] for t in trajectory], device=DEVICE)

        n_op_global = states[0]['op_feats'].shape[0]
        n_mac_global = states[0]['mac_feats'].shape[0]
        # Machine mask of the instance in the padded layout of a packed state.
        padded_mac_mask = padded_op_mac_mask(inst, n_op_global, n_mac_global)

        # Value estimates.  The episode is walked in bounded chunks and not as one graph: see
        # VALUE_CHUNK.  The value of a state is a function of that state alone, so this reproduces
        # the per-state series the replaced loop read one host value at a time.
        with torch.no_grad():
            values_u = torch.empty(T, dtype=torch.float32, device=DEVICE)
            for start in range(0, T, VALUE_CHUNK):
                chunk_states = states[start:start + VALUE_CHUNK]
                chunk_len = len(chunk_states)
                packed_chunk, _packed_op_mask = pack_states(chunk_states, n_op_global,
                                                          n_mac_global)
                h_op_chunk, _h_mac_chunk = self.gat(packed_chunk)
                g_embs_chunk = h_op_chunk.reshape(chunk_len, n_op_global, -1).mean(dim=1)
                if self.state_norm is not None:
                    g_embs_chunk = self.state_norm.normalize(g_embs_chunk)
                values_u[start:start + chunk_len] = self.critic_u(g_embs_chunk).squeeze(-1)
                del packed_chunk, h_op_chunk, g_embs_chunk

            # Generalized advantage estimation, in the form the replaced loop used: the same
            # backwards recursion, kept on the device.
            next_values = torch.cat([values_u[1:], torch.zeros(1, device=DEVICE)])
            deltas = rewards + self.gamma * next_values - values_u
            advantages = torch.empty_like(deltas)
            running = torch.zeros((), device=DEVICE)
            for step in reversed(range(T)):
                running = deltas[step] + self.gamma * self.lam * running
                advantages[step] = running
            returns = values_u + advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        # Update in the structure of the method the sweep measures: several epochs over the
        # collected episode, walked in chunks whose gradients accumulate into one optimiser step,
        order = np.arange(T)
        for _epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(order)
            for start in range(0, T, UPDATE_CHUNK):
                chunk = order[start:start + UPDATE_CHUNK]
                chunk_len = len(chunk)
                if chunk_len == 0:
                    continue
                # The loss of a chunk is the mean over its states, and its gradient is the sum of
                # the per-state gradients divided by the length of the chunk.  Walking the chunk in
                self.opt.zero_grad()
                for group_start in range(0, chunk_len, UPDATE_SUBCHUNK):
                    group = chunk[group_start:group_start + UPDATE_SUBCHUNK]
                    group_len = len(group)
                    positions = torch.as_tensor(group, dtype=torch.long, device=DEVICE)
                    rows = torch.arange(group_len, device=DEVICE)
                    group_packed, group_op_mask = pack_states(
                        [states[int(position)] for position in group], n_op_global,
                        n_mac_global)

                    h_op_all, h_mac_all = self.gat(group_packed)
                    batch_h_op = h_op_all.reshape(group_len, n_op_global, -1)
                    batch_h_mac = h_mac_all.reshape(group_len, n_mac_global, -1)
                    g_embs = batch_h_op.mean(dim=1)
                    if self.state_norm is not None:
                        g_embs = self.state_norm.normalize(g_embs)

                    group_op_indices = op_indices[positions]
                    group_mac_indices = mac_indices[positions]
                    batch_op_masks = group_op_mask.view(group_len, n_op_global)

                    curr_u_probs_batch = self.upper(g_embs, batch_h_op, batch_op_masks)
                    curr_u_prob = curr_u_probs_batch[rows, group_op_indices]

                    # The machine mask of the chosen operation and the heuristic term of the lower
                    # policy (Eq. 10), both gathered from the padded tables of the instance.  The
                    group_valid = (group_op_indices < inst.total_ops).to(batch_h_op.dtype)
                    group_safe = group_op_indices.clamp(max=inst.total_ops - 1)
                    batch_mac_masks_op = padded_mac_mask[group_safe] * group_valid.unsqueeze(1)
                    batch_pt_rows = torch.zeros((group_len, n_mac_global), device=DEVICE)
                    batch_load_rows = torch.zeros((group_len, n_mac_global), device=DEVICE)
                    batch_wait_rows = torch.zeros((group_len, n_mac_global), device=DEVICE)
                    batch_pt_rows[:, :inst.n_machines] = (inst.pt_rows_norm[group_safe]
                                                          * group_valid.unsqueeze(1))
                    group_loads = torch.stack([states[int(position)]['machine_load']
                                               for position in group])
                    group_avails = torch.stack([states[int(position)]['machine_available_time']
                                                for position in group])
                    batch_load_rows[:, :inst.n_machines] = (
                        group_loads
                        / (group_loads.sum(dim=1, keepdim=True) + EPS)
                        * group_valid.unsqueeze(1))
                    batch_wait_rows[:, :inst.n_machines] = (
                        group_avails
                        / (group_avails.max(dim=1, keepdim=True).values + 1.0 + EPS)
                        * group_valid.unsqueeze(1))
                    lower_heuristic = -(W1_DEFAULT * batch_pt_rows
                                        + W2_DEFAULT * batch_load_rows
                                        + W3_DEFAULT * batch_wait_rows)

                    selected_op_emb = batch_h_op[rows, group_op_indices]
                    curr_l_probs_batch = self.lower(g_embs, selected_op_emb, batch_h_mac,
                                                    batch_mac_masks_op, lower_heuristic)
                    curr_l_prob = curr_l_probs_batch[rows, group_mac_indices]

                    group_old_u_probs = old_u_probs[positions]
                    group_old_l_probs = old_l_probs[positions]
                    adv = advantages[positions]

                    ratio_u = curr_u_prob / (group_old_u_probs + EPS)
                    ratio_l = curr_l_prob / (group_old_l_probs + EPS)

                    surr1_u = ratio_u * adv
                    surr2_u = torch.clamp(ratio_u, 1-self.clip, 1+self.clip) * adv
                    loss_u = -torch.min(surr1_u, surr2_u)

                    surr1_l = ratio_l * adv
                    surr2_l = torch.clamp(ratio_l, 1-self.clip, 1+self.clip) * adv
                    loss_l = -torch.min(surr1_l, surr2_l)

                    kl_u = torch.log(group_old_u_probs / (curr_u_prob + EPS) + EPS)
                    kl_l = torch.log(group_old_l_probs / (curr_l_prob + EPS) + EPS)

                    v_pred = self.critic_u(g_embs).squeeze(-1)
                    # Mean squared error over the states of the group.
                    loss_v = (v_pred - returns[positions]) ** 2

                    # The sum over the group divided by the length of the chunk, so that the groups
                    # of one chunk leave the gradient of the mean over that chunk.
                    group_loss = ((loss_u + loss_l + 0.5 * loss_v
                                   + self.beta * (kl_u + kl_l)).sum() / chunk_len)
                    group_loss.backward()

                # Gradient clip of the family.  The 0.5 this script used damps every update forty
                # times harder than the clip of the runs of the comparison and damps the direction
                # of the gradient instead of only its length.
                torch.nn.utils.clip_grad_norm_(self.params, MAX_GRAD_NORM)
                self.opt.step()

# Evaluation function
# Vectorised collection
def train_collector(agent, cfg, next_instance, n_episodes, geometry, num_envs=None,
                    ppo_epochs=PPO_EPOCHS, label="", report_every=100):
    """Train ``n_episodes`` episodes of one configuration with a batch of environments.

    ``next_instance(index)`` returns the instance of the next episode to start, where the index is
    the number of episodes the collector has started.  A schedule that depends on the progress of
    the run therefore sees the same sequence it saw while the episodes were collected one after
    the other.  ``geometry`` is the tuple the environments are built with.

    The batch is a throughput device and not a change of the recipe: an environment runs one
    episode, the episode is updated as soon as it ends and the environment takes the next one, so
    a configuration still takes one gradient structure per episode and the same number of
    updates as its episode budget.
    """
    num_envs = NUM_ENVS if num_envs is None else max(1, int(num_envs))
    max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date = geometry

    envs, insts, states, trajectories = [], [], [], []
    started = 0
    for _ in range(num_envs):
        instance = next_instance(started)
        started += 1
        env = HeteroGraphEnv(instance, max_jobs, max_total_ops, max_machines,
                             max_proc_time, max_due_date, cfg)
        envs.append(env)
        insts.append(instance)
        states.append(env.reset())
        trajectories.append([])

    finished = 0
    failures = 0
    started_at = time.time()
    while finished < n_episodes:
        try:
            actions, probs_list, indices_list = agent.get_action_batch(
                states, insts, deterministic=False, check_invalid=False)
        except Exception as exc:  # noqa: BLE001 - a batch that cannot be decoded is rebuilt
            # A failure that disappears after the batch is rebuilt is a state that cannot be
            # decoded any more; one that survives a hundred rebuilds is a defect of the code or of
            # the configuration, and retrying it would spin the loop while the scan waits.
            failures += 1
            print("    the batch could not be decoded (%s); rebuilding it" % exc, flush=True)
            if failures >= 100:
                raise
            for index in range(num_envs):
                instance = next_instance(started)
                started += 1
                envs[index] = HeteroGraphEnv(instance, max_jobs, max_total_ops, max_machines,
                                             max_proc_time, max_due_date, cfg)
                insts[index] = instance
                states[index] = envs[index].reset()
                trajectories[index] = []
            continue
        failures = 0

        for index in range(num_envs):
            next_state, reward, done = envs[index].step(actions[index])
            trajectories[index].append({
                'state': states[index],
                'action': actions[index],
                'reward': reward.item(),
                'old_probs': probs_list[index],
                'indices': indices_list[index],
            })
            states[index] = next_state
            if not done:
                continue

            # The episode of this environment ended: update it with the instance it was collected
            # on and give the environment the next episode of the schedule.
            trajectory = trajectories[index]
            trajectories[index] = []
            if trajectory:
                for _ in range(ppo_epochs):
                    agent.update(trajectory, insts[index])
            finished += 1
            if label and report_every and (finished % report_every == 0
                                          or finished == n_episodes):
                rate = (time.time() - started_at) / max(1, finished)
                print("    %s: episode %d/%d (%.2f s/ep)"
                      % (label, finished, n_episodes, rate), flush=True)
            if finished >= n_episodes:
                return
            instance = next_instance(started)
            started += 1
            envs[index] = HeteroGraphEnv(instance, max_jobs, max_total_ops, max_machines,
                                         max_proc_time, max_due_date, cfg)
            insts[index] = instance
            states[index] = envs[index].reset()


def evaluate(agent, test_instances, global_max_jobs, global_max_ops, global_max_machines, global_max_proc_time, global_max_due_date,
             cfg, deterministic=True):
    """
    Evaluates agent performance on test instances.
    test_instances: list of MOFJSPInstance objects

    The decode is the greedy one of the policy, one rollout per instance, and that frame is the one
    every row of this scan is measured in: the scan reports how the objective of this model answers
    a parameter, not the absolute quality of a method.  The comparison experiment of the study
    measures a different frame - the best of one greedy pass and fifty sampled rollouts per instance
    - so its figures and the ones below differ by construction and are not comparable in absolute
    terms.
    """
    agent.gat.eval()
    agent.upper.eval()
    agent.lower.eval()
    agent.critic_u.eval()
    agent.critic_l.eval()
    F_list = []
    Cmax_list = []
    LB_list = []
    Tardy_list = []

    with torch.no_grad():
        # Iterate directly over MOFJSPInstance objects, no unpacking needed
        for inst in tqdm(test_instances, desc="Evaluating", leave=False):
            agent.inst = inst  # Set current instance for get_action
            # The reported objective of the comparison keeps the best of one deterministic rollout
            # and fifty sampled ones, so a configuration has to be measured the same way here: a
            def new_env():
                return HeteroGraphEnv(inst, global_max_jobs, global_max_ops,
                                      global_max_machines, global_max_proc_time,
                                      global_max_due_date, cfg)

            def objective_of(env):
                Cmax = max(env.job_completion_time).item()
                LB = (torch.std(env.machine_load).item()
                      if env.machine_load.numel() > 0 else 0.0)
                Tardy = torch.sum(torch.clamp(env.job_completion_time - env.due_dates_tensor,
                                              min=0)).item()
                return cfg.w1 * Cmax + cfg.w2 * LB + cfg.w3 * Tardy, Cmax, LB, Tardy

            if EVAL_BATCH <= 1:
                # The sequential form, kept byte for byte so that a scan can be reproduced against
                # the frame it was measured in when the batched form is in doubt.
                best = None
                for run_index in range(EVAL_ROLLOUTS):
                    env = new_env()
                    state = env.reset()
                    done = False
                    run_deterministic = (run_index == 0)
                    while not done:
                        action, _, _ = agent.get_action(state,
                                                       deterministic=run_deterministic)
                        next_state, _, done = env.step(action)
                        state = next_state
                    candidate = objective_of(env)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            else:
                # (1) the greedy rollout, which is the canonical inference of the policy
                env = new_env()
                state = env.reset()
                done = False
                while not done:
                    actions, _, _ = agent.get_action_batch([state], [inst], deterministic=True, check_invalid=False)
                    next_state, _, done = env.step(actions[0])
                    state = next_state
                best = objective_of(env)

                # (2) the sampled rollouts, EVAL_BATCH of them advanced in lockstep.  Every rollout
                # still runs one episode from a reset of its own, and the set of the objectives is
                # the set the protocol keeps the best of.
                for start in range(1, EVAL_ROLLOUTS, EVAL_BATCH):
                    width = min(EVAL_BATCH, EVAL_ROLLOUTS - start)
                    envs = [new_env() for _ in range(width)]
                    states = [item.reset() for item in envs]
                    finished = [False] * width
                    while not all(finished):
                        live = [index for index in range(width) if not finished[index]]
                        actions, _, _ = agent.get_action_batch(
                            [states[index] for index in live], [inst] * len(live),
                            deterministic=False, check_invalid=False)
                        for slot, index in enumerate(live):
                            next_state, _, done = envs[index].step(actions[slot])
                            states[index] = next_state
                            finished[index] = bool(done)
                    for item in envs:
                        candidate = objective_of(item)
                        if candidate[0] < best[0]:
                            best = candidate
            F_val, Cmax, LB, Tardy = best
            F_list.append(F_val)
            Cmax_list.append(Cmax)
            LB_list.append(LB)
            Tardy_list.append(Tardy)

    # The per-instance objectives travel with the means: the sweep reports the spread of a
    # configuration and tests the best against the runner-up, which needs the samples and not
    # only their average.
    return np.mean(F_list), np.mean(Cmax_list), np.mean(LB_list), np.mean(Tardy_list), F_list

# Parameter sensitivity scan main function
def parameter_sensitivity_scan():
    # Load all instances and split
    print("Loading instances...")
    all_instances, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = load_all_instances(INSTANCE_DIR, FILE_PATTERN)
    random.shuffle(all_instances)
    total = len(all_instances)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)
    train_tuples = all_instances[:train_end]
    val_tuples = all_instances[train_end:val_end]
    test_tuples = all_instances[val_end:]
    print(f"Training set: {len(train_tuples)}, Validation set: {len(val_tuples)}, Test set: {len(test_tuples)}")
    print(f"Global max jobs: {max_jobs}, max machines: {max_machines}, max total operations: {max_total_ops}, max processing time: {max_proc_time}, max due date: {max_due_date}, max capability: {max_capability}")

    # Wrap instances
    train_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability) for jobs, caps, dues, fname, size in train_tuples]
    val_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability) for jobs, caps, dues, fname, size in val_tuples]
    test_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability) for jobs, caps, dues, fname, size in test_tuples]

    # Geometry every environment of the scan is built with, and the schedule of the training
    # instances.  The pool is drawn uniformly, exactly as the single environment loop drew it: the
    geometry = (max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date)

    # Curriculum of the main run and of the weight study.  The small instances carry the first
    # half of the budget, the medium ones join in the third quarter and the whole pool is reached
    small_instances = [inst for inst in train_instances
                       if getattr(inst, "size", "") == "small"] or train_instances
    medium_instances = [inst for inst in train_instances
                        if getattr(inst, "size", "") == "medium"]
    large_instances = [inst for inst in train_instances
                       if getattr(inst, "size", "") == "large"]

    def draw_training_instance(episode_index):
        if FORCE_LARGE and large_instances:
            return random.choice(large_instances)
        progress = episode_index / max(1, NUM_EPOCHS)
        if progress < 0.5 or not medium_instances:
            pool = small_instances
        elif progress < 0.75 or not large_instances:
            pool = small_instances + medium_instances
        else:
            pool = small_instances + medium_instances + large_instances
        return random.choice(pool)

    dim_op = 3
    dim_mac = 3
    # Width of the encoder of every configuration.  It was 64, i.e. sixteen times smaller than the
    # network the comparison measures, so the sweep could not report the sensitivity of the method
    enc_width = 256
    hidden = 512
    out = 256
    # The step size of the recipe: every configuration of the sweep has to start from the same
    # operating point as the method the sweep quantifies.
    lr = 2e-4

    # Define configuration class
    class Config:
        def __init__(self, w1, w2, w3, gamma, beta, K):
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3
            self.gamma = gamma
            self.beta = beta
            self.K = K
            self.reward_scaling = REWARD_SCALING
            self.reward_clip = REWARD_CLIP

    param_configs = {
        # The weights of the objectives are not scanned here.  Which weighting is the best one, and
        # whether the three objectives have to be traded against each other at all, is the question of
        'w': [],
        # The band of the configuration: 0.2 is the value the penalty starts with and 0.05 its
        # floor, so the scan covers both ends of the schedule and the small values the measured
        # divergence of a run (0.0001 to 0.008) sits in.
        'beta': [0.001, 0.005, 0.01, 0.02, 0.05, 0.2],
        # One replaces the 0.999: the measured optimum of this host is the undiscounted return,
        # and 0.999 is indistinguishable from it while costing one full training.
        'gamma': [0.90, 0.95, 0.99, 0.995, 1.0],
        'K': [1, 2, 4, 8],
        # The step size of the recipe, centred on the 2e-4 the method is trained with: it decides how
        # far one update moves, and the two points a factor of two either side are what such a curve is
        # read for.
        'lr': [1e-4, 2e-4, 5e-4],
        # The clip of the reward, centred on the 2.0 of the recipe.  It bounds both the shaped reward
        # and the terminal objective, so too small a value truncates the signal a configuration can
        # learn from and too large a one lets a single outlier dominate the bootstrap.
        'reward_clip': [1.0, 2.0, 5.0]
    }

    # Values selected by the current recipe.  They are marked on every curve and compared with the
    # best point of the sweep, so the study can support the choice instead of only ranking values.
    CURRENT_VALUES = {
        'beta': [0.01, 0.2],          # KL floor and initial penalty of the adaptive schedule
        'gamma': [1.0],
        'K': [4],
        'lr': [2e-4],
        'reward_clip': [2.0],
    }

    # Parallel scan.  One configuration is trained by a single process and the host has idle cores
    # while the device carries the training, so the five parameters are scanned by five processes
    _PARAM_ONLY = [name.strip() for name in os.environ.get("PARAM_ONLY", "").split(",") if name.strip()]
    # The same list can also come from a command line switch: an entry of the boot list (which starts
    # every component of the wave) can only pass arguments and not variables, and the five parameters
    import sys as _sys

    _only_argument = ""
    for _index, _argument in enumerate(_sys.argv):
        if _argument == "--only" and _index + 1 < len(_sys.argv):
            _only_argument = _sys.argv[_index + 1]
    if _only_argument:
        _PARAM_ONLY = [name.strip() for name in _only_argument.split(",") if name.strip()]
    _value_argument = ""
    for _index, _argument in enumerate(_sys.argv):
        if _argument == "--value" and _index + 1 < len(_sys.argv):
            _value_argument = _sys.argv[_index + 1]
    _PARAM_ASSEMBLE = os.environ.get("PARAM_ASSEMBLE", "") not in ("", "0")
    if _PARAM_ASSEMBLE:
        for _name in list(param_configs):
            param_configs[_name] = []
    elif _PARAM_ONLY:
        for _name in list(param_configs):
            if _name not in _PARAM_ONLY:
                param_configs[_name] = []

    if _value_argument and not _PARAM_ASSEMBLE:
        def _matches_value(candidate):
            try:
                return abs(float(candidate) - float(_value_argument)) <= 1e-12
            except (TypeError, ValueError):
                return str(candidate) == _value_argument
        _targets = _PARAM_ONLY or list(param_configs)
        kept = 0
        for _name in _targets:
            _grid = [value for value in param_configs[_name] if _matches_value(value)]
            param_configs[_name] = _grid
            kept += len(_grid)
        print("--value %s matched %d grid entries" % (_value_argument, kept), flush=True)

    results = {
        'w': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'beta': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'gamma': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'K': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'lr': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'reward_clip': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []}
    }

    if PARAM_ASSEMBLE_FROM_CACHE:
        for _name in ('beta', 'gamma', 'K', 'lr', 'reward_clip'):
            _rows = _load_value_cache(_name)
            for _row in sorted(_rows.values(), key=lambda item: float(item['value'])):
                results[_name]['values'].append(_row['value'])
                for _key in ('F', 'Cmax', 'LB', 'Tardy', 'epochs', 'sd', 'F_list'):
                    results[_name].setdefault(_key, []).append(_row[_key])
            param_configs[_name] = []
        param_configs['w'] = []

    # Compute total number of scans for progress bar
    total_scans = (len(param_configs['w']) + len(param_configs['beta'])
                    + len(param_configs['gamma']) + len(param_configs['K'])
                    + len(param_configs['lr']) + len(param_configs['reward_clip']))
    pbar_scan = tqdm(total=total_scans, desc="Parameter scanning", position=0)

    # Scan weights w
    print("\n========== Scanning multi-objective weights w ==========")
    for w_dict in param_configs['w']:
        w1, w2, w3 = w_dict['w1'], w_dict['w2'], w_dict['w3']
        print(f"Training w=({w1:.2f},{w2:.2f},{w3:.2f})")
        cfg = Config(w1, w2, w3, gamma=1.0, beta=0.2, K=4)
        gat = HeteroGAT(dim_op, dim_mac, enc_width, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                      gamma=cfg.gamma, beta=cfg.beta, lr=lr, lam=1.0,
                                      state_norm=state_norm)

        train_collector(agent, cfg, draw_training_instance, NUM_EPOCHS, geometry,
                        label="w=(%.2f,%.2f,%.2f)" % (w1, w2, w3))
        # Evaluate
        F, C, L, T_val, F_list = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['w']['values'].append(f"({w1},{w2},{w3})")
        results['w']['F'].append(F)
        results['w']['Cmax'].append(C)
        results['w']['LB'].append(L)
        results['w']['Tardy'].append(T_val)
        results['w']['epochs'].append(NUM_EPOCHS)
        results['w'].setdefault('sd', []).append(float(np.std(F_list)))
        results['w'].setdefault('F_list', []).append([float(v) for v in F_list])
        pbar_scan.update(1)

    # Scan beta
    print("\n========== Scanning KL penalty coefficient beta ==========")
    for beta in param_configs['beta']:
        if reuse_cached_row(results['beta'], 'beta', beta):
            pbar_scan.update(1)
            continue
        print(f"Training beta={beta}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=1.0, beta=beta, K=4)
        gat = HeteroGAT(dim_op, dim_mac, enc_width, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                      gamma=cfg.gamma, beta=cfg.beta, lr=lr, lam=1.0,
                                      state_norm=state_norm)
        train_collector(agent, cfg, draw_training_instance, NUM_EPOCHS, geometry,
                        label="beta=%s" % beta)
        if PARAM_PROBE:
            print("probe complete", flush=True)
            return
        F, C, L, T_val, F_list = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['beta']['values'].append(beta)
        results['beta']['F'].append(F)
        results['beta']['Cmax'].append(C)
        results['beta']['LB'].append(L)
        results['beta']['Tardy'].append(T_val)
        results['beta']['epochs'].append(NUM_EPOCHS)
        results['beta'].setdefault('sd', []).append(float(np.std(F_list)))
        results['beta'].setdefault('F_list', []).append([float(v) for v in F_list])
        remember_row('beta', beta, results['beta'])
        del agent, gat, upper, lower, critic_u, critic_l, state_norm
        release_device()
        pbar_scan.update(1)

    # Scan gamma
    print("\n========== Scanning discount factor gamma ==========")
    for gamma in param_configs['gamma']:
        if reuse_cached_row(results['gamma'], 'gamma', gamma):
            pbar_scan.update(1)
            continue
        print(f"Training gamma={gamma}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=gamma, beta=0.2, K=4)
        gat = HeteroGAT(dim_op, dim_mac, enc_width, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                      gamma=cfg.gamma, beta=cfg.beta, lr=lr, lam=1.0,
                                      state_norm=state_norm)
        train_collector(agent, cfg, draw_training_instance, NUM_EPOCHS, geometry,
                        label="gamma=%s" % gamma)
        if PARAM_PROBE:
            print("probe complete", flush=True)
            return
        F, C, L, T_val, F_list = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['gamma']['values'].append(gamma)
        results['gamma']['F'].append(F)
        results['gamma']['Cmax'].append(C)
        results['gamma']['LB'].append(L)
        results['gamma']['Tardy'].append(T_val)
        results['gamma']['epochs'].append(NUM_EPOCHS)
        results['gamma'].setdefault('sd', []).append(float(np.std(F_list)))
        results['gamma'].setdefault('F_list', []).append([float(v) for v in F_list])
        remember_row('gamma', gamma, results['gamma'])
        del agent, gat, upper, lower, critic_u, critic_l, state_norm
        release_device()
        pbar_scan.update(1)

    # Scan K
    print("\n========== Scanning number of attention heads K ==========")
    for K in param_configs['K']:
        if out % K != 0:
            print(f"Warning: out_dim={out} not divisible by K={K}, skipping")
            pbar_scan.update(1)
            continue
        if reuse_cached_row(results['K'], 'K', K):
            pbar_scan.update(1)
            continue
        print(f"Training K={K}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=1.0, beta=0.2, K=K)
        gat = HeteroGAT(dim_op, dim_mac, enc_width, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                      gamma=cfg.gamma, beta=cfg.beta, lr=lr, lam=1.0,
                                      state_norm=state_norm)
        train_collector(agent, cfg, draw_training_instance, NUM_EPOCHS, geometry,
                        label="K=%s" % K)
        if PARAM_PROBE:
            print("probe complete", flush=True)
            return
        F, C, L, T_val, F_list = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['K']['values'].append(K)
        results['K']['F'].append(F)
        results['K']['Cmax'].append(C)
        results['K']['LB'].append(L)
        results['K']['Tardy'].append(T_val)
        results['K']['epochs'].append(NUM_EPOCHS)
        results['K'].setdefault('sd', []).append(float(np.std(F_list)))
        results['K'].setdefault('F_list', []).append([float(v) for v in F_list])
        remember_row('K', K, results['K'])
        del agent, gat, upper, lower, critic_u, critic_l, state_norm
        release_device()
        pbar_scan.update(1)

    # Scan learning rate
    print("\n========== Scanning learning rate ==========")
    for lr_value in param_configs['lr']:
        if reuse_cached_row(results['lr'], 'lr', lr_value):
            pbar_scan.update(1)
            continue
        print(f"Training learning rate={lr_value}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=1.0, beta=0.2, K=4)
        gat = HeteroGAT(dim_op, dim_mac, enc_width, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                      gamma=cfg.gamma, beta=cfg.beta, lr=lr_value, lam=1.0,
                                      state_norm=state_norm)
        train_collector(agent, cfg, draw_training_instance, NUM_EPOCHS, geometry,
                        label="lr=%s" % lr_value)
        if PARAM_PROBE:
            print("probe complete", flush=True)
            return
        F, C, L, T_val, F_list = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['lr']['values'].append(lr_value)
        results['lr']['F'].append(F)
        results['lr']['Cmax'].append(C)
        results['lr']['LB'].append(L)
        results['lr']['Tardy'].append(T_val)
        results['lr']['epochs'].append(NUM_EPOCHS)
        results['lr'].setdefault('sd', []).append(float(np.std(F_list)))
        results['lr'].setdefault('F_list', []).append([float(v) for v in F_list])
        remember_row('lr', lr_value, results['lr'])
        del agent, gat, upper, lower, critic_u, critic_l, state_norm
        release_device()
        pbar_scan.update(1)

    # Scan reward clip
    print("\n========== Scanning reward clip ==========")
    for clip in param_configs['reward_clip']:
        if reuse_cached_row(results['reward_clip'], 'reward_clip', clip):
            pbar_scan.update(1)
            continue
        print(f"Training reward clip={clip}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=1.0, beta=0.2, K=4)
        cfg.reward_clip = clip
        gat = HeteroGAT(dim_op, dim_mac, enc_width, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                      gamma=cfg.gamma, beta=cfg.beta, lr=lr, lam=1.0,
                                      state_norm=state_norm)
        train_collector(agent, cfg, draw_training_instance, NUM_EPOCHS, geometry,
                        label="reward_clip=%s" % clip)
        if PARAM_PROBE:
            print("probe complete", flush=True)
            return
        F, C, L, T_val, F_list = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['reward_clip']['values'].append(clip)
        results['reward_clip']['F'].append(F)
        results['reward_clip']['Cmax'].append(C)
        results['reward_clip']['LB'].append(L)
        results['reward_clip']['Tardy'].append(T_val)
        results['reward_clip']['epochs'].append(NUM_EPOCHS)
        results['reward_clip'].setdefault('sd', []).append(float(np.std(F_list)))
        results['reward_clip'].setdefault('F_list', []).append([float(v) for v in F_list])
        remember_row('reward_clip', clip, results['reward_clip'])
        del agent, gat, upper, lower, critic_u, critic_l, state_norm
        release_device()
        pbar_scan.update(1)

    pbar_scan.close()

    if PARAM_PROBE or PARAM_VALUE_ONLY:
        print("probe complete", flush=True)
        return

    # Plot curves
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False
    # Six panels for five sweeps: the weights are measured by the multi-objective weight study, and the
    # step size and the reward clip joined the three hyper-parameters of the recipe.
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    param_names = ['beta', 'gamma', 'K', 'lr', 'reward_clip']
    titles = ['KL penalty coefficient β', 'Discount factor γ', 'Number of attention heads K',
              'Learning rate', 'Reward clip $r_c$']
    # The clip of the reward is reported as r_c with c as a subscript, the symbol the manuscript uses
    # for it.  Its key stays reward_clip, which names its CSV and its cache entry, so the caching and
    _reloaded = set()
    for _param in param_names:
        if results[_param]['values']:
            continue
        _path = os.path.join(FIGURE_DIR, 'sensitivity_%s.csv' % _param)
        if not os.path.isfile(_path):
            continue
        try:
            _frame = pd.read_csv(_path)
            results[_param] = {
                'values': list(_frame['Value']),
                'F': [float(v) for v in _frame['Avg F']],
                'Cmax': [float(v) for v in _frame['Avg Cmax']],
                'LB': [float(v) for v in _frame['Avg LB']],
                'Tardy': [float(v) for v in _frame['Avg Tardy']],
                'epochs': [int(v) for v in _frame['Epochs']],
                'sd': [float(v) for v in _frame['F sd']],
                'F_list': [],
            }
            _reloaded.add(_param)
        except Exception as _exc:  # noqa: BLE001 - a CSV that cannot be read is scanned instead
            print("    %s: the CSV could not be read (%s)" % (_param, _exc))
    for idx, (param, ax) in enumerate(zip(param_names, axes.flatten())):
        values = results[param]['values']
        C_vals = results[param]['Cmax']
        L_vals = results[param]['LB']
        T_vals = results[param]['Tardy']
        ax.plot(values, C_vals, marker='o', label='Cmax', color='blue')
        ax.plot(values, L_vals, marker='s', label='LB', color='orange')
        ax.plot(values, T_vals, marker='^', label='Tardy', color='green')
        for _i, _current in enumerate(CURRENT_VALUES.get(param, [])):
            ax.axvline(_current, color='crimson', linestyle='--', linewidth=1.1,
                       alpha=0.75, label='Current' if _i == 0 else None)
        ax.set_xlabel(titles[idx])
        ax.set_ylabel('Objective value')
        ax.legend()
        ax.grid(True)
        # The weights used to be plotted as categories here; that sweep is measured by its own study
        # now, so every sweep on this figure is numeric.
    plt.tight_layout()
    if _PARAM_ONLY:
        # A process that scans one parameter writes its figure under its own name: the assembled
        # figure of the study belongs to the run that found every parameter on disk.
        plt.savefig(os.path.join(FIGURE_DIR, 'parameter_sensitivity_%s.png'
                                 % "_".join(sorted(_PARAM_ONLY))), dpi=150)
    else:
        plt.savefig(os.path.join(FIGURE_DIR, 'parameter_sensitivity.png'), dpi=150)
    plt.show()

    print("\n\n========== Parameter Sensitivity Detection Results Summary ==========")
    for param in param_names:
        if not results[param]['values']:
            continue  # this process did not scan the parameter; its files belong to another one
        if param in _reloaded:
            print("\n--- %s (reloaded from its own CSV) ---" % param)
            continue
        print(f"\n--- {param} ---")
        df = pd.DataFrame({
            'Value': results[param]['values'],
            'Avg F': results[param]['F'],
            'Avg Cmax': results[param]['Cmax'],
            'Avg LB': results[param]['LB'],
            'Avg Tardy': results[param]['Tardy'],
            'Epochs': results[param]['epochs'],
            'IsCurrent': [any(abs(float(_v) - float(_c)) <= 1e-12
                            for _c in CURRENT_VALUES.get(param, []))
                           for _v in results[param]['values']],
            'F sd': results[param].get('sd', []),
            'F ci95': [1.96 * s / np.sqrt(max(1, len(v))) for s, v in
                       zip(results[param].get('sd', []), results[param].get('F_list', []))]
        })
        print(df.to_string(index=False))
        df.to_csv(os.path.join(FIGURE_DIR, f'sensitivity_{param}.csv'), index=False)

        # Key comparison of the parameter: the configuration with the lowest mean objective
        # against the runner-up, on the per-instance objectives of the same test set.  A sweep
        def _cliffs(a, b):
            """Cliff's delta of two samples: positive means the first tends to be larger."""
            greater = lesser = 0
            for x in a:
                for y in b:
                    if x > y:
                        greater += 1
                    elif x < y:
                        lesser += 1
            total = len(a) * len(b)
            return (greater - lesser) / total if total else 0.0

        lists = results[param].get('F_list', [])
        means = results[param]['F']
        if len(lists) >= 2 and all(len(entry) > 1 for entry in lists):
            order = sorted(range(len(means)), key=lambda i: means[i])
            best_i, second_i = order[0], order[1]
            try:
                from scipy.stats import wilcoxon
                _, p_value = wilcoxon(lists[best_i], lists[second_i])
            except Exception as exc:  # noqa: BLE001 - the report survives a missing scipy
                p_value = float('nan')
                print(f"    wilcoxon unavailable: {exc}")
            delta = _cliffs(lists[best_i], lists[second_i])
            magnitude = abs(delta)
            effect = ('negligible' if magnitude < 0.147 else
                      'small' if magnitude < 0.33 else
                      'medium' if magnitude < 0.474 else 'large')
            pair = "%s vs %s" % (results[param]['values'][best_i],
                                 results[param]['values'][second_i])
            print("    best %s | p=%.4g | cliffs_delta=%+.3f (%s)"
                  % (pair, p_value, delta, effect))
            pd.DataFrame({'pair': [pair], 'p_value': [p_value],
                          'cliffs_delta': [delta], 'effect': [effect]}).to_csv(
                os.path.join(FIGURE_DIR, f'sensitivity_{param}_test.csv'), index=False)

        # Compare the values selected by the current recipe with the best point of the sweep.
        # This is the table that supports (or rejects) the choice, rather than just ranking values.
        current_rows = []
        if lists and all(len(entry) > 1 for entry in lists):
            order = sorted(range(len(means)), key=lambda i: means[i])
            best_i = order[0]
            best_mean = means[best_i]
            for current_value in CURRENT_VALUES.get(param, []):
                current_i = min(range(len(results[param]['values'])),
                                key=lambda i: abs(float(results[param]['values'][i])
                                                 - float(current_value)))
                if current_i == best_i:
                    p_current = 1.0
                    delta_current = 0.0
                else:
                    try:
                        from scipy.stats import wilcoxon
                        _, p_current = wilcoxon(lists[current_i], lists[best_i])
                    except Exception:  # noqa: BLE001
                        p_current = float('nan')
                    delta_current = _cliffs(lists[current_i], lists[best_i])
                mag = abs(delta_current)
                effect_current = ('negligible' if mag < 0.147 else
                                  'small' if mag < 0.33 else
                                  'medium' if mag < 0.474 else 'large')
                rank_current = order.index(current_i) + 1
                gap_current = ((means[current_i] - best_mean) / best_mean * 100.0
                               if best_mean else float('nan'))
                if current_i == best_i:
                    conclusion = 'current value is the best point'
                elif not np.isnan(p_current) and p_current < 0.05 and effect_current != 'negligible':
                    conclusion = 'significant difference from the best point'
                else:
                    conclusion = 'statistically indistinguishable from the best point'
                current_rows.append({
                    'current_value': current_value,
                    'best_value': results[param]['values'][best_i],
                    'current_rank': rank_current,
                    'current_mean_F': means[current_i],
                    'best_mean_F': best_mean,
                    'gap_percent': gap_current,
                    'p_value': p_current,
                    'cliffs_delta': delta_current,
                    'effect': effect_current,
                    'conclusion': conclusion,
                })
                print("    current %.6g: rank=%d gap=%+.2f%% p=%.4g delta=%+.3f (%s) %s"
                      % (current_value, rank_current, gap_current, p_current,
                         delta_current, effect_current, conclusion))
            pd.DataFrame(current_rows).to_csv(
                os.path.join(FIGURE_DIR, f'sensitivity_{param}_current.csv'), index=False)
        else:
            print("    current-value comparison skipped: per-instance lists unavailable")

    print("\nPlot saved as parameter_sensitivity.png, tables saved as CSV files.")

if __name__ == "__main__":
    parameter_sensitivity_scan()




