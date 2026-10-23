"""
Homogeneous disjunctive graph and PPO+KL multi-objective flexible job shop scheduling solution (multi-instance training version)
- Completely fix index out-of-bounds issues: multiple safety guards in collate_states and GAT layers
- Stable reward design to avoid drastic fluctuations
- Gradient accumulation and conservative update strategy
- Enhanced multi-process robustness: timeout exit, worker liveness detection, queue exception handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
import glob
import warnings
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque
import matplotlib
# Without a display session (background run, stdout redirected to a file, headless server)
# the Qt backend cannot be initialised and the process would crash; fall back to the file-only
# Agg backend and keep the default backend for interactive runs.
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import re
import gc
import argparse
import multiprocessing as mp
from queue import Empty, Full
import signal

# Every artefact of this script is written under Figure_And_File and its
# checkpoint under model/, so that neither depends on the working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def allow_foreign_numpy_checkpoint():
    """Make a checkpoint of the other numpy major readable in this process.

    A checkpoint stores the module names of the numpy that wrote it.  The device build of this
    project runs numpy 2, which keeps its internals in ``numpy._core``, while the host build runs
    numpy 1, which keeps them in ``numpy.core``.  Reading a device checkpoint on the host therefore
    fails with ``No module named 'numpy._core'`` unless the two names are aliased to the same
    modules first.  The function does nothing on numpy 2, where those names already exist.
    """
    if hasattr(np, "_core"):
        return
    for name in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        module = getattr(np.core, name, None)
        if module is not None:
            sys.modules.setdefault("numpy._core." + name, module)
    sys.modules.setdefault("numpy._core", np.core)

FIGURE_DIR = os.path.join(_PROJECT_ROOT, "Figure_And_File", "PPO_GAT")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Attempt to import numba, fall back to pure Python if not available
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not installed, using pure Python version (may be slower)")

# Mixed precision
from torch.cuda.amp import autocast, GradScaler

# ----------------------------
# 1. Global configuration and constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10
W1, W2, W3 = 0.4, 0.3, 0.3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# 2. Configuration class (stable training settings)
# ----------------------------
class Config:
    n_episodes = 12000
    model_save_dir = os.path.join(_PROJECT_ROOT, "model", "ppo_gat")

    # ========== Stable hyperparameters ==========
    # The settings of this baseline were far away from the rest of the learned family, and it ranked
    lr = 5e-4                       # Learning rate
    # The schedule of the family: the step schedule halves the step size every two thousand
    # episodes.  The exponential 0.999 of this baseline halved it every seven hundred, i.e. it
    lr_decay = 0.5
    lr_decay_steps = 2000
    min_lr = 1e-5
    gamma = 0.99
    # Undiscounted return, see the note in PPO_HGAT.py: the objective of an instance is only
    # complete at the end of its schedule, and at gamma 0.99 the terminal signal of a 400 decision
    gamma = 1.0
    gae_lambda = 0.95
    # Monte-Carlo advantage, see the note in PPO_HGAT.py: a window of twenty steps cannot carry the
    # terminal objective of a 400 step episode, which is the objective this baseline never moved.
    gae_lambda = 1.0
    kl_target = 0.1                  # KL target of the family
    beta_start = 0.2                  # Initial beta of the family
    value_coef = 0.5                 # Value coefficient of the family
    # Entropy coefficient of this wave.  The reported objective of the comparison keeps the best of
    # fifty-one rollouts of an instance (one greedy pass and fifty samples), so a policy that keeps
    entropy_coef = 0.02               # Entropy coefficient
    entropy_decay = 0.9995
    # Gradient clip of the family.  The proposed method documents that the 20.0 of its collapsed
    # wave clipped the joint update hard enough to change the direction of the gradient instead of
    max_grad_norm = 1.0
    # Restored to the value of the family (ten).  The one was set to stop the value term of this
    # baseline from diverging to 1.1e6, but it clips the gradient of the policy as well, and the
    max_grad_norm = 10.0

    # Capacity of the family.  The 64/64/128 of this baseline is four to sixteen times smaller than
    # the encoders of HGNN and of the proposed method, so a review that compares the methods on the
    gat_hidden_dim = 256                # Network capacity of the family
    gat_out_dim = 256
    policy_hidden_dim = 512
    # Depth of this wave.  The re-measured shards put this baseline behind the graph baselines of
    # the same width at the small and the medium instances (measured 15x10: 1446 against 1115 of
    gat_num_layers = 3
    gat_num_heads = 4

    # The family trains without a value clip (see the recipe of the proposed method).
    value_clip = None
    # Enabled, for the reason the ablation study records when it enables the same key: the value
    # term of this family diverges without a bound and its gradient then dominates the encoder that
    value_clip = 0.5
    # Reward scaling of the family, so the critic of this baseline sees the same signal as the other
    # learned methods.
    reward_scaling = 5.0
    reward_clip = 2.0
    use_state_norm = True
    # This baseline schedules on the homogeneous graph of its own environment (see
    # HomogeneousDisjunctiveGraphEnv below); the heterogeneous machine-competition edges of the
    instance_dir = os.path.join(_PROJECT_ROOT, "DataSet", "mo_fjsp_instances")
    file_pattern = "mo_fjsp_*_train.txt"
    sampling_temperature = 1.0
    stats_window = 100
    max_steps_per_episode = 2000

    train_step_size = 256              # Training step threshold
    ppo_epochs = 1
    use_amp = False

    # Two workers feed the training batches.  Every run of this host shares the cores with
    # four sibling runs, so the pool is kept small: an uncapped pool thrashes the cores of the
    num_workers = 2
    # Cut from two to one.  The measurement above no longer holds: the two workers of this pool
    # grew to 3.4 GiB and 3.3 GiB (about 6.7 GiB together, beside the 0.7 GiB of the parent) while
    num_workers = 1
    # Back to two.  The one was a memory mitigation: with two workers this pool measured about
    # 6.7 GiB on 2026-10-11 while another training ran on the device and the host reached 97 %.
    num_workers = 2
    worker_queue_size = 2

    curriculum_enabled = True
    curriculum_stages = [
        (0.0, 0.3, ['small']),
        (0.3, 0.6, ['small', 'medium']),
        (0.6, 1.0, ['small', 'medium', 'large'])
    ]
    # Convergence control.  The reported weights are selected on a fixed validation set instead
    # of on the training return: the curriculum moves the instance distribution over the run, so
    val_interval = 250
    # Eight per size takes the complete held-out validation split (eight small, six medium and six
    # large files, i.e. the twenty instances of the paper), for every method alike.
    val_instances_per_size = 8
    val_seed = 0
    # A stale check is one validation of 250 episodes, so the old patience of eight held a
    # run 2000 episodes past its peak.
    val_patience = 8
    val_tol = 0.01
    # No early stop before this budget: the curriculum needs at least this many episodes to cover
    # the three instance sizes, but the window also has to open before the peak of a method.
    min_episodes = 1000
    # The curriculum advances on a fixed episode scale instead of on the --episodes budget, so that
    # the schedule does not change when the budget of the run is changed.
    curriculum_episodes = 4000
    # Floor of the entropy coefficient: the decay alone drives the coefficient to 1e-4 over a long
    # run, i.e. the policy becomes deterministic before the curriculum has finished, which is the
    entropy_min = 0.02
    # Rolling checkpoint an interrupted run continues from, its interval and the interval of the
    # progress lines.
    resume_name = 'ppo_gat_last.pth'
    checkpoint_interval = 25
    log_interval = 100

cfg = Config()

# ----------------------------
# 3. File reading module (unchanged)
# ----------------------------
@dataclass
class Operation:
    job_id: int
    op_id: int
    machine_times: Dict[int, float]

def parse_size_from_filename(filename: str) -> str:
    match = re.search(r'(small|medium|large)', filename)
    return match.group(1) if match else "unknown"

def read_fjsp_instance(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        sys.exit(1)

    num_jobs, num_machines = map(int, lines[0].split())
    machine_ids = list(range(num_machines))

    job_lines = lines[1:1 + num_jobs]
    due_date_line = lines[1 + num_jobs]
    due_dates = list(map(float, due_date_line.split()))

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
                machine_id = nums[idx] - 1
                proc_time = float(nums[idx + 1])
                machine_times[machine_id] = proc_time
                idx += 2
            op = Operation(job_idx, op_idx, machine_times)
            job_ops.append(op)
        jobs.append(job_ops)

    return jobs, machine_ids, due_dates

def load_all_instances(instance_dir: str, pattern: str):
    file_paths = glob.glob(os.path.join(instance_dir, pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files matching {pattern} found in directory {instance_dir}")

    instance_list = []
    max_jobs = 0
    max_machines = 0
    max_proc_time = 0
    max_due_date = 0

    for fp in file_paths:
        jobs, machine_ids, due_dates = read_fjsp_instance(fp)
        file_name = os.path.basename(fp)
        size = parse_size_from_filename(file_name)
        instance_list.append((jobs, machine_ids, due_dates, file_name, size))
        max_jobs = max(max_jobs, len(jobs))
        max_machines = max(max_machines, len(machine_ids))
        for job_ops in jobs:
            for op in job_ops:
                for t in op.machine_times.values():
                    max_proc_time = max(max_proc_time, t)
        max_due_date = max(max_due_date, max(due_dates) if due_dates else 0)

    return instance_list, max_jobs, max_machines, max_proc_time, max_due_date

# ----------------------------
# 4. MOFJSP instance wrapper class
# ----------------------------
class MOFJSPInstance:
    def __init__(self, jobs, machine_ids, due_dates, file_name, size):
        self.jobs = jobs
        self.machine_ids = machine_ids
        self.due_dates = due_dates
        self.file_name = file_name
        self.size = size
        self.n_jobs = len(jobs)
        self.n_machines = len(machine_ids)
        self.total_ops = sum(len(job_ops) for job_ops in jobs)
        self.max_proc_time = 0
        for job_ops in jobs:
            for op in job_ops:
                self.max_proc_time = max(self.max_proc_time, max(op.machine_times.values()))
        self.max_due_date = max(due_dates) if due_dates else 1.0

# ----------------------------
# 5. State normalization tool (CPU version)
# ----------------------------
class TorchRunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4, device=DEVICE):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.device = device
        self.mean.requires_grad_(False)
        self.var.requires_grad_(False)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.device = device
        return self

    def update(self, x: torch.Tensor):
        x = x.float()
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
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)

# ----------------------------
# 6. Numba acceleration functions (unchanged)
# ----------------------------
if HAS_NUMBA:
    @jit(nopython=True)
    def update_op_features_numba(node_features, unsched_indices, step_idx_vals, due_time_vals,
                                 job_id_norm_vals, ready_time_vals, is_available_vals):
        for i in range(len(unsched_indices)):
            idx = unsched_indices[i]
            node_features[idx, 0] = step_idx_vals[i]
            node_features[idx, 1] = due_time_vals[i]
            node_features[idx, 2] = job_id_norm_vals[i]
            node_features[idx, 3] = ready_time_vals[i]
            node_features[idx, 4] = is_available_vals[i]
        return node_features

    @jit(nopython=True)
    def update_mac_features_numba(node_features, mac_indices, load_vals, avail_vals,
                                  mac_id_norm_vals):
        for i in range(len(mac_indices)):
            idx = mac_indices[i]
            node_features[idx, 0] = mac_id_norm_vals[i]
            node_features[idx, 1] = load_vals[i]
            node_features[idx, 2] = avail_vals[i]
        return node_features
else:
    # Pure Python fallback
    def update_op_features_numba(node_features, unsched_indices, step_idx_vals, due_time_vals,
                                 job_id_norm_vals, ready_time_vals, is_available_vals):
        for i in range(len(unsched_indices)):
            idx = unsched_indices[i]
            node_features[idx, 0] = step_idx_vals[i]
            node_features[idx, 1] = due_time_vals[i]
            node_features[idx, 2] = job_id_norm_vals[i]
            node_features[idx, 3] = ready_time_vals[i]
            node_features[idx, 4] = is_available_vals[i]
        return node_features

    def update_mac_features_numba(node_features, mac_indices, load_vals, avail_vals,
                                  mac_id_norm_vals):
        for i in range(len(mac_indices)):
            idx = mac_indices[i]
            node_features[idx, 0] = mac_id_norm_vals[i]
            node_features[idx, 1] = load_vals[i]
            node_features[idx, 2] = avail_vals[i]
        return node_features

# ----------------------------
# 7. Environment class (stable reward design)
# ----------------------------
class HomogeneousDisjunctiveGraphEnv:
    def __init__(self, instance: MOFJSPInstance, global_max_jobs, global_max_machines,
                 global_max_proc_time, global_max_due_date, cfg):
        self.inst = instance
        self.global_max_jobs = global_max_jobs
        self.global_max_machines = global_max_machines
        self.global_max_proc_time = global_max_proc_time
        self.global_max_due_date = global_max_due_date
        self.cfg = cfg
        self.action_dim = global_max_jobs * global_max_machines

        # Precompute static information
        self.op_machines = {}
        self.op_processing_times = {}
        self.job_op_lists = {}
        for job in range(instance.n_jobs):
            ops = []
            for op in range(len(instance.jobs[job])):
                ops.append((job, op))
                machines = list(instance.jobs[job][op].machine_times.keys())
                self.op_machines[(job, op)] = machines
                for m in machines:
                    self.op_processing_times[(job, op, m)] = instance.jobs[job][op].machine_times[m]
            self.job_op_lists[job] = ops

        self.total_op_nodes = instance.total_ops
        self.total_machine_nodes = instance.n_machines
        self.total_nodes = self.total_op_nodes + self.total_machine_nodes
        self.feature_dim = 8

        # Node index mapping
        self.op_to_idx = {}
        self.job_op_to_idx = np.full((instance.n_jobs, max(len(j) for j in instance.jobs)), -1, dtype=np.int64)
        self.idx_to_job = [0] * self.total_op_nodes
        self.idx_to_op  = [0] * self.total_op_nodes
        idx = 0
        for job in range(instance.n_jobs):
            for op in range(len(instance.jobs[job])):
                self.op_to_idx[(job, op)] = idx
                self.job_op_to_idx[job, op] = idx
                self.idx_to_job[idx] = job
                self.idx_to_op[idx] = op
                idx += 1
        self.mac_to_idx = {m: self.total_op_nodes + m for m in range(instance.n_machines)}

        # Precompute static edges
        src_all, tgt_all, feat_all = [], [], []
        self.op_to_edge_indices = {}
        for job in range(instance.n_jobs):
            op_list = self.job_op_lists[job]
            for i in range(len(op_list)-1):
                op1, op2 = op_list[i], op_list[i+1]
                src, tgt = self.op_to_idx[op1], self.op_to_idx[op2]
                edge_idx = len(src_all)
                src_all.append(src)
                tgt_all.append(tgt)
                feat_all.append(1.0)
                self.op_to_edge_indices.setdefault(op1, []).append(edge_idx)
                self.op_to_edge_indices.setdefault(op2, []).append(edge_idx)
        for (job, op) in self.op_to_idx.keys():
            src = self.op_to_idx[(job, op)]
            for m in self.op_machines[(job, op)]:
                tgt = self.mac_to_idx[m]
                feat = self.op_processing_times[(job, op, m)] / (global_max_proc_time + EPS)
                edge_idx = len(src_all)
                src_all.append(src)
                tgt_all.append(tgt)
                feat_all.append(feat)
                self.op_to_edge_indices.setdefault((job, op), []).append(edge_idx)

        self.src_all = np.array(src_all, dtype=np.int64)
        self.tgt_all = np.array(tgt_all, dtype=np.int64)
        self.feat_all = np.array(feat_all, dtype=np.float32)
        self.edge_valid = np.ones(len(src_all), dtype=bool)

        # Precompute static features
        self.static_step_idx = np.zeros(self.total_op_nodes, dtype=np.float32)
        self.static_due_time = np.zeros(self.total_op_nodes, dtype=np.float32)
        self.static_job_id_norm = np.zeros(self.total_op_nodes, dtype=np.float32)
        for (job, op), idx in self.op_to_idx.items():
            self.static_step_idx[idx] = op / max(1, len(instance.jobs[job]) - 1)
            self.static_due_time[idx] = instance.due_dates[job] / (global_max_due_date + EPS)
            self.static_job_id_norm[idx] = job / max(1, global_max_jobs - 1)

        self.static_mac_id_norm = np.zeros(instance.n_machines, dtype=np.float32)
        for m in range(instance.n_machines):
            self.static_mac_id_norm[m] = m / max(1, global_max_machines - 1)

        self.reset()

    def reset(self):
        self.jobs = self.inst.n_jobs
        self.machines = self.inst.n_machines
        self.scheduled_ops = []
        self.unscheduled_op_indices = set(range(self.total_op_nodes))
        self.op_scheduled = np.zeros(self.total_op_nodes, dtype=bool)
        self.machine_available_time = np.zeros(self.machines, dtype=np.float32)
        self.machine_load = np.zeros(self.machines, dtype=np.float32)
        self.job_next_op_idx = np.zeros(self.jobs, dtype=np.int32)
        self.job_completion_time = np.zeros(self.jobs, dtype=np.float32)
        self.current_time = 0.0
        self.node_features = np.zeros((self.total_nodes, self.feature_dim), dtype=np.float32)
        self.edge_valid[:] = True
        self.job_last_finish = np.zeros(self.jobs, dtype=np.float32)
        self.total_immediate_reward = 0.0
        self.step_count = 0
        return self._get_state()

    def get_available_ops(self):
        available = []
        for job in range(self.jobs):
            next_op = self.job_next_op_idx[job]
            if next_op < len(self.inst.jobs[job]):
                available.append((job, next_op))
        return available

    def get_action_mask(self):
        mask = torch.zeros(self.action_dim, dtype=torch.float32)
        for (job, op) in self.get_available_ops():
            for m in self.op_machines[(job, op)]:
                idx = job * self.global_max_machines + m
                mask[idx] = 1.0
        return mask

    def step(self, action_idx: int):
        job = action_idx // self.global_max_machines
        machine = action_idx % self.global_max_machines
        op = self.job_next_op_idx[job]
        p_time = self.op_processing_times[(job, op, machine)]

        prev_finish = self.job_last_finish[job] if op > 0 else 0.0
        start_time = max(prev_finish, self.machine_available_time[machine])
        end_time = start_time + p_time

        self.scheduled_ops.append((job, op, machine, start_time, end_time))
        op_key = (job, op)
        op_idx = self.op_to_idx[op_key]
        if op_idx in self.unscheduled_op_indices:
            self.unscheduled_op_indices.remove(op_idx)
        self.op_scheduled[op_idx] = True
        self.machine_available_time[machine] = end_time
        self.machine_load[machine] += p_time
        self.job_next_op_idx[job] = op + 1
        if self.job_next_op_idx[job] == len(self.inst.jobs[job]):
            self.job_completion_time[job] = end_time
        self.current_time = max(self.current_time, end_time)

        self.job_last_finish[job] = end_time

        if op_key in self.op_to_edge_indices:
            for e_idx in self.op_to_edge_indices[op_key]:
                self.edge_valid[e_idx] = False

        w = [W1, W2, W3]

        # Stable reward design
        norm_p = p_time / (self.global_max_proc_time + EPS)
        machine_load_ratio = self.machine_load[machine] / (np.sum(self.machine_load) + EPS)
        load_penalty = machine_load_ratio
        due = self.inst.due_dates[job]
        due_proximity = max(0, due - self.current_time) / (self.global_max_due_date + EPS)

        immediate_reward = - (0.4 * norm_p + 0.3 * load_penalty + 0.3 * (1 - due_proximity))
        self.total_immediate_reward += immediate_reward
        self.step_count += 1

        done = (len(self.unscheduled_op_indices) == 0)
        if done:
            metrics = self.get_final_metrics()
            cmax_norm = metrics['Cmax'] / (self.global_max_proc_time * self.inst.total_ops + EPS)
            lb_norm = metrics['LB'] / (np.mean(self.machine_load) + EPS)
            tardy_norm = metrics['Tardy'] / (self.global_max_due_date * self.inst.n_jobs + EPS)
            final_reward = - (w[0] * cmax_norm + w[1] * lb_norm + w[2] * tardy_norm)
            reward = self.total_immediate_reward / max(1, self.step_count) + final_reward
        else:
            reward = immediate_reward

        reward = np.clip(reward, -2.0, 0.5)

        next_state = self._get_state() if not done else None
        return next_state, float(reward), done, {}

    def get_final_metrics(self):
        Cmax = self.current_time
        loads = self.machine_load
        if loads.size > 0:
            mean_load = np.mean(loads)
            LB = np.sqrt(np.mean((loads - mean_load) ** 2))
        else:
            LB = 0.0
        tardy = 0
        for job in range(self.jobs):
            due = self.inst.due_dates[job]
            comp = self.job_completion_time[job]
            tardy += max(0, comp - due)
        return {'Cmax': Cmax, 'LB': LB, 'Tardy': tardy}

    def _get_state(self):
        available_ops = set(self.get_available_ops())

        unsched_op_indices = list(self.unscheduled_op_indices)
        n_unsched = len(unsched_op_indices)
        if n_unsched > 0:
            step_idx_vals = np.zeros(n_unsched, dtype=np.float32)
            due_time_vals = np.zeros(n_unsched, dtype=np.float32)
            job_id_norm_vals = np.zeros(n_unsched, dtype=np.float32)
            ready_time_vals = np.zeros(n_unsched, dtype=np.float32)
            is_available_vals = np.zeros(n_unsched, dtype=np.float32)

            for i, op_idx in enumerate(unsched_op_indices):
                job = self.idx_to_job[op_idx]
                op = self.idx_to_op[op_idx]
                step_idx_vals[i] = self.static_step_idx[op_idx]
                due_time_vals[i] = self.static_due_time[op_idx]
                job_id_norm_vals[i] = self.static_job_id_norm[op_idx]
                if op > 0:
                    ready_time_vals[i] = self.job_last_finish[job]
                else:
                    ready_time_vals[i] = 0.0
                is_available_vals[i] = 1.0 if (job, op) in available_ops else 0.0

            self.node_features = update_op_features_numba(
                self.node_features, np.array(unsched_op_indices, dtype=np.int64),
                step_idx_vals, due_time_vals, job_id_norm_vals,
                ready_time_vals, is_available_vals
            )

        if self.machines > 0:
            total_load = self.machine_load.sum()
            max_load = max(total_load, 1.0)
            mac_indices = np.array([self.mac_to_idx[m] for m in range(self.machines)], dtype=np.int64)
            load_vals = self.machine_load / max_load
            avail_vals = self.machine_available_time / (self.current_time + 1.0 + EPS)
            mac_id_norm_vals = self.static_mac_id_norm

            self.node_features = update_mac_features_numba(
                self.node_features, mac_indices, load_vals, avail_vals, mac_id_norm_vals
            )

        valid = np.where(self.edge_valid)[0]
        if len(valid) > 0:
            src = self.src_all[valid]
            tgt = self.tgt_all[valid]
            feat = self.feat_all[valid]
        else:
            src = tgt = np.array([], dtype=np.int64)
            feat = np.array([], dtype=np.float32)

        op_node_idx, mac_node_idx = self.action_node_indices()

        return {
            'node_features': torch.from_numpy(self.node_features).float(),
            'edge_src': torch.from_numpy(src).long(),
            'edge_tgt': torch.from_numpy(tgt).long(),
            'edge_features': torch.from_numpy(feat).float().unsqueeze(-1),
            'action_mask': self.get_action_mask(),
            'op_node_idx': op_node_idx,
            'mac_node_idx': mac_node_idx
        }

    def action_node_indices(self):
        """Node index of the operation and of the machine behind every entry of the action grid.

        The policy scores an entry ``job * global_max_machines + machine`` from the embeddings of
        the two nodes it refers to - the next operation of that job and that machine - so the state
        has to carry the mapping from the action grid to the nodes of the graph.  A job whose
        operations are all scheduled and a machine the instance does not own keep index 0; the
        action mask of the state already forbids their entries of the grid, so the embedding they
        resolve to never reaches the score of a feasible action.
        """
        op_node_idx = np.zeros(self.global_max_jobs, dtype=np.int64)
        for job in range(self.jobs):
            next_op = self.job_next_op_idx[job]
            if next_op < len(self.inst.jobs[job]):
                op_node_idx[job] = self.job_op_to_idx[job, next_op]
        mac_node_idx = np.zeros(self.global_max_machines, dtype=np.int64)
        for machine in range(self.machines):
            mac_node_idx[machine] = self.mac_to_idx[machine]
        return (torch.from_numpy(op_node_idx).long(),
                torch.from_numpy(mac_node_idx).long())

# ----------------------------
# 8. GAT network (with multiple index safety guards)
# ----------------------------
def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer

class HomogeneousGATLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(num_heads, 3 * self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        orthogonal_init(self.W, gain=nn.init.calculate_gain('relu'))
        orthogonal_init(self.W_edge, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, edge_src, edge_tgt, edge_feat):
        num_nodes = h.shape[0]
        if num_nodes == 0:
            return torch.zeros_like(h)

        # ---- Safe index preprocessing ----
        if edge_src.numel() > 0:
            # Ensure correct types
            edge_src = edge_src.long()
            edge_tgt = edge_tgt.long()
            # Clip to valid range
            max_idx = num_nodes - 1
            edge_src = torch.clamp(edge_src, 0, max_idx)
            edge_tgt = torch.clamp(edge_tgt, 0, max_idx)
            # Double-check, filter if any still invalid (should not happen)
            valid = (edge_src >= 0) & (edge_src < num_nodes) & (edge_tgt >= 0) & (edge_tgt < num_nodes)
            if not valid.all():
                print(f"GATLayer: filtering {valid.logical_not().sum()} invalid edges after clamp.")
                edge_src = edge_src[valid]
                edge_tgt = edge_tgt[valid]
                edge_feat = edge_feat[valid]

        Wh = self.W(h).view(-1, self.num_heads, self.head_dim)

        # If no valid edges, return linear transform
        if edge_feat.shape[0] == 0:
            return F.relu(Wh.view(-1, self.out_dim))

        # ---- Safe index operations ----
        try:
            We = self.W_edge(edge_feat).view(-1, self.num_heads, self.head_dim)
            # Double-check indices are still in bounds (prevent race conditions)
            if edge_src.max() >= num_nodes or edge_tgt.max() >= num_nodes:
                print(f"GATLayer CRITICAL: index still out of bounds after filtering, max_src={edge_src.max()}, max_tgt={edge_tgt.max()}, num_nodes={num_nodes}. Falling back to linear.")
                return F.relu(Wh.view(-1, self.out_dim))

            h_src = Wh[edge_src]
            h_tgt = Wh[edge_tgt]
            concat = torch.cat([h_src, h_tgt, We], dim=-1)
            e = self.leaky_relu(torch.einsum('hd,ehd->eh', self.a, concat))
            alpha = F.softmax(e, dim=0)
            alpha = self.dropout(alpha)
            weighted_src = h_src * alpha.unsqueeze(-1)

            new_feat = torch.zeros_like(Wh, dtype=weighted_src.dtype)
            new_feat.index_add_(0, edge_tgt, weighted_src)
            new_feat = F.relu(new_feat.view(-1, self.out_dim))
            new_feat = self.ln(new_feat)
            return new_feat
        except Exception as e:
            print(f"GATLayer attention failed: {e}, falling back to linear.")
            return F.relu(Wh.view(-1, self.out_dim))

class HomogeneousGAT(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HomogeneousGATLayer(in_dim, edge_dim, hidden_dim, num_heads, dropout))
        for _ in range(num_layers - 1):
            self.layers.append(HomogeneousGATLayer(hidden_dim, edge_dim, hidden_dim, num_heads, dropout))
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        orthogonal_init(self.out_proj[0], gain=nn.init.calculate_gain('relu'))
        orthogonal_init(self.out_proj[3], gain=1.0)

    def forward(self, state_repr, node_counts=None):
        h = state_repr['node_features']
        src, tgt, feat = state_repr['edge_src'], state_repr['edge_tgt'], state_repr['edge_features']
        for layer in self.layers:
            h = layer(h, src, tgt, feat)
        if node_counts is None:
            global_emb = h.mean(dim=0)
            global_emb = self.out_proj(global_emb)
            return global_emb, h
        else:
            split_h = torch.split(h, node_counts, dim=0)
            global_embs = [self.out_proj(sub_h.mean(dim=0)) for sub_h in split_h]
            return torch.stack(global_embs, dim=0), h

# ----------------------------
# 9. Policy and value networks
JOINT_LOGITS_CLIP = 20.0

class PolicyNetwork(nn.Module):
    """Joint (job, machine) policy head scored from the node embeddings of the encoder.

    The head of the previous wave was an MLP from the permutation invariant graph embedding to the
    ``global_max_jobs * global_max_machines`` flat action logits.  Every entry of the grid owned a
    weight vector of its own, so the head had to memorise the identity of a hundred jobs and thirty
    machines from the pooled state alone.  The validation objective of that wave stayed at its
    first check for the whole run, i.e. the baseline never improved over its randomly initialised
    policy in three thousand episodes, which is what this head replaces.

    The score used here is the bilinear form of the joint action space,
    ``<q(h_op(job)), k(h_mac(machine))>``, plus one bias per side, both conditioned on the graph
    embedding.  The decoder is still a single softmax over the joint (job, machine) grid, so the
    baseline keeps the flat, non hierarchical action space that the ablation of the proposed
    method attributes to it; what changes is that the parameters that discriminate the actions are
    shared through the node embeddings of the encoder instead of being stored per action index.
    """

    def __init__(self, emb_dim, hidden_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.op_query = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU())
        self.mac_key = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU())
        self.op_bias = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 2, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1))
        self.mac_bias = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 2, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1))
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def joint_logits(self, global_emb, h_ops, h_macs):
        """Flat logits of the joint action grid, laid out as ``job * n_machines + machine``.

        ``global_emb`` holds one embedding per state, ``h_ops`` the embedding of the next operation
        of every job and ``h_macs`` the embedding of every machine.  All three accept a leading
        batch dimension, so the single state of a rollout and a batch of states of an update share
        one code path.
        """
        n_jobs = h_ops.shape[-2]
        n_machines = h_macs.shape[-2]
        leading = global_emb.shape[:-1]
        query = self.op_query(h_ops) * self.scale
        key = self.mac_key(h_macs)
        logits = query @ key.transpose(-1, -2)
        g_op = global_emb.unsqueeze(-2).expand(*leading, n_jobs, global_emb.shape[-1])
        g_mac = global_emb.unsqueeze(-2).expand(*leading, n_machines, global_emb.shape[-1])
        logits = logits + self.op_bias(torch.cat([g_op, h_ops], dim=-1))
        logits = logits + self.mac_bias(torch.cat([g_mac, h_macs], dim=-1)).transpose(-1, -2)
        return logits.reshape(*leading, n_jobs * n_machines)

    def forward(self, global_emb, h_ops, h_macs, action_mask):
        logits = self.joint_logits(global_emb, h_ops, h_macs).float() / self.temperature
        feasible = action_mask > 0.5
        # Shift the feasible logits by their maximum before clipping them: softmax is shift
        # invariant, so the relative differences between the actions survive, whereas the fixed
        if bool(feasible.any()):
            shifted = logits.masked_fill(~feasible, -1e9)
            logits = logits - shifted.max(dim=-1, keepdim=True).values
        logits = torch.clamp(logits, -JOINT_LOGITS_CLIP, JOINT_LOGITS_CLIP)
        logits = logits.masked_fill(~feasible, -1e9)
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or not bool(feasible.any()):
            # A state without a feasible action has no meaningful distribution; the callers replace
            # this fallback by the uniform distribution over the actions their mask does allow.
            probs = torch.ones_like(probs) / probs.shape[-1]
        return probs

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            nn.Dropout(0.1),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            nn.Dropout(0.1),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)
        )
    def forward(self, state_emb):
        return self.net(state_emb).squeeze(-1)

# ----------------------------
# 10. Helper functions (completely fix index out-of-bounds)
# ----------------------------
def collate_states(states):
    """
    Combine states of multiple graphs, strictly filter edges for each graph to ensure merged indices are completely legal.
    """
    node_feats = torch.cat([s['node_features'] for s in states], dim=0)
    node_counts = [s['node_features'].shape[0] for s in states]

    offset = 0
    edge_src, edge_tgt, edge_feats = [], [], []

    for i, s in enumerate(states):
        num_nodes_i = node_counts[i]
        src_i = s['edge_src'].clone()
        tgt_i = s['edge_tgt'].clone()
        feat_i = s['edge_features'].clone()

        if src_i.numel() > 0:
            # Strictly check and filter edge indices
            valid_mask = (src_i >= 0) & (src_i < num_nodes_i) & (tgt_i >= 0) & (tgt_i < num_nodes_i)
            if not valid_mask.all():
                # Keep only valid edges
                src_i = src_i[valid_mask]
                tgt_i = tgt_i[valid_mask]
                feat_i = feat_i[valid_mask]
            # If the graph has no valid edges, skip
            if src_i.numel() > 0:
                # Add offset
                src = src_i + offset
                tgt = tgt_i + offset
                edge_src.append(src)
                edge_tgt.append(tgt)
                edge_feats.append(feat_i)
        offset += num_nodes_i

    if edge_src:
        edge_src = torch.cat(edge_src, dim=0)
        edge_tgt = torch.cat(edge_tgt, dim=0)
        edge_feats = torch.cat(edge_feats, dim=0)
    else:
        edge_src = torch.empty(0, dtype=torch.long)
        edge_tgt = torch.empty(0, dtype=torch.long)
        edge_feats = torch.empty((0, 1))

    # Final check after merging
    if edge_src.numel() > 0:
        max_allowed = node_feats.shape[0] - 1
        final_valid = (edge_src >= 0) & (edge_src <= max_allowed) & (edge_tgt >= 0) & (edge_tgt <= max_allowed)
        if not final_valid.all():
            print(f"collate_states: final filtering {final_valid.logical_not().sum()} edges.")
            edge_src = edge_src[final_valid]
            edge_tgt = edge_tgt[final_valid]
            edge_feats = edge_feats[final_valid]

    return {'node_features': node_feats,
            'edge_src': edge_src,
            'edge_tgt': edge_tgt,
            'edge_features': edge_feats}, node_counts

# ----------------------------
# 11. PPO trainer (with exception handling)
# ----------------------------
class PPO_KL_Trainer:
    def __init__(self, gat, policy, value, cfg, state_norm=None):
        self.gat, self.policy, self.value = gat, policy, value
        self.cfg = cfg
        self.state_norm = state_norm
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.kl_target = cfg.kl_target
        self.beta = cfg.beta_start
        self.value_coef = cfg.value_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.value_clip = cfg.value_clip

        params = list(gat.parameters()) + list(policy.parameters()) + list(value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay)
        self.model_dir = cfg.model_save_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_reward = -np.inf
        self.best_model_path = os.path.join(self.model_dir, "ppo_gat_best.pth")
        self.kl_ema = None
        self.kl_alpha = 0.9
        self.stats_window = cfg.stats_window
        self.recent = {k: deque(maxlen=self.stats_window) for k in ['policy_loss','value_loss','entropy','kl','beta','advantage','value']}
        self.gradient_accumulation_steps = 2
        # The weights are loaded by train(): the rolling checkpoint is preferred over the best
        # model, and only the caller knows which of the two exists.

    def _checkpoint_payload(self):
        """Return the keys of a checkpoint; the best model and the rolling one share them.

        The comparison experiment loads the best model through the keys below, so a new writer of
        the same file has to keep every one of them.
        """
        return {
            'gat': self.gat.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'beta': self.beta,
            'entropy_coef': self.entropy_coef,
            'best_reward': self.best_reward,
            'state_norm_mean': self.state_norm.mean if self.state_norm else None,
            'state_norm_var': self.state_norm.var if self.state_norm else None,
            'state_norm_count': self.state_norm.count if self.state_norm else None,
        }

    def _save_model(self, reward=None):
        """Write the best model when ``reward`` improves the best training return seen so far."""
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            _atomic_path = self.best_model_path + ".tmp"
            torch.save(self._checkpoint_payload(), _atomic_path)
            os.replace(_atomic_path, self.best_model_path)
            print(f"New best model saved with reward {reward:.4f}")

    def save_best(self, reward=None):
        """Write the best model unconditionally.

        The reported weights are the ones the validation set selected, so the write must not depend
        on the training return of a single episode, which the curriculum makes incomparable.
        """
        if reward is not None and np.isfinite(reward):
            self.best_reward = max(self.best_reward, reward)
        _atomic_path = self.best_model_path + ".tmp"
        torch.save(self._checkpoint_payload(), _atomic_path)
        os.replace(_atomic_path, self.best_model_path)
        print(f"Best model saved (validation selection, reward {self.best_reward:.4f})")

    def save_rolling(self, path, episode, val_curve, val_reference, best_val, converged=False):
        """Write the rolling checkpoint an interrupted run continues from.

        The weights, the state of the optimiser and the validation record are stored together, so a
        restart repeats at most ``checkpoint_interval`` episodes instead of the whole run.
        """
        payload = self._checkpoint_payload()
        payload.update({
            'episode_counter': int(episode),
            'val_curve': [tuple(entry) for entry in val_curve],
            'val_reference': list(val_reference) if val_reference else [],
            'best_val': float(best_val),
            'converged': bool(converged),
        })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_path = path + ".tmp"
        torch.save(payload, _atomic_path)
        os.replace(_atomic_path, path)

    def load_rolling(self, path):
        """Continue an interrupted run and return its checkpoint, or None when there is none.

        A missing or unreadable checkpoint returns None so that the caller starts a fresh run.
        """
        if not os.path.exists(path):
            return None
        try:
            allow_foreign_numpy_checkpoint()
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            self.gat.load_state_dict(ckpt['gat'])
            self.policy.load_state_dict(ckpt['policy'])
            self.value.load_state_dict(ckpt['value'])
            self.optimizer.load_state_dict(ckpt['optim'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.beta = ckpt.get('beta', self.beta)
            self.entropy_coef = ckpt.get('entropy_coef', self.entropy_coef)
            self.best_reward = ckpt.get('best_reward', self.best_reward)
            if self.state_norm and ckpt.get('state_norm_mean') is not None:
                self.state_norm.mean = ckpt['state_norm_mean'].to(DEVICE)
                self.state_norm.var = ckpt['state_norm_var'].to(DEVICE)
                self.state_norm.count = ckpt['state_norm_count']
            return ckpt
        except Exception as exc:  # noqa: BLE001 - a broken checkpoint starts a fresh run
            print(f"Rolling checkpoint {path} could not be read: {exc}")
            return None

    def _load_model(self):
        if os.path.exists(self.best_model_path):
            try:
                allow_foreign_numpy_checkpoint()
                ckpt = torch.load(self.best_model_path, map_location=DEVICE, weights_only=False)
                self.gat.load_state_dict(ckpt['gat'])
                self.policy.load_state_dict(ckpt['policy'])
                self.value.load_state_dict(ckpt['value'])
                self.optimizer.load_state_dict(ckpt['optim'])
                self.scheduler.load_state_dict(ckpt['scheduler'])
                self.beta = ckpt.get('beta', self.beta)
                self.entropy_coef = ckpt.get('entropy_coef', self.entropy_coef)
                if self.state_norm and 'state_norm_mean' in ckpt and ckpt['state_norm_mean'] is not None:
                    self.state_norm.mean = ckpt['state_norm_mean'].to(DEVICE)
                    self.state_norm.var = ckpt['state_norm_var'].to(DEVICE)
                    self.state_norm.count = ckpt['state_norm_count']
                print(f"Loaded model from {self.best_model_path}")
            except Exception as e:
                print(f"Load failed: {e}, using fresh parameters.")
        else:
            print("Starting training from scratch.")

    def compute_gae(self, rewards, values, next_values, dones):
        advantages, gae = [], 0
        T = len(rewards)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return torch.stack(advantages), torch.stack(returns)

    def train_step(self, trajectories):
        try:
            all_steps = [step for ep in trajectories for step in ep]
            T = len(all_steps)
            if T == 0:
                return {}

            actions = torch.tensor([s['action'] for s in all_steps], device=DEVICE, dtype=torch.long)
            rewards = torch.tensor([s['reward'] for s in all_steps], device=DEVICE, dtype=torch.float32)
            dones = torch.tensor([s['done'] for s in all_steps], device=DEVICE, dtype=torch.float32)
            old_logprobs = torch.tensor([s['logprob'] for s in all_steps], device=DEVICE, dtype=torch.float32)
            old_probs = torch.stack([s['old_probs'] for s in all_steps], dim=0).to(DEVICE)

            if torch.isnan(old_probs).any():
                old_probs = torch.nan_to_num(old_probs, nan=1.0/old_probs.shape[-1])

            states_cpu = [s['state_repr'] for s in all_steps]
            next_states_cpu = [s['next_state_repr'] for s in all_steps]

            batch_size = min(64, T)
            num_batches = (T + batch_size - 1) // batch_size

            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            total_kl = 0
            num_valid_batches = 0
            advantages_sum = 0.0
            returns_sum = 0.0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, T)

                batch_actions = actions[start_idx:end_idx]
                batch_rewards = rewards[start_idx:end_idx]
                batch_dones = dones[start_idx:end_idx]
                batch_old_logprobs = old_logprobs[start_idx:end_idx]
                batch_old_probs = old_probs[start_idx:end_idx]
                batch_states = states_cpu[start_idx:end_idx]
                batch_next_states = next_states_cpu[start_idx:end_idx]

                try:
                    with torch.no_grad():
                        batch_state_gpu = [self._to_device(s) for s in batch_states]
                        merged_state, node_counts = collate_states(batch_state_gpu)
                        global_emb, _ = self.gat(merged_state, node_counts=node_counts)
                        global_emb = global_emb.float()

                        if self.state_norm:
                            self.state_norm.update(global_emb)
                            emb_norm = self.state_norm.normalize(global_emb)
                        else:
                            emb_norm = global_emb / (global_emb.norm(dim=-1, keepdim=True) + 1e-8)

                        values_old = self.value(emb_norm)

                        non_done_indices = [i for i in range(len(batch_dones)) if not batch_dones[i].item()]
                        if non_done_indices:
                            non_done_states = [batch_next_states[i] for i in non_done_indices]
                            non_done_gpu = [self._to_device(s) for s in non_done_states]
                            next_merged, next_counts = collate_states(non_done_gpu)
                            next_emb, _ = self.gat(next_merged, node_counts=next_counts)
                            next_emb = next_emb.float()

                            if self.state_norm:
                                next_norm = self.state_norm.normalize(next_emb)
                            else:
                                next_norm = next_emb / (next_emb.norm(dim=-1, keepdim=True) + 1e-8)

                            next_values = self.value(next_norm)

                            next_values_full = torch.zeros(len(batch_dones), device=DEVICE)
                            for i, idx in enumerate(non_done_indices):
                                next_values_full[idx] = next_values[i]
                        else:
                            next_values_full = torch.zeros(len(batch_dones), device=DEVICE)

                    advantages, returns = self.compute_gae(batch_rewards, values_old, next_values_full, batch_dones)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    advantages = advantages.detach()
                    returns = returns.detach()
                    advantages_sum += advantages.mean().item()
                    returns_sum += returns.mean().item()

                    batch_state_gpu = [self._to_device(s) for s in batch_states]
                    merged_state, node_counts = collate_states(batch_state_gpu)
                    mask_batch = torch.stack([s['action_mask'] for s in batch_state_gpu], dim=0).to(DEVICE)

                    new_emb, new_nodes = self.gat(merged_state, node_counts=node_counts)
                    new_emb = new_emb.float()
                    new_norm = self.state_norm.normalize(new_emb) if self.state_norm else new_emb / (new_emb.norm(dim=-1, keepdim=True) + 1e-8)

                    # The graphs of a batch follow one another in the merged node matrix, so the node
                    # indices of a state are shifted by the number of nodes before it.  The next
                    shift = torch.tensor([0] + list(node_counts[:-1]), device=DEVICE,
                                         dtype=torch.long).cumsum(0).unsqueeze(1)
                    op_nodes = torch.stack([s['op_node_idx'] for s in batch_state_gpu], dim=0)
                    mac_nodes = torch.stack([s['mac_node_idx'] for s in batch_state_gpu], dim=0)
                    probs_new = self.policy(new_norm, new_nodes[op_nodes + shift],
                                            new_nodes[mac_nodes + shift], mask_batch)

                    if torch.isnan(probs_new).any():
                        continue

                    dist = torch.distributions.Categorical(probs=probs_new)
                    logprobs_new = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    values_new = self.value(new_norm)

                    # The KL of the family (the same estimator as the proposed method and the
                    # ablation trainer): the log ratio of the action that was actually taken,
                    old_action_probs = batch_old_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                    kl = (old_action_probs * (torch.log(old_action_probs + EPS) - logprobs_new)).mean()
                    if torch.isnan(kl) or torch.isinf(kl):
                        continue

                    ratio = torch.exp(logprobs_new - batch_old_logprobs)
                    ratio = torch.clamp(ratio, 0.1, 10.0)

                    surr1 = ratio * advantages
                    # The clip of the family is 0.2 (see the recipe of the proposed method); the
                    # fixed 0.1 of this baseline made every surrogate twice as conservative.
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean() + self.beta * kl

                    if self.value_clip:
                        val_clipped = values_old + torch.clamp(values_new - values_old, -self.value_clip, self.value_clip)
                        value_loss = 0.5 * torch.max(F.mse_loss(values_new, returns), F.mse_loss(val_clipped, returns))
                    else:
                        value_loss = F.mse_loss(values_new, returns)

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_kl += kl.item()
                    num_valid_batches += 1

                except RuntimeError as e:
                    print(f"Batch {batch_idx} error: {e}, skipping this batch.")
                    continue

            if num_valid_batches % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if num_valid_batches == 0:
                return {}

            avg_kl = total_kl / num_valid_batches
            self.kl_ema = avg_kl if self.kl_ema is None else self.kl_alpha * self.kl_ema + (1-self.kl_alpha) * avg_kl
            self.beta *= (1 + 0.05 * (self.kl_ema - self.kl_target))
            # The range of the family controller.  The ceiling used to be 0.2, which is exactly
            # beta_start: the adapted coefficient could then only decay and could never tighten the
            self.beta = np.clip(self.beta, 0.01, 10.0)

            self.scheduler.step()
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(pg['lr'], self.cfg.min_lr)

            avg_policy_loss = total_policy_loss / num_valid_batches
            avg_value_loss = total_value_loss / num_valid_batches
            avg_entropy = total_entropy / num_valid_batches
            avg_advantages = advantages_sum / num_valid_batches if num_valid_batches > 0 else 0
            avg_returns = returns_sum / num_valid_batches if num_valid_batches > 0 else 0

            self.recent['policy_loss'].append(avg_policy_loss)
            self.recent['value_loss'].append(avg_value_loss)
            self.recent['entropy'].append(avg_entropy)
            self.recent['kl'].append(avg_kl)
            self.recent['beta'].append(self.beta)
            self.recent['advantage'].append(avg_advantages)
            self.recent['value'].append(avg_returns)

            total_loss = avg_policy_loss + self.value_coef * avg_value_loss - self.entropy_coef * avg_entropy
            return {'kl': avg_kl, 'entropy': avg_entropy,
                    'policy_loss': avg_policy_loss, 'value_loss': avg_value_loss,
                    'beta': self.beta, 'total_loss': total_loss}

        except Exception as e:
            print(f"Fatal error in train_step: {e}, skipping this update.")
            return {}

    def parameters(self):
        return list(self.gat.parameters()) + list(self.policy.parameters()) + list(self.value.parameters())

    def get_stats(self):
        return {k: np.mean(v) if v else 0.0 for k, v in self.recent.items()}

    def _to_device(self, s):
        gpu_s = {}
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                gpu_s[k] = v.to(DEVICE, non_blocking=True)
            else:
                gpu_s[k] = v
        return gpu_s

# ----------------------------
# 12. Multi-process worker function
# ----------------------------
def worker_process(worker_id, instance_tuples, max_jobs, max_machines, max_proc_time,
                   max_due_date, cfg, param_queue, data_queue, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    seed = SEED + worker_id * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    instance_list = [MOFJSPInstance(*t) for t in instance_tuples]
    size_to_instances = {}
    for inst in instance_list:
        size_to_instances.setdefault(inst.size, []).append(inst)

    allowed_sizes = ['small']

    def get_allowed_instances():
        instances = []
        for sz in allowed_sizes:
            instances.extend(size_to_instances.get(sz, []))
        return instances

    gat = HomogeneousGAT(8, 1, cfg.gat_hidden_dim, cfg.gat_out_dim,
                         num_layers=cfg.gat_num_layers, num_heads=cfg.gat_num_heads).to('cpu')
    policy = PolicyNetwork(cfg.gat_out_dim, cfg.policy_hidden_dim, temperature=0.5).to('cpu')
    value = ValueNetwork(cfg.gat_out_dim, cfg.policy_hidden_dim).to('cpu')
    state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device='cpu') if cfg.use_state_norm else None

    # Wait for initial parameters from main process
    try:
        params = param_queue.get(timeout=30)
    except Empty:
        print(f"Worker {worker_id}: timeout waiting for initial params, exiting.")
        return

    gat.load_state_dict(params['gat'])
    policy.load_state_dict(params['policy'])
    value.load_state_dict(params['value'])
    if state_norm and 'state_norm_mean' in params:
        state_norm.mean = params['state_norm_mean'].to('cpu')
        state_norm.var = params['state_norm_var'].to('cpu')
        state_norm.count = params['state_norm_count']

    episode_count = 0
    local_trajectories = []
    nan_count = 0

    try:
        while not stop_event.is_set():
            allowed_instances = get_allowed_instances()
            if not allowed_instances:
                time.sleep(1)
                continue
            inst = random.choice(allowed_instances)

            try:
                env = HomogeneousDisjunctiveGraphEnv(inst, max_jobs, max_machines, max_proc_time, max_due_date, cfg)
                state = env.reset()
                done = False
                trajectory = []
                ep_reward = 0

                for step_cnt in range(1, cfg.max_steps_per_episode + 1):
                    with torch.no_grad():
                        global_emb, node_embs = gat(state)
                        global_emb = global_emb.float()
                        if state_norm:
                            emb_norm = state_norm.normalize(global_emb.unsqueeze(0)).squeeze(0)
                        else:
                            emb_norm = global_emb
                        mask = state['action_mask']
                        # The joint head scores an action from the embedding of the operation node and
                        # of the machine node it refers to, so the node embeddings of the graph are
                        # passed on together with the two index vectors the state carries.
                        probs = policy(emb_norm, node_embs[state['op_node_idx']],
                                       node_embs[state['mac_node_idx']], mask)

                        if torch.isnan(probs).any():
                            nan_count += 1
                            if nan_count > 10:
                                print(f"Worker {worker_id}: Repeated NaN ({nan_count} times)")
                            valid_actions = torch.nonzero(mask).squeeze(-1)
                            if len(valid_actions) == 0:
                                probs = torch.ones_like(probs) / probs.shape[-1]
                            else:
                                probs = torch.zeros_like(probs)
                                probs[valid_actions] = 1.0 / len(valid_actions)
                        else:
                            nan_count = 0

                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample().item()
                        logprob = dist.log_prob(torch.tensor(action)).item()

                    next_state, reward, done, _ = env.step(action)
                    ep_reward += reward

                    trajectory.append({
                        'state_repr': state,
                        'next_state_repr': next_state,
                        'action': action,
                        'reward': reward,
                        'done': done,
                        'logprob': logprob,
                        'old_probs': probs.cpu()
                    })
                    state = next_state
                    if done:
                        break

                episode_count += 1
                local_trajectories.append(trajectory)

            except Exception as e:
                print(f"Worker {worker_id} exception in episode {episode_count}: {e}")
                continue

            if len(local_trajectories) >= cfg.worker_queue_size:
                try:
                    data_queue.put(local_trajectories, timeout=5)
                    local_trajectories = []
                except (Full, OSError) as e:
                    print(f"Worker {worker_id}: data queue put failed: {e}, will retry later.")
                    time.sleep(1)

            # Check for new parameters or curriculum updates
            try:
                msg = param_queue.get_nowait()
                if 'sizes' in msg:
                    allowed_sizes = msg['sizes']
                    print(f"Worker {worker_id} curriculum updated to {allowed_sizes}")
                else:
                    gat.load_state_dict(msg['gat'])
                    policy.load_state_dict(msg['policy'])
                    value.load_state_dict(msg['value'])
                    if state_norm and 'state_norm_mean' in msg:
                        state_norm.mean = msg['state_norm_mean'].to('cpu')
                        state_norm.var = msg['state_norm_var'].to('cpu')
                        state_norm.count = msg['state_norm_count']
            except Empty:
                pass
            except Exception as e:
                print(f"Worker {worker_id} error processing param: {e}")

    finally:
        # Clean up resources
        del gat, policy, value, state_norm
        gc.collect()

    # Send remaining trajectories
    if local_trajectories:
        try:
            data_queue.put(local_trajectories, timeout=5)
        except:
            pass

    print(f"Worker {worker_id} finished after {episode_count} episodes.")

# ----------------------------
# 13. Main training function
# ----------------------------
def weighted_objective(objectives):
    """Weighted objective of one solution: [makespan, load balance, total tardiness]."""
    return W1 * objectives[0] + W2 * objectives[1] + W3 * objectives[2]


def build_validation_set(instance_list, per_size=2):
    """Fixed validation instances, ``per_size`` of every size group, picked by file name.

    The set never changes during a run, so the objective it reports is comparable between two
    checks.  The curriculum sampler draws a different instance at every episode instead, which
    makes the training return of two different episodes incomparable.
    """
    by_size = {}
    for inst in instance_list:
        by_size.setdefault(inst.size, []).append(inst)
    validation = []
    for size in ('small', 'medium', 'large'):
        bucket = sorted(by_size.get(size, []), key=lambda item: item.file_name)
        validation.extend(bucket[:per_size])
    return validation


def stage_for_progress(stages, progress):
    """Return the instance sizes of the curriculum stage that ``progress`` selects."""
    for start, end, sizes in stages:
        if start <= progress < end:
            return sizes
    return stages[-1][2] if stages else ['small', 'medium', 'large']


# One deterministic rollout plus k-1 sampled ones, and the best of that set scores an instance:
# that is the protocol the comparison reports every method with.  Selecting on the greedy pass
VALIDATION_ROLLOUTS = 4
def evaluate_validation(gat, policy, state_norm, validation, max_jobs, max_machines,
                        max_proc_time, max_due_date, cfg, seed=0):
    """Greedy rollouts of the validation instances; returns [Cmax, LB, Tardy] per instance.

    Greedy decoding is the canonical inference of the policy and it dominates the sampled
    trajectories, so the validation objective is measured on it instead of on a sample.
    """
    gat.eval()
    policy.eval()
    objectives = []
    for inst in validation:
        best = None
        for run_index in range(VALIDATION_ROLLOUTS):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = HomogeneousDisjunctiveGraphEnv(inst, max_jobs, max_machines, max_proc_time,
                                                 max_due_date, cfg)
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < cfg.max_steps_per_episode:
                with torch.no_grad():
                    global_emb, node_embs = gat(state)
                    global_emb = global_emb.float()
                    if state_norm is not None:
                        emb_norm = state_norm.normalize(global_emb.unsqueeze(0)).squeeze(0)
                    else:
                        emb_norm = global_emb
                    probs = policy(emb_norm, node_embs[state['op_node_idx']],
                                   node_embs[state['mac_node_idx']], state['action_mask'])
                    if torch.isnan(probs).any():
                        mask = state['action_mask']
                        valid_actions = torch.nonzero(mask).squeeze(-1)
                        if len(valid_actions) == 0:
                            probs = torch.ones_like(probs) / probs.shape[-1]
                        else:
                            probs = torch.zeros_like(probs)
                            probs[valid_actions] = 1.0 / len(valid_actions)
                    if run_index == 0:
                        action = int(torch.argmax(probs).item())
                    else:
                        action = int(torch.multinomial(probs, 1).item())
                state, _, done, _ = env.step(action)
                steps += 1
            metrics = env.get_final_metrics()
            candidate = [float(metrics['Cmax']), float(metrics['LB']),
                         float(metrics['Tardy'])]
            if best is None or weighted_objective(candidate) < weighted_objective(best):
                best = candidate
        objectives.append(best)
    gat.train()
    policy.train()
    return objectives


def train(cfg):
    print(f"Loading instances from {cfg.instance_dir}...")
    instance_tuples, max_jobs, max_machines, max_proc_time, max_due_date = load_all_instances(cfg.instance_dir,
                                                                                              cfg.file_pattern)
    print(
        f"Loaded {len(instance_tuples)} instances. max_jobs={max_jobs}, max_machines={max_machines}, max_proc_time={max_proc_time}, max_due_date={max_due_date}")

    # The reported weights are selected on a fixed validation set, one that the curriculum sampler
    # cannot move: the sampler changes the instance distribution over the run, so the training
    val_tuples = load_all_instances(cfg.instance_dir, "mo_fjsp_*_val.txt")[0]
    validation_objects = [MOFJSPInstance(*t) for t in val_tuples]
    validation_set = build_validation_set(
        validation_objects, per_size=int(getattr(cfg, 'val_instances_per_size', 2)))
    print(f"Validation set: {[inst.file_name for inst in validation_set]}")

    # Initialize networks
    gat = HomogeneousGAT(8, 1, cfg.gat_hidden_dim, cfg.gat_out_dim,
                         num_layers=cfg.gat_num_layers, num_heads=cfg.gat_num_heads).to(DEVICE)
    policy = PolicyNetwork(cfg.gat_out_dim, cfg.policy_hidden_dim, temperature=0.5).to(DEVICE)
    value = ValueNetwork(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
    state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=DEVICE) if cfg.use_state_norm else None
    trainer = PPO_KL_Trainer(gat, policy, value, cfg, state_norm)

    model_path = os.path.join(cfg.model_save_dir, "ppo_gat_best.pth")
    resume_path = os.path.join(cfg.model_save_dir, getattr(cfg, 'resume_name', 'ppo_gat_last.pth'))
    # The rolling checkpoint is preferred over the best model: it also carries the optimiser state,
    # the entropy coefficient and the episode counter, so an interrupted run continues where it
    # stopped instead of retraining from the first episode.
    start_episode = 0
    val_curve = []
    val_reference = None
    best_val = float('inf')
    stale_checks = 0
    converged = False
    rolling = trainer.load_rolling(resume_path)
    if rolling:
        start_episode = min(int(rolling.get('episode_counter', 0) or 0), cfg.n_episodes)
        val_curve = [tuple(entry) for entry in rolling.get('val_curve', []) or []]
        val_reference = list(rolling.get('val_reference', []) or []) or None
        best_val = float(rolling.get('best_val', float('inf')))
        converged = bool(rolling.get('converged', False))
        # The patience of the early stop survives a restart.  Without this rebuild the run of
        # this baseline was restarted four times in one day and each restart opened a fresh
        stale_checks = 0
        running_best = float('inf')
        for _episode, _score in val_curve:
            if _score < running_best * (1.0 - cfg.val_tol):
                running_best = _score
                stale_checks = 0
            else:
                stale_checks += 1
        print(f"Continuing from episode {start_episode}/{cfg.n_episodes} "
              f"(best validation {best_val:.6f})")
        if converged and start_episode >= cfg.n_episodes:
            print("The rolling checkpoint already holds a converged run; nothing to train.")
            return trainer
    else:
        # No rolling checkpoint: the best model of the previous run is the only state to continue.
        trainer._load_model()

    init_params = {
        'gat': gat.state_dict(),
        'policy': policy.state_dict(),
        'value': value.state_dict(),
    }
    if state_norm:
        init_params.update({
            'state_norm_mean': state_norm.mean.cpu(),
            'state_norm_var': state_norm.var.cpu(),
            'state_norm_count': state_norm.count,
        })

    param_queue = mp.Queue()
    data_queue = mp.Queue()
    stop_event = mp.Event()

    workers = []
    for i in range(cfg.num_workers):
        p = mp.Process(target=worker_process,
                       args=(i, instance_tuples, max_jobs, max_machines, max_proc_time,
                             max_due_date, cfg, param_queue, data_queue, stop_event))
        p.start()
        workers.append(p)

    # Send initial parameters
    for _ in range(cfg.num_workers):
        param_queue.put(init_params)

    # The curriculum of a resumed run starts at the stage its episode counter selects: the stages
    # are addressed by the episode count, and a run that skips this initialisation would keep the
    # workers on the first stage until an episode of a later stage is reached.
    curriculum_scale = max(1, int(getattr(cfg, 'curriculum_episodes', cfg.n_episodes)))
    progress = min(0.999, start_episode / curriculum_scale)
    current_allowed_sizes = (stage_for_progress(cfg.curriculum_stages, progress)
                             if cfg.curriculum_enabled else ['small', 'medium', 'large'])
    for _ in range(cfg.num_workers):
        param_queue.put({'sizes': current_allowed_sizes})

    total_episodes = start_episode
    total_steps = 0
    trajectories_buffer = []
    ep_rewards_buffer = []
    normalized_rewards = []
    # Episode number of every recorded reward.  The three curve lists below are sampled per record
    # (one entry per collected episode for the reward, one per update for KL and loss, padded to
    reward_episodes = []
    kl_episodes = []
    loss_episodes = []
    best_norm_reward = -float('inf')
    pbar = tqdm(total=cfg.n_episodes, initial=start_episode, desc="Training", unit="ep")
    # Reference of the validation objective.  It is the weighted objective of a deterministic
    # earliest-completion dispatch of the same instance (dispatch_reference.py) instead of the
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from dispatch_reference import reference_objectives
    val_reference = [weighted_objective(obj) for obj in reference_objectives(
        [inst.file_name for inst in validation_set], cfg.instance_dir)]
    # A curve written by the earlier frame holds a ratio to the first check of that run and is
    # not comparable with this one, so the run starts a fresh curve instead of mixing the two.
    if val_curve and abs(val_curve[0][1] - 1.0) < 1e-12:
        print("The restored validation curve is in the old frame and is reset.")
        val_curve = []
        best_val = float('inf')
        stale_checks = 0
    last_worker_alive_check = time.time()

    try:
        while total_episodes < cfg.n_episodes and not converged:
            # Periodically check worker liveness
            if time.time() - last_worker_alive_check > 10:
                alive = any(p.is_alive() for p in workers)
                if not alive:
                    print("All workers have died. Stopping training.")
                    break
                last_worker_alive_check = time.time()

            try:
                worker_trajectories = data_queue.get(timeout=10)
            except Empty:
                # Timeout, check workers again
                if not any(p.is_alive() for p in workers):
                    print("All workers have terminated, stopping training.")
                    break
                continue
            except (ConnectionResetError, OSError) as e:
                print(f"Queue error: {e}, checking workers...")
                if not any(p.is_alive() for p in workers):
                    print("All workers have terminated, stopping training.")
                    break
                continue

            for traj in worker_trajectories:
                total_episodes += 1
                total_steps += len(traj)
                trajectories_buffer.append(traj)
                ep_reward = sum(step['reward'] for step in traj)
                norm_reward = ep_reward / len(traj)
                normalized_rewards.append(norm_reward)
                reward_episodes.append(total_episodes)
                ep_rewards_buffer.append(ep_reward)

                # Curriculum learning progress update
                if cfg.curriculum_enabled:
                    progress = min(0.999, total_episodes / curriculum_scale)
                    new_sizes = None
                    for start, end, sizes in cfg.curriculum_stages:
                        if start <= progress < end:
                            if sizes != current_allowed_sizes:
                                new_sizes = sizes
                            break
                    if new_sizes is not None:
                        current_allowed_sizes = new_sizes
                        print(f"\nCurriculum step: now using {new_sizes} (progress {progress:.2f})")
                        curriculum_msg = {'sizes': new_sizes}
                        for _ in range(cfg.num_workers):
                            try:
                                param_queue.put(curriculum_msg, timeout=5)
                            except:
                                pass

                # The smoothed training return is the curve that the figure reports; it no longer
                # selects the weights, because two episodes of different curriculum stages are not
                # comparable.
                if len(normalized_rewards) >= cfg.stats_window:
                    smoothed = float(np.mean(normalized_rewards[-cfg.stats_window:]))
                    if smoothed > best_norm_reward:
                        best_norm_reward = smoothed

                # Reach training step threshold
                if total_steps >= cfg.train_step_size:
                    batch_size = len(trajectories_buffer)
                    stats = trainer.train_step(trajectories_buffer)
                    # The floor keeps the policy exploratory for the whole run: the decay alone
                    # would drive the coefficient to 1e-4 and make the policy deterministic long
                    # before the curriculum has finished.
                    trainer.entropy_coef = max(trainer.entropy_coef * cfg.entropy_decay,
                                               getattr(cfg, 'entropy_min', 0.0))

                    if stats:
                        # Repeat KL and loss batch_size times to align with episodes
                        kl_episodes.extend([stats['kl']] * batch_size)
                        loss_episodes.extend([stats['total_loss']] * batch_size)

                        pbar.set_postfix({
                            'NormR': f"{norm_reward:.4f}",
                            'KL': f"{stats['kl']:.4f}",
                            'Beta': f"{stats['beta']:.3f}",
                            'L(θ)': f"{stats['total_loss']:.3f}",
                            'Steps': len(traj)
                        })
                    else:
                        pbar.set_postfix({
                            'NormR': f"{norm_reward:.4f}",
                            'Steps': len(traj)
                        })

                    # Send new parameters to all workers
                    new_params = {
                        'gat': gat.state_dict(),
                        'policy': policy.state_dict(),
                        'value': value.state_dict(),
                    }
                    if state_norm:
                        new_params.update({
                            'state_norm_mean': state_norm.mean.cpu(),
                            'state_norm_var': state_norm.var.cpu(),
                            'state_norm_count': state_norm.count,
                        })
                    for _ in range(cfg.num_workers):
                        try:
                            param_queue.put(new_params, timeout=5)
                        except:
                            pass

                    trajectories_buffer = []
                    total_steps = 0
                    ep_rewards_buffer = []

                    gc.collect()
                else:
                    pbar.set_postfix({
                        'NormR': f"{norm_reward:.4f}",
                        'Steps': len(traj)
                    })

                pbar.update(1)
                if total_episodes % max(1, int(getattr(cfg, 'log_interval', 100))) == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Ep {total_episodes:4d} | "
                          f"NormR {norm_reward:8.4f} | Steps {len(traj):4d}")

                # Validation block.  It selects the reported weights and it stops the run as soon as
                # the average objective stops improving, i.e. the run is never trained past the point
                # where the policy keeps getting better.
                if total_episodes >= cfg.min_episodes and total_episodes % max(1, cfg.val_interval) == 0:
                    objectives = evaluate_validation(
                        gat, policy, state_norm, validation_set, max_jobs, max_machines,
                        max_proc_time, max_due_date, cfg, seed=int(getattr(cfg, 'val_seed', 0)))
                    scores = [weighted_objective(obj) for obj in objectives]
                    if val_reference is not None and len(val_reference) != len(scores):
                        # A validation set that was resized after the checkpoint was written
                        # would otherwise be scored against the reference of the old set, and
                        # a score of the new frame could never beat the best of the old one.
                        val_reference = None
                        best_val = float('inf')
                        stale_checks = 0
                    if val_reference is None:
                        # The first check defines the reference of every instance, so the score is a
                        # relative objective that does not depend on the instance size, which
                        # dominates the raw weighted value.
                        val_reference = [value if abs(value) > 1e-9 else 1.0 for value in scores]
                    val_score = float(np.mean([value / ref
                                               for value, ref in zip(scores, val_reference)]))
                    val_curve.append((total_episodes, val_score))
                    print(f"Validation {len(val_curve):3d} at episode {total_episodes}: "
                          f"relative objective {val_score:.6f} (best {best_val:.6f})")
                    if val_score < best_val * (1.0 - cfg.val_tol):
                        best_val = val_score
                        stale_checks = 0
                        # The weights the validation selected are the ones the comparison experiment
                        # loads, so the best model is written here and not on the episode return.
                        trainer.save_best(best_norm_reward)
                        # The best checkpoint carries the same bookkeeping as the rolling one.
                        # Its selection and its protection read the validation objective and the
                        trainer.save_rolling(model_path, total_episodes, val_curve, val_reference,
                                             best_val, False)
                        trainer.save_rolling(resume_path, total_episodes, val_curve, val_reference,
                                             best_val, False)
                    else:
                        stale_checks += 1
                        if stale_checks >= cfg.val_patience:
                            print(f"Early stop after {total_episodes} episodes: the validation "
                                  f"objective did not improve for {stale_checks} checks "
                                  f"(best={best_val:.6f}).")
                            converged = True

                # Rolling checkpoint: it bounds the work a restart has to repeat to
                # checkpoint_interval episodes.
                if total_episodes % max(1, int(getattr(cfg, 'checkpoint_interval', 25))) == 0:
                    trainer.save_rolling(resume_path, total_episodes, val_curve, val_reference,
                                         best_val, converged)

                if converged:
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping workers...")
    finally:
        stop_event.set()
        # Clear queues to allow workers to exit smoothly
        # Wait briefly for workers to respond to stop_event
        time.sleep(1)
        # Keep draining data_queue until all workers are done
        while any(p.is_alive() for p in workers):
            try:
                # Try to clear the queue to avoid workers blocking on put
                while not data_queue.empty():
                    data_queue.get_nowait()
            except:
                pass
            # Wait a bit
            time.sleep(1)

        # Wait for workers to finish, timeout 20 seconds
        for p in workers:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # Clean up queues (may be corrupted, attempt safe close)
        try:
            while not data_queue.empty():
                data_queue.get_nowait()
        except:
            pass
        try:
            while not param_queue.empty():
                param_queue.get_nowait()
        except:
            pass

        # Process the last untrained batch (use local buffer, not relying on queue)
        if trajectories_buffer:
            print("Training last batch...")
            batch_size = len(trajectories_buffer)
            stats = trainer.train_step(trajectories_buffer)
            if stats:
                # Ensure lengths match to avoid plotting errors later
                needed_kl = len(normalized_rewards) - len(kl_episodes)
                if needed_kl > 0:
                    kl_episodes.extend([stats['kl']] * needed_kl)
                else:
                    kl_episodes.extend([stats['kl']] * batch_size)
                needed_loss = len(normalized_rewards) - len(loss_episodes)
                if needed_loss > 0:
                    loss_episodes.extend([stats['total_loss']] * needed_loss)
                else:
                    loss_episodes.extend([stats['total_loss']] * batch_size)

        # If still length mismatch, pad or trim
        if len(kl_episodes) < len(normalized_rewards):
            kl_episodes.extend([kl_episodes[-1] if kl_episodes else 0] * (len(normalized_rewards) - len(kl_episodes)))
        elif len(kl_episodes) > len(normalized_rewards):
            kl_episodes = kl_episodes[:len(normalized_rewards)]
        if len(loss_episodes) < len(normalized_rewards):
            loss_episodes.extend([loss_episodes[-1] if loss_episodes else 0] * (len(normalized_rewards) - len(loss_episodes)))
        elif len(loss_episodes) > len(normalized_rewards):
            loss_episodes = loss_episodes[:len(normalized_rewards)]

    pbar.close()

    # The finished run is checkpointed as well, so a later restart of this script finds every
    # episode already trained and only refreshes the curves.
    trainer.save_rolling(resume_path, total_episodes, val_curve, val_reference, best_val, converged)
    print("Training completed." if not converged else "Training stopped early (converged).")
    print(f"Best validation objective: {best_val:.6f}")
    print("Training Finished. Best Normalized Reward:", best_norm_reward)

    # Validation curve: the relative objective of the fixed validation set.  It is the quantity
    # that selects the reported weights and that stops the run, so it is plotted next to the
    # training curve.
    if val_curve:
        plt.figure(figsize=(7, 4))
        plt.plot([step for step, _ in val_curve], [score for _, score in val_curve],
                 marker='o', color='green', label='Validation objective')
        plt.title("Validation Objective Curve")
        plt.xlabel("Episode")
        plt.ylabel("Relative objective")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "validation_curve.png"))

    # Plot curves
    # The three curves of this figure come from the lists that this run filled.  A run that finds an
    if not (normalized_rewards or kl_episodes or loss_episodes):
        print("No curve data in this run (nothing was trained); keeping the existing figures.")
        return
    plt.figure(figsize=(15, 5))

    def moving_average(data, window_size=20):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def set_ylim(*series):
        """Y range of a panel that covers every series drawn in it.

        The range was taken from the raw series alone, so a transient spike of the raw data set the
        scale of the panel and compressed the moving average, which is an order of magnitude
        smaller after the spike, onto the axis.  A series that spans more than two orders of
        magnitude is drawn on a log axis, the only scale on which both the transient and the
        average are legible.
        """
        values = [float(value) for one in series for value in one if np.isfinite(value)]
        if not values:
            return
        low, high = min(values), max(values)
        if low > 0 and high / low > 100.0:
            plt.yscale('log')
            plt.ylim(low * 0.5, high * 2.0)
        else:
            margin = 0.05 * (high - low) if high > low else 0.1
            plt.ylim(low - margin, high + margin)

    # Reward curve
    plt.subplot(1, 3, 1)
    plt.plot(reward_episodes, normalized_rewards, alpha=0.3, color='blue', label='Raw')
    if len(normalized_rewards) >= 20:
        ma_rewards = moving_average(normalized_rewards, 20)
        plt.plot(reward_episodes[19:], ma_rewards, color='blue', linewidth=2, label='Moving Avg (20)')
    plt.title("Normalized Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Norm Reward")
    plt.grid(True)
    plt.legend()
    set_ylim(normalized_rewards, ma_rewards if len(normalized_rewards) >= 20 else [])
    # Guarded: the helper above switches the axis to a logarithmic scale when the values span more
    # than two orders of magnitude, and ticklabel_format then raises
    try:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    except Exception:
        pass

    # KL curve
    plt.subplot(1, 3, 2)
    plt.plot(reward_episodes, kl_episodes, alpha=0.3, color='orange', label='Raw')
    if len(kl_episodes) >= 20:
        ma_kl = moving_average(kl_episodes, 20)
        plt.plot(reward_episodes[19:], ma_kl, color='orange', linewidth=2, label='Moving Avg (20)')
    plt.title("KL Divergence Curve")
    plt.xlabel("Episode")
    plt.ylabel("KL")
    plt.grid(True)
    plt.legend()
    set_ylim(kl_episodes, ma_kl if len(kl_episodes) >= 20 else [])
    try:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    except Exception:
        pass

    # Loss curve
    plt.subplot(1, 3, 3)
    plt.plot(reward_episodes, loss_episodes, alpha=0.3, color='red', label='Raw')
    if len(loss_episodes) >= 20:
        ma_loss = moving_average(loss_episodes, 20)
        plt.plot(reward_episodes[19:], ma_loss, color='red', linewidth=2, label='Moving Avg (20)')
    plt.title("Total Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("L(θ)")
    plt.grid(True)
    plt.legend()
    set_ylim(loss_episodes, ma_loss if len(loss_episodes) >= 20 else [])
    try:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ppo_gat_curves.png"))
    plt.show()

def main():
    """Training entry point: runs Config.n_episodes episodes by default, --episodes overrides it."""
    parser = argparse.ArgumentParser(description="Train the homogeneous-graph GAT baseline.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override Config.n_episodes (used to fit the time budget)")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes > 0:
        cfg.n_episodes = args.episodes
        print(f"n_episodes overridden to {cfg.n_episodes}")
    if not os.path.exists(cfg.instance_dir):
        print(f"Warning: Instance directory {cfg.instance_dir} not found. Using current directory.")
        cfg.instance_dir = "."
    mp.set_start_method('spawn', force=True)
    train(cfg)


if __name__ == "__main__":
    main()

