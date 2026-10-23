"""
DQN+MLP Solution Implementation
Uses the most traditional DQN version (no variants) combined with a multi-layer perceptron
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import matplotlib
# Without a display session (background run, stdout redirected to a file, headless server)
# the Qt backend cannot be initialised and the process would crash; fall back to the file-only
# Agg backend and keep the default backend for interactive runs.
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import os
import glob
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
from torch.cuda.amp import autocast, GradScaler
import time
import numba
from numba import njit

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

FIGURE_DIR = os.path.join(_PROJECT_ROOT, "Figure_And_File", "DQN")
os.makedirs(FIGURE_DIR, exist_ok=True)


# 1. Configuration parameters

class Config:
    # Training parameters
    # Step size of the value update.  Three generations of this baseline (all with different weights,
    lr = 1e-4
    # The decay per epoch.  At 0.999 the learning rate had fallen from 5e-5 to 2.5e-6 by the end of
    # the last run (a factor of twenty) while the validation objective was still moving, so the
    lr_decay = 0.999
    min_lr = 1e-6
    # Undiscounted return, as in every other method that learns in this project (the mean-aggregation
    # baseline, the two attention baselines and the proposed method all carry one, and their configs
    gamma = 0.99
    # One step further along a measured direction.  The value of this baseline is decided by a
    # single terminal term that carries the weighted objective, while the per-step reward is only a
    gamma = 0.99
    # Dense credit assignment of the scored objective, see the note over the potential of the
    # environment.  The terminal term carries the weighted objective and the gamma above damps it to
    reward_shaping = False
    # Window of the n-step target.  The reward of one step of this environment is a local proxy and
    # the scored objective enters only as a terminal term, so a one-step target hands that term to a
    n_step = 10
    # Where this baseline stands, measured - and why the margin of the qualification gate has to be
    # read with care here.  The untrained level of a method is a property of its architecture *and*
    batch_size = 1024
    # Depth of the replay buffer.  One epoch adds about 4800 transitions (32 environments of some
    # 150 operations), so the previous capacity of 20000 held four epochs of experience and the
    buffer_capacity = 20000
    target_update_freq = 500

    epsilon_start = 1.0
    epsilon_end = 0.1
    # Slower exploration decay.  At 0.999 the policy was fully greedy by episode 1200 of a 6000
    # episode run, i.e. it stopped exploring before the curriculum had reached the medium and large
    # instances, and every epoch after that only replayed the samples of the first stage.
    epsilon_decay = 0.9995

    # Width and depth of the value network.  A wave with 1024 units over three hidden layers ran at
    # 4.8 s per episode on the cpu while the rest of the family trains at two to nine, and its
    hidden_dim = 512
    num_hidden_layers = 2

    objective_weights = [0.4, 0.3, 0.3]

    max_steps = 1000
    epochs = 30000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_jobs = None
    max_machines = None
    global_max_proc_time = None
    global_max_due_date = None

    model_dir = os.path.join(_PROJECT_ROOT, "model", "dqn")
    model_name = "dqn_model.pth"
    model_path = os.path.join(model_dir, model_name)

    instance_dir = os.path.join(_PROJECT_ROOT, "DataSet", "mo_fjsp_instances")

    use_state_norm = True
    # Reward scale and clip of the shared comparison.  This baseline used 2.0 and 1.0 while every other
    # trainer of the wave - and the reward of the paper - uses 5.0 and 2.0.  The weaker signal is what
    reward_scaling = 5.0
    # Not scaled for this baseline.  The five is a value of the PPO family, and there it is
    # harmless because the advantages are normalised per batch, so the scale of the reward cancels
    reward_scaling = 1.0
    # Scale of the reward inside the temporal-difference target.  The environment normalises its
    # is rescaled, so the comparison stays on one reward and one objective.  It is one here, and
    target_scale = 1.0
    reward_clip = 2.0

    sampling_temperature = 1.0
    precollect_episodes = 1
    # The gradient clip of the previous waves.  A value of one clipped every update of this network
    # to a fraction of its own gradient: the wave trained with it improved by 4 % over the previous
    grad_norm = 10.0

    # Gradient steps per training epoch.  The last wave of this baseline ran 256 steps of a batch
    # of 1024 against a buffer of 20000 samples, i.e. thirteen passes over the whole buffer in one
    gradient_steps = 256
    log_interval = 200
    use_amp = True

    num_envs = 32
    steps_per_update = 64

    # Model selection and early stopping.  One episode is a noisy sample of the policy (its
    # instance scale and its operation count change from episode to episode), so the reported
    rollout_name = "dqn_last.pth"
    checkpoint_interval = 250
    val_interval = 250
    # Four instances per size reduce the sampling noise of a check: with two instances the
    # reported objective swung by ten percent between two checks of one run.
    val_instances_per_size = 8
    val_seed = 0
    # A stale check is one validation of 250 episodes, so a patience of six allows at most
    # 1500 episodes past the peak before the run is closed.
    val_patience = 6
    val_tol = 0.01
    # No early stop before this budget.  The window also has to open before the peak of a run:
    # the best checkpoint of the previous generation was written at episode 1937 for the DAN
    # baseline, which a window of 2500 could never observe.
    min_epochs = 500
    # The curriculum advances on a fixed episode scale instead of on the epoch budget.
    curriculum_episodes = 4000

    curriculum_enabled = True                     # Enable curriculum learning
    # Curriculum stages: each stage corresponds to an episode range [start_ratio, end_ratio) and a list of allowed sizes
    curriculum_stages = [
        (0.0, 0.3, ['small']),                    # First 30% episodes only use small
        (0.3, 0.6, ['small', 'medium']),          # Next 30% use small + medium
        (0.6, 1.0, ['small', 'medium', 'large'])   # Last 40% use all sizes
    ]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# 2. File reading module

@dataclass
class InstanceData:
    num_jobs: int
    num_machines: int
    job_op_ranges: np.ndarray          # shape (num_jobs, 2)
    op_machines: np.ndarray             # shape (total_ops, max_alt)
    op_times: np.ndarray                # shape (total_ops, max_alt)
    due_dates: np.ndarray               # shape (num_jobs,)
    max_proc_time: float
    max_due_date: float
    total_ops: int

def parse_size_from_filename(filename: str) -> str:
    match = re.search(r'(small|medium|large)', filename)
    return match.group(1) if match else "unknown"

def read_fjsp_instance_structured(file_path: str):
    """Read instance and convert to structured arrays"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        exit(1)
    if len(lines) < 2:
        raise ValueError("File format error: insufficient lines")
    num_jobs, num_machines = map(int, lines[0].split())
    job_lines = lines[1:1 + num_jobs]
    if len(job_lines) != num_jobs:
        raise ValueError(f"File format error: expected {num_jobs} job lines, got {len(job_lines)}")
    due_date_line = lines[1 + num_jobs]
    due_dates = np.array(list(map(float, due_date_line.split())), dtype=np.float32)

    op_machines_list = []
    op_times_list = []
    job_op_ranges = []
    op_idx = 0
    max_alt = 0
    max_proc_time = 0
    total_ops = 0

    for job_idx in range(num_jobs):
        nums = list(map(int, job_lines[job_idx].split()))
        num_ops = nums[0]
        start_idx = op_idx
        idx = 1
        job_ops_machines = []
        job_ops_times = []
        for _ in range(num_ops):
            k = nums[idx]
            idx += 1
            machines = []
            times = []
            for _ in range(k):
                machine_id = nums[idx] - 1
                proc_time = float(nums[idx + 1])
                machines.append(machine_id)
                times.append(proc_time)
                max_proc_time = max(max_proc_time, proc_time)
                idx += 2
            job_ops_machines.append(machines)
            job_ops_times.append(times)
            max_alt = max(max_alt, k)
        for machines, times in zip(job_ops_machines, job_ops_times):
            op_machines_list.append(machines)
            op_times_list.append(times)
        op_idx += num_ops
        job_op_ranges.append((start_idx, op_idx))
        total_ops += num_ops

    op_machines_arr = np.full((total_ops, max_alt), -1, dtype=np.int32)
    op_times_arr = np.zeros((total_ops, max_alt), dtype=np.float32)
    for i, (machines, times) in enumerate(zip(op_machines_list, op_times_list)):
        for j, (m, t) in enumerate(zip(machines, times)):
            op_machines_arr[i, j] = m
            op_times_arr[i, j] = t

    job_op_ranges = np.array(job_op_ranges, dtype=np.int32)
    max_due_date = due_dates.max() if due_dates.size > 0 else 1.0

    return InstanceData(
        num_jobs=num_jobs,
        num_machines=num_machines,
        job_op_ranges=job_op_ranges,
        op_machines=op_machines_arr,
        op_times=op_times_arr,
        due_dates=due_dates,
        max_proc_time=float(max_proc_time),
        max_due_date=float(max_due_date),
        total_ops=total_ops
    )

def load_all_instances_structured(instance_dir: str, file_pattern: str = "mo_fjsp_*_train.txt"):
    file_paths = glob.glob(os.path.join(instance_dir, file_pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files matching {file_pattern} found in directory {instance_dir}")
    instance_list = []
    max_jobs = 0
    max_machines = 0
    max_proc_time = 0
    max_due_date = 0
    for fp in file_paths:
        data = read_fjsp_instance_structured(fp)
        file_name = os.path.basename(fp)
        size = parse_size_from_filename(file_name)
        instance_list.append((data, file_name, size))
        max_jobs = max(max_jobs, data.num_jobs)
        max_machines = max(max_machines, data.num_machines)
        max_proc_time = max(max_proc_time, data.max_proc_time)
        max_due_date = max(max_due_date, data.max_due_date)
    return instance_list, max_jobs, max_machines, max_proc_time, max_due_date



# 3. Instance wrapper class

class JobShopInstance:
    def __init__(self, data: InstanceData, max_jobs, max_machines, file_name, size):
        self.data = data
        self.file_name = file_name
        self.size = size                     # 'small', 'medium', 'large'
        self.num_jobs = data.num_jobs
        self.num_machines = data.num_machines
        self.max_jobs = max_jobs
        self.max_machines = max_machines
        self.total_ops = data.total_ops
        self.max_proc_time = data.max_proc_time
        self.max_due_date = data.max_due_date
        self.due_dates = data.due_dates
        self.job_op_ranges = data.job_op_ranges
        self.op_machines = data.op_machines
        self.op_times = data.op_times



# 4. Instance sampler

class CurriculumSampler:
    """Selects allowed sizes according to curriculum stage and samples uniformly from them"""
    def __init__(self, instances: List[JobShopInstance], cfg: Config):
        self.instances = instances
        self.cfg = cfg
        # Classify by size
        self.instances_by_size = {'small': [], 'medium': [], 'large': []}
        for inst in instances:
            self.instances_by_size[inst.size].append(inst)
        # Currently allowed sizes
        self.allowed_sizes = ['small']   # initially only small
        self.current_stage = 0

    def update_stage(self, episode_progress: float):
        """Select the sizes of the current curriculum stage.

        The progress is clamped into the range of the stages and the last stage is used when no
        stage matches, so an unmatched call can never leave the sampler at the first stage.
        """
        if not self.cfg.curriculum_enabled:
            self.allowed_sizes = ['small', 'medium', 'large']
            return
        stages = self.cfg.curriculum_stages
        # The caller clips its progress to 1.0 and the last stage of the table ends at that value,
        # so a call with exactly 1.0 matches no stage at all.  A run that resumes past the end of
        progress = min(max(episode_progress, 0.0), 1.0)
        for start, end, sizes in stages:
            if start <= progress < end:
                self.allowed_sizes = sizes
                return
        self.allowed_sizes = list(stages[-1][2])

    def sample(self) -> JobShopInstance:
        # Uniformly sample from all instances in allowed sizes
        allowed_instances = []
        for sz in self.allowed_sizes:
            allowed_instances.extend(self.instances_by_size[sz])
        if not allowed_instances:
            # Fallback: sample from all instances if allowed list is empty (possible misconfiguration)
            allowed_instances = self.instances
        return random.choice(allowed_instances)


# 5. State normalization tool

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4, device='cpu'):
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
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)


# 6. Environment class (with reward adaptive scaling)

@njit(cache=True)
def build_state(machine_avail_time, machine_load, job_next_op_idx, job_avail_time,
                num_jobs_actual, num_machines_actual, max_jobs, max_machines,
                due_dates, job_op_ranges, op_machines, op_times,
                max_proc_time, max_due_date, total_ops):
    # ... identical to before ...
    current_global_time = np.max(machine_avail_time[:num_machines_actual])
    max_load_est = max(np.max(machine_load[:num_machines_actual]), 100.0)
    feat_m_load = machine_load.copy()
    if max_load_est > 0:
        feat_m_load[:num_machines_actual] /= max_load_est
    max_time_est = max(np.max(machine_avail_time[:num_machines_actual]), 100.0)
    feat_m_time = machine_avail_time.copy()
    if max_time_est > 0:
        feat_m_time[:num_machines_actual] /= max_time_est

    feat_j_progress = np.zeros(max_jobs, dtype=np.float32)
    feat_j_urgency = np.zeros(max_jobs, dtype=np.float32)
    feat_j_proc = np.zeros(max_jobs, dtype=np.float32)

    for j_id in range(num_jobs_actual):
        start_op, end_op = job_op_ranges[j_id]
        total_ops_job = end_op - start_op
        curr_op_idx = job_next_op_idx[j_id]
        feat_j_progress[j_id] = curr_op_idx / total_ops_job if total_ops_job > 0 else 0.0
        due = due_dates[j_id]
        feat_j_urgency[j_id] = current_global_time / due if due > 0 else 1.0
        if curr_op_idx < total_ops_job:
            global_op_idx = start_op + curr_op_idx
            sum_t = 0.0
            count = 0
            for k in range(op_times.shape[1]):
                t = op_times[global_op_idx, k]
                if t > 0:
                    sum_t += t
                    count += 1
            avg_proc = sum_t / count if count > 0 else 0.0
            feat_j_proc[j_id] = avg_proc / max_proc_time if max_proc_time > 0 else 0.0
        else:
            feat_j_proc[j_id] = 0.0

    return np.concatenate((feat_m_load, feat_m_time, feat_j_progress, feat_j_urgency,
                           feat_j_proc))

class MOFJSP_Env:
    def __init__(self, cfg, instance: JobShopInstance):
        self.cfg = cfg
        self.inst = instance
        self.num_jobs_actual = instance.num_jobs
        self.num_machines_actual = instance.num_machines
        self.max_jobs = cfg.max_jobs
        self.max_machines = cfg.max_machines
        self.max_proc_time = instance.max_proc_time
        self.max_due_date = instance.max_due_date
        self.due_dates = instance.due_dates
        self.job_op_ranges = instance.job_op_ranges
        self.op_machines = instance.op_machines
        self.op_times = instance.op_times
        self.total_ops = instance.total_ops
        self.reset()

    def reset(self):
        self.machine_avail_time = np.zeros(self.max_machines, dtype=np.float32)
        self.machine_load = np.zeros(self.max_machines, dtype=np.float32)
        self.job_next_op_idx = np.zeros(self.max_jobs, dtype=np.int32)
        self.job_avail_time = np.zeros(self.max_jobs, dtype=np.float32)
        self.finished_ops_count = 0
        self.invalid_actions = 0
        return self._get_state()

    def _get_state(self):
        return build_state(
            self.machine_avail_time, self.machine_load, self.job_next_op_idx, self.job_avail_time,
            self.num_jobs_actual, self.num_machines_actual, self.max_jobs, self.max_machines,
            self.due_dates, self.job_op_ranges, self.op_machines, self.op_times,
            self.max_proc_time, self.max_due_date, self.total_ops
        )

    def get_action_mask(self):
        mask = np.zeros(self.max_jobs * self.max_machines, dtype=np.float32)
        n_j = self.num_jobs_actual
        job_next = self.job_next_op_idx
        op_ranges = self.job_op_ranges
        op_machines = self.op_machines
        max_m = self.max_machines
        for j_id in range(n_j):
            curr_op_idx = job_next[j_id]
            if curr_op_idx >= op_ranges[j_id, 1] - op_ranges[j_id, 0]:
                continue
            global_op_idx = op_ranges[j_id, 0] + curr_op_idx
            for k in range(op_machines.shape[1]):
                m_id = op_machines[global_op_idx, k]
                if m_id == -1:
                    break
                action_idx = j_id * max_m + m_id
                mask[action_idx] = 1.0
        return mask

    def _partial_penalty(self):
        """Scaled and clipped penalty of the schedule built so far.

        The three terms are the ones of the terminal reward of this environment on the partial
        schedule, and the result is scaled and clipped exactly like that terminal reward, so the
        potential of the finished schedule is the terminal penalty itself.  Unfinished jobs have a
        completion time of zero and contribute no tardiness, so the sum over every job is the sum
        the terminal block computes.
        """
        inst = self.inst
        n_m = inst.num_machines
        w = self.cfg.objective_weights

        load = self.machine_load[:n_m]
        avg_load = float(np.mean(load)) if n_m > 0 else 0.0
        c_max = float(self.machine_avail_time[:n_m].max()) if n_m > 0 else 0.0
        lb = float(np.sqrt(np.mean((load - avg_load) ** 2))) if n_m > 0 else 0.0
        t_tardy = sum(max(0.0, float(self.job_avail_time[j]) - inst.due_dates[j])
                      for j in range(inst.num_jobs))

        penalty = (w[0] * c_max / (inst.max_proc_time * self.total_ops + 1.0)
                   + w[1] * lb / (avg_load + 1e-8)
                   + w[2] * t_tardy / (inst.max_due_date * self.total_ops + 1.0))
        penalty /= self.cfg.reward_scaling
        return float(np.clip(penalty, 0.0, self.cfg.reward_clip))

    def step(self, action_idx):
        j_id = action_idx // self.max_machines
        m_id = action_idx % self.max_machines
        inst = self.inst
        curr_op_idx = self.job_next_op_idx[j_id]
        global_op_idx = inst.job_op_ranges[j_id, 0] + curr_op_idx
        proc_time = None
        for k in range(inst.op_machines.shape[1]):
            if inst.op_machines[global_op_idx, k] == m_id:
                proc_time = inst.op_times[global_op_idx, k]
                break
        if proc_time is None:
            # Handle invalid action: it does not advance the environment, so a policy that keeps
            # proposing it would spin until the step limit and the episode would then be reported as
            print(f"Warning: invalid action {action_idx} (j={j_id}, m={m_id})")
            self.invalid_actions += 1
            if self.invalid_actions >= 5:
                return self._get_state(), -1.0, True, {}
            return self._get_state(), -1.0, False, {}

        # Penalty of the schedule built so far, read before this decision is applied: the reward of
        # the step gains the decrease of this potential, see the note over the terminal block.
        penalty_before = self._partial_penalty()

        start_time = max(self.job_avail_time[j_id], self.machine_avail_time[m_id])
        end_time = start_time + proc_time
        self.job_avail_time[j_id] = end_time
        self.machine_avail_time[m_id] = end_time
        self.machine_load[m_id] += proc_time
        self.job_next_op_idx[j_id] += 1
        self.finished_ops_count += 1

        w = self.cfg.objective_weights
        L_max_est = max(self.machine_load[:inst.num_machines].sum(), 1.0)
        due_date = inst.due_dates[j_id]
        # Operation urgency: zero while the operation finishes before its due date and growing with
        # the lateness once the due date is passed, which is the term the shared environment of the
        n_i = inst.job_op_ranges[j_id, 1] - inst.job_op_ranges[j_id, 0]
        u_ij = max(0, end_time - due_date) / (inst.max_due_date * n_i + 1e-5)

        # Base reward
        reward = - (w[0] * (proc_time / inst.max_proc_time) +
                   w[1] * (self.machine_load[m_id] / L_max_est) +
                   w[2] * u_ij)
        # Normalise the reward by the number of operations of the instance: every operation is
        # scheduled exactly once, so each step contributes to a mean instead of to a sum that
        scale_factor = max(1.0, float(inst.total_ops))
        reward /= scale_factor
        reward /= self.cfg.reward_scaling
        done = (self.finished_ops_count == self.total_ops)
        info = {}
        if done:
            c_max = max(self.machine_avail_time[:inst.num_machines])
            avg_load = np.mean(self.machine_load[:inst.num_machines])
            lb = np.sqrt(np.mean((self.machine_load[:inst.num_machines] - avg_load) ** 2))
            t_tardy = sum(max(0, self.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))
            # The three objectives of the finished schedule are normalised by the size of the
            # instance, i.e. by the same operation count as the local reward.  The original
            global_penalty = (w[0] * c_max / (inst.max_proc_time * self.total_ops + 1.0)
                              + w[1] * lb / (avg_load + 1e-8)
                              + w[2] * t_tardy / (inst.max_due_date * self.total_ops + 1.0))
            global_penalty /= self.cfg.reward_scaling
            info['final_metrics'] = {'Cmax': c_max, 'LB': lb, 'Tardy': t_tardy}

        if self.cfg.reward_shaping:
            # r_t gains P(s_t) - P(s_t+1).  The potential of the finished schedule is the terminal
            # penalty this environment used to subtract in one lump, so the sum of the steps of an
            reward += penalty_before - self._partial_penalty()
        elif done:
            reward -= global_penalty
        reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)
        next_state = self._get_state()
        if done:
            next_state = np.zeros_like(next_state)
        return next_state, float(reward), done, info


# 7. DQN Network

# --------------------------------------------------------------------------- #
# Dispatching rules of the value based baseline.
RULE_NAMES = ("SPT", "LPT", "MWKR", "LWKR", "MOPNR", "EDD", "LEAST_LOADED",
              "MOST_LOADED", "EARLIEST_MACHINE")
N_RULES = len(RULE_NAMES)


def rule_cost_table(env):
    """Cost of every legal action under every rule, as (legal actions, cost table).

    One row per (job, machine) pair the mask allows and one column per rule, holding the quantities the
    classical rules rank: the processing time of the pair, the work and the number of operations the job
    still needs, its due date, the load of the machine and the moment that machine becomes free.
    """
    mask = env.get_action_mask()
    legal = np.nonzero(mask)[0]
    cost = np.zeros((len(legal), N_RULES), dtype=np.float64)
    if len(legal) == 0:
        return legal, cost
    inst = env.inst
    max_m = env.max_machines
    for row, action in enumerate(legal):
        job = action // max_m
        machine = action % max_m
        op_index = inst.job_op_ranges[job, 0] + env.job_next_op_idx[job]
        proc = float(inst.max_proc_time)
        for k in range(inst.op_machines.shape[1]):
            if inst.op_machines[op_index, k] == machine:
                proc = float(inst.op_times[op_index, k])
                break
        ops_left = int(inst.job_op_ranges[job, 1] - inst.job_op_ranges[job, 0]
                       - env.job_next_op_idx[job])
        work_left = 0.0
        for offset in range(env.job_next_op_idx[job],
                            inst.job_op_ranges[job, 1] - inst.job_op_ranges[job, 0]):
            global_op = inst.job_op_ranges[job, 0] + offset
            best = float(inst.max_proc_time)
            for k in range(inst.op_machines.shape[1]):
                if inst.op_machines[global_op, k] == -1:
                    break
                best = min(best, float(inst.op_times[global_op, k]))
            work_left += best
        cost[row, 0] = proc
        cost[row, 1] = -proc
        cost[row, 2] = -work_left
        cost[row, 3] = work_left
        cost[row, 4] = -ops_left
        cost[row, 5] = float(inst.due_dates[job])
        cost[row, 6] = float(env.machine_load[machine])
        cost[row, 7] = -float(env.machine_load[machine])
        cost[row, 8] = float(env.machine_avail_time[machine])
    return legal, cost


def rule_actions(env):
    """The action every rule picks among the legal ones, in the order of RULE_NAMES."""
    legal, cost = rule_cost_table(env)
    if len(legal) == 0:
        return np.zeros(0, dtype=np.int64)
    return legal[np.argmin(cost, axis=0)].astype(np.int64)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers=3):
        super().__init__()
        layers = []
        input_dim = state_dim
        for i in range(num_layers):
            linear = nn.Linear(input_dim, hidden_dim)
            orthogonal_init(linear, gain=nn.init.calculate_gain('relu'))
            layers.append(linear)
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        output = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(output, gain=1.0)
        layers.append(output)
        self.net = nn.Sequential(*layers)
    def forward(self, state):
        return self.net(state)


# 8. GPU ring buffer

class NStepAccumulator:
    """Write n-step transitions into the replay buffer of the agent.

    One transition of one environment enters the window of that environment.  As soon as the window
    holds ``n_step`` transitions, the oldest of them is written with the discounted sum of the
    rewards of the window and with the state that follows the newest one as its bootstrap; when an
    episode ends, every remaining entry is written with its return to the end of the episode and
    without a bootstrap, which is how the standard n-step replay handles a window that crosses the
    end of an episode.
    """

    def __init__(self, n_step, gamma, num_envs):
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.window = [[] for _ in range(int(num_envs))]
        self.written = 0

    def push(self, memory, index, state, action, reward, next_state, done, mask, next_mask):
        """Feed one transition of one environment and write what the window completes.

        Only the rewards and the masks of the pending decisions are kept, so the window costs a few
        hundred bytes per environment of this run.
        """
        window = self.window[index]
        window.append((state, action, reward, mask, next_state, next_mask))
        if done:
            # The episode ended inside the window: every entry that is left ends inside it as well,
            # so its return is the discounted sum of the rewards that follow it and no bootstrap is
            # left to take.
            while window:
                total = window[0][2].new_zeros(())
                for step, entry in enumerate(window):
                    total = total + (self.gamma ** step) * entry[2]
                state_j, action_j, _r, mask_j, _ns, _nm = window.pop(0)
                memory.push(state_j, action_j, total, next_state, done, mask_j, next_mask)
                self.written += 1
            return
        if len(window) < self.n_step:
            return
        total = window[0][2].new_zeros(())
        for step, entry in enumerate(window[:self.n_step]):
            total = total + (self.gamma ** step) * entry[2]
        state_j, action_j, _r, mask_j, _ns, _nm = window.pop(0)
        # The bootstrap is the state that follows the newest decision of the window, which is the
        # transition that has just arrived; a transition that ends an episode is handled above, so
        # the episode cannot have ended in between.
        memory.push(state_j, action_j, total, next_state, done, mask_j, next_mask)
        self.written += 1


class GPURingBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        self.state_buffer = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.action_buffer = torch.zeros(capacity, dtype=torch.long, device=device)
        self.reward_buffer = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_state_buffer = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.done_buffer = torch.zeros(capacity, dtype=torch.float32, device=device)
        # The two action masks are the largest tensors of the buffer and they hold 0/1 only, so
        # they are stored as bytes: the depth of the buffer is tripled below without increasing the
        # memory of the replay, and only the next mask is read (to mask the bootstrapped maximum).
        self.mask_buffer = torch.zeros((capacity, action_dim), dtype=torch.uint8, device=device)
        self.next_mask_buffer = torch.zeros((capacity, action_dim), dtype=torch.uint8, device=device)
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.state_buffer[self.pos] = state
        self.action_buffer[self.pos] = action
        self.reward_buffer[self.pos] = reward
        self.next_state_buffer[self.pos] = next_state
        self.done_buffer[self.pos] = done
        self.mask_buffer[self.pos] = torch.as_tensor(np.asarray(mask)).to(torch.uint8)
        self.next_mask_buffer[self.pos] = torch.as_tensor(np.asarray(next_mask)).to(torch.uint8)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, num_samples):
        indices = torch.randint(0, self.size, (num_samples,), device=self.device)
        return (self.state_buffer[indices],
                self.action_buffer[indices],
                self.reward_buffer[indices],
                self.next_state_buffer[indices],
                self.done_buffer[indices],
                self.mask_buffer[indices],
                self.next_mask_buffer[indices])
    def __len__(self):
        return self.size


# 9. DQN Agent

class DQN_Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dim = 2 * cfg.max_machines + 3 * cfg.max_jobs
        # The action is the dispatching rule, not the (operation, machine) pair: see rule_actions().
        self.action_dim = N_RULES
        self.policy_net = QNetwork(self.state_dim, self.action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(cfg.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(cfg.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = GPURingBuffer(cfg.buffer_capacity, self.state_dim, self.action_dim, cfg.device)
        self.epsilon = cfg.epsilon_start
        self.steps_done = 0
        self.recent_q_values = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)
        self.model_path = cfg.model_path
        # Rolling checkpoint of an interrupted run.  The constructor does not restore it: the
        # agent is restored explicitly by train(), so a run that is started on purpose really
        # starts at the first episode instead of continuing silently.
        self.rollout_path = os.path.join(cfg.model_dir, cfg.rollout_name)
        self.episode_counter = 0
        # Validation bookkeeping: the validation curve, the reference objective of every
        # validation instance and the best value seen so far survive an interruption.
        self.val_curve = []
        self.val_reference = []
        self.best_val = float('inf')
        self.converged = False
        os.makedirs(cfg.model_dir, exist_ok=True)
        if cfg.use_amp and cfg.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _save_checkpoint(self, path, extra=None):
        """Write the weights together with everything a restart needs.

        The replay buffer is not stored: it is refilled within the first few hundred episodes of
        the restarted run, while its contents would dominate the size of the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_path = path + ".tmp"
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_counter': self.episode_counter,
            'val_curve': [tuple(entry) for entry in self.val_curve],
            'val_reference': list(self.val_reference),
            'best_val': self.best_val,
            'converged': self.converged,
            # Statistics of the state normaliser.  The normaliser is created inside the training
            # loop and not on this agent, so the loop attaches them to the agent once and every
            **(extra or getattr(self, 'norm_extra', None) or {}),
        }, _atomic_path)
        os.replace(_atomic_path, path)

    def _load_checkpoint(self, path):
        """Restore a rolling checkpoint; returns False when the file does not exist."""
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found.")
            return False
        allow_foreign_numpy_checkpoint()
        checkpoint = torch.load(path, map_location=self.cfg.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError:
            pass
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episode_counter = int(checkpoint.get('episode_counter', 0) or 0)
        self.val_curve = [tuple(entry) for entry in (checkpoint.get('val_curve', []) or [])]
        self.val_reference = list(checkpoint.get('val_reference', []) or [])
        self.best_val = float(checkpoint.get('best_val', float('inf')))
        self.converged = bool(checkpoint.get('converged', False))
        # Statistics of the state normaliser of this checkpoint, when it carries them.  The training
        # loop restores them into its normaliser and the evaluation normalises with them as well.
        self.checkpoint_norm = {name: checkpoint[name] for name in
                               ('norm_mean', 'norm_var', 'norm_count') if name in checkpoint}
        print(f"Loaded checkpoint {path} at episode {self.episode_counter}, "
              f"epsilon={self.epsilon:.4f}")
        return True

    def _save_model(self, path=None):
        self._save_checkpoint(path or self.model_path)

    def _load_model(self, path=None):
        return self._load_checkpoint(path or self.model_path)

    def select_action_batch(self, states, masks, epsilon):
        """states, masks are already on GPU"""
        batch_size = states.shape[0]
        explore_mask = torch.rand(batch_size, device=self.cfg.device) < epsilon
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.cfg.device)

        with torch.no_grad():
            q_values = self.policy_net(states)
            # Illegal actions are excluded with a true negative infinity instead of a fixed
            # finite floor.  A floor only masks while the network output stays above it, and the
            q_values_masked = q_values.masked_fill(masks == 0, float('-inf'))
            greedy_actions = q_values_masked.argmax(dim=1)

        if explore_mask.any():
            for i in range(batch_size):
                if explore_mask[i]:
                    valid = torch.nonzero(masks[i]).squeeze()
                    if valid.ndim == 0:
                        valid = valid.unsqueeze(0)
                    if len(valid) == 0:
                        actions[i] = greedy_actions[i]
                    else:
                        rand_idx = torch.randint(0, len(valid), (1,), device=self.cfg.device)
                        actions[i] = valid[rand_idx]

        if (~explore_mask).any():
            actions[~explore_mask] = greedy_actions[~explore_mask]

        # Final validity check
        actions_cpu = actions.cpu()
        masks_cpu = masks.cpu().numpy()
        for i in range(batch_size):
            act = actions_cpu[i].item()
            if act >= masks.shape[1] or masks_cpu[i, act] == 0:
                valid_indices = np.nonzero(masks_cpu[i])[0]
                if len(valid_indices) > 0:
                    actions_cpu[i] = np.random.choice(valid_indices)
                else:
                    actions_cpu[i] = 0
        return actions_cpu.numpy().tolist()

    def update_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            return False
        updated = False
        for _ in range(self.cfg.gradient_steps):
            if len(self.memory) < self.cfg.batch_size:
                break
            s, a, r, ns, d, m, nm = self.memory.sample(self.cfg.batch_size)
            a = a.unsqueeze(1); r = r.unsqueeze(1); d = d.unsqueeze(1)
            if self.scaler:
                with autocast():
                    q = self.policy_net(s)
                    current_q = q.gather(1, a)
                    with torch.no_grad():
                        next_q = self.target_net(ns)
                        # The illegal actions of the next state are removed from the *choice* with a
                        # true -inf, and the bootstrapped value is then read from the unmasked
                        next_online = self.policy_net(ns)
                        next_actions = next_online.masked_fill(nm == 0, float('-inf')).argmax(
                            dim=1, keepdim=True)
                        next_max = next_q.gather(1, next_actions)
                        target = (r * self.cfg.target_scale
                                  + (1 - d) * self.cfg.gamma * next_max)
                        # The target stays in the absolute units of the return.  Centring and
                        # scaling it over a batch - the analogue of the advantage normalisation of
                    loss = self.loss_fn(current_q, target)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                q = self.policy_net(s)
                current_q = q.gather(1, a)
                with torch.no_grad():
                    next_q = self.target_net(ns)
                    # Illegal actions removed from the choice with a true -inf, and the
                    # bootstrapped value read from the unmasked estimate, as in the branch above.
                    next_online = self.policy_net(ns)
                    next_actions = next_online.masked_fill(nm == 0, float('-inf')).argmax(
                        dim=1, keepdim=True)
                    next_max = next_q.gather(1, next_actions)
                    target = (r * self.cfg.target_scale
                              + (1 - d) * self.cfg.gamma * next_max)
                    # The same absolute target as in the branch above, for the same reason.
                loss = self.loss_fn(current_q, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_norm)
                self.optimizer.step()
            self.recent_q_values.append(q.mean().item())
            self.recent_losses.append(loss.item())
            updated = True
            # ``target_update_freq`` counts gradient steps, which is what its name says: the
            # counter used to be advanced once per epoch, so with 256 steps per epoch the target
            self.steps_done += 1
            if self.steps_done % self.cfg.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        if updated:
            self.scheduler.step()

        return updated

    def get_stats(self):
        avg_q = np.mean(self.recent_q_values) if self.recent_q_values else 0
        avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
        buffer_usage = len(self.memory) / self.cfg.buffer_capacity
        target_updates = self.steps_done // self.cfg.target_update_freq
        return {'avg_q': avg_q, 'avg_loss': avg_loss, 'buffer_usage': buffer_usage, 'target_updates': target_updates}


# 10. Environment manager

class EnvManager:
    def __init__(self, cfg, instance_list, num_envs, agent, state_norm):
        self.cfg = cfg
        self.num_envs = num_envs
        self.envs = []
        self.states = []          # store numpy arrays
        self.masks = []
        self.dones = [True] * num_envs
        self.episode_rewards = [0.0] * num_envs
        self.step_counts = [0] * num_envs
        self.total_ops_list = []
        self.instance_list = instance_list
        # Use curriculum sampler instead of InstanceSampler
        self.curriculum_sampler = CurriculumSampler(instance_list, cfg)
        self.agent = agent
        self.state_norm = state_norm
        self.device = cfg.device

        for i in range(num_envs):
            self._reset_env(i)

    def _reset_env(self, idx):
        inst = self.curriculum_sampler.sample()
        env = MOFJSP_Env(self.cfg, inst)
        if idx < len(self.envs):
            self.envs[idx] = env
        else:
            self.envs.append(env)
        # The state and the mask replace the entries of the environment that was reset instead of
        # being appended: an appended entry is discarded at the end of the collection loop, which
        # left the state list one decision behind the environment it belongs to.
        state = env.reset()
        # The mask of this baseline is trivial: every rule resolves to a legal action, so every rule
        # is available in every state.
        mask = np.ones(N_RULES, dtype=np.float32)
        if idx < len(self.states):
            self.states[idx] = state
            self.masks[idx] = mask
            self.total_ops_list[idx] = env.inst.total_ops
        else:
            self.states.append(state)
            self.masks.append(mask)
            self.total_ops_list.append(env.inst.total_ops)
        self.dones[idx] = False
        self.episode_rewards[idx] = 0.0
        self.step_counts[idx] = 0
        # No append here.  The two branches above already hold one entry per environment, and a
        # third append grew the list by one entry at every reset: after a few dozen episodes the

    def step_all(self):
        """Take a step for all environments, return a list of experiences"""
        states_np = np.stack(self.states)
        masks_np = np.stack(self.masks)
        states = torch.from_numpy(states_np).to(self.device)
        masks = torch.from_numpy(masks_np).to(self.device)

        # ========== Online state normalization update ==========
        if self.state_norm is not None:
            self.state_norm.update(states)          # Update statistics using the current batch
            states_norm = self.state_norm.normalize(states)
        else:
            states_norm = states
        # ======================================================

        actions = self.agent.select_action_batch(states_norm, masks, self.agent.epsilon)

        next_states = []
        rewards = []
        dones = []
        infos = []
        for i, rule in enumerate(actions):
            # The rule the value function chose is resolved to the action it names among the legal ones.
            legal = rule_actions(self.envs[i])
            act = int(legal[int(rule)]) if len(legal) else 0
            try:
                ns, r, d, info = self.envs[i].step(act)
            except Exception as e:
                print(f"Environment {i} step error: {e}, action={act}, resetting environment")
                ns = self.envs[i].reset()
                r = 0.0
                d = False
                info = {}
                self.dones[i] = False
            self.episode_rewards[i] += r
            self.step_counts[i] += 1
            if d:
                info['episode_reward'] = self.episode_rewards[i]
                info['total_ops'] = self.total_ops_list[i]
                self._reset_env(i)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        new_masks = [np.ones(N_RULES, dtype=np.float32) for _ in range(self.num_envs)]

        # A finished episode hands its terminal state to its own transition and the state of the
        # new episode to the next decision: `_reset_env` installed that state for this index, while
        # the terminal state of the finished episode belongs to the transition that just ended.
        for i, done in enumerate(dones):
            if done:
                next_states[i] = self.states[i]
        self.states = next_states
        self.masks = new_masks

        experiences = []
        for i in range(self.num_envs):
            exp = (
                states_np[i], actions[i], rewards[i], next_states[i], dones[i],
                masks_np[i], new_masks[i], infos[i] if dones[i] else None
            )
            experiences.append(exp)
        return experiences


# 11. Main training function

def weighted_objective(objectives):
    """Weighted objective of one solution: [makespan, load balance, total tardiness]."""
    weights = Config.objective_weights
    return weights[0] * objectives[0] + weights[1] * objectives[1] + weights[2] * objectives[2]


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


def _norm_extra(state_norm):
    """Statistics of the state normaliser, ready to be stored next to the weights.

    They are stored as plain lists: the weights are read by interpreters of two different
    environments, and a tensor of one numpy major version does not unpickle in the other.
    """
    if state_norm is None:
        return {}
    return {
        'norm_mean': [float(v) for v in state_norm.mean.detach().cpu().reshape(-1)] if torch.is_tensor(state_norm.mean) else [float(v) for v in np.asarray(state_norm.mean).reshape(-1)],
        'norm_var': [float(v) for v in state_norm.var.detach().cpu().reshape(-1)] if torch.is_tensor(state_norm.var) else [float(v) for v in np.asarray(state_norm.var).reshape(-1)],
        'norm_count': float(state_norm.count),
    }


def evaluate_validation(agent, validation, cfg, state_norm=None, seed=0):
    """Greedy rollouts of the validation instances; returns [Cmax, LB, Tardy] per instance.

    Greedy decoding follows the trained Q-values and dominates the epsilon-greedy exploration of
    the collection phase, so the validation objective is measured on it.  The running statistics
    of the state normaliser are frozen for the whole block: a validation rollout must not change
    the normalisation of the training run.
    """
    was_training = agent.policy_net.training
    agent.policy_net.eval()
    objectives = []
    # One deterministic rollout plus k-1 sampled ones, the best of the set scores the instance:
    # the protocol the comparison reports every method with.
    VALIDATION_ROLLOUTS = 4
    for inst in validation:
        best = None
        for run_index in range(VALIDATION_ROLLOUTS):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = MOFJSP_Env(cfg, inst)
            raw_state = env.reset()
            done = False
            info = {}
            steps = 0
            limit = max(1, inst.total_ops + 1)
            while not done and steps < limit:
                mask = np.ones(N_RULES, dtype=np.float32)
                state_t = torch.from_numpy(np.asarray(raw_state, dtype=np.float32)).to(cfg.device)
                if state_norm is not None:
                    state_t = state_norm.normalize(state_t)
                mask_t = torch.from_numpy(mask.astype(np.float32)).to(cfg.device)
                with torch.no_grad():
                    q_values = agent.policy_net(state_t.unsqueeze(0)).squeeze(0)
                    # Negative infinity, as in select_action_batch: a finite floor is only an
                    # exclusion while the Q output stays above it, and a rollout that stops early
                    # is scored as the pessimistic partial schedule below.
                    q_values = q_values.masked_fill(mask_t == 0, float('-inf'))
                    if run_index == 0:
                        rule = int(q_values.argmax().item())
                    else:
                        rule = int(torch.multinomial(
                            torch.softmax(q_values, dim=0), 1).item())
                # The rule is resolved to the legal action it names before the environment is stepped.
                legal = rule_actions(env)
                action = int(legal[rule]) if len(legal) else 0
                next_raw, _, done, info = env.step(action)
                if next_raw is not None:
                    raw_state = next_raw
                steps += 1
            metrics = info.get('final_metrics') if done else None
            if metrics is None:
                # The rollout stopped before every operation was scheduled, either because an
                # invalid action was proposed or because the step limit was reached.  Such a
                remaining = max(0, int(inst.total_ops) - int(env.finished_ops_count))
                machines = env.machine_avail_time[:inst.num_machines]
                loads = env.machine_load[:inst.num_machines]
                horizon = float(machines.max()) if machines.size else 0.0
                horizon += remaining * float(inst.max_proc_time)
                tardy = sum(max(0.0, float(env.job_avail_time[j]) - float(inst.due_dates[j]))
                            for j in range(inst.num_jobs))
                tardy += remaining * float(inst.max_due_date)
                metrics = {'Cmax': horizon,
                           'LB': float(np.std(loads)) if loads.size else 0.0,
                           'Tardy': float(tardy)}
            candidate = [float(metrics['Cmax']), float(metrics['LB']),
                         float(metrics['Tardy'])]
            if best is None or weighted_objective(candidate) < weighted_objective(best):
                best = candidate
        objectives.append(best)
    if was_training:
        agent.policy_net.train()
    return objectives


def train():
    cfg = Config()
    set_seed()

    # Load instances
    print("Loading instances...")
    start_load = time.time()
    instance_tuples, max_jobs, max_machines, global_max_proc, global_max_due = \
        load_all_instances_structured(cfg.instance_dir)
    cfg.max_jobs = max_jobs
    cfg.max_machines = max_machines
    cfg.global_max_proc_time = global_max_proc
    cfg.global_max_due_date = global_max_due

    instance_list = []
    for data, fname, size in instance_tuples:
        inst = JobShopInstance(data, max_jobs, max_machines, fname, size)
        instance_list.append(inst)
    print(f"Instance loading completed, time {time.time() - start_load:.2f}s")

    # The reported weights are selected on a fixed validation set, one that the curriculum
    # sampler cannot move: the sampler changes the instance distribution over the run, so the
    val_tuples = load_all_instances_structured(cfg.instance_dir, "mo_fjsp_*_val.txt")[0]
    validation_instances = [JobShopInstance(data, max_jobs, max_machines, fname, size)
                            for (data, fname, size) in val_tuples]
    validation_set = build_validation_set(validation_instances, per_size=cfg.val_instances_per_size)
    print(f"Validation set: {[inst.file_name for inst in validation_set]}")

    agent = DQN_Agent(cfg)
    print(f"Device: {cfg.device}")
    print(f"State dimension: {agent.state_dim}, action dimension: {agent.action_dim}")
    print(f"Number of epochs: {cfg.epochs}")
    print(f"Number of parallel environments: {cfg.num_envs}")

    # State normalization initialization (initialize with a few samples, then update online)
    if cfg.use_state_norm:
        state_norm = RunningMeanStd(shape=(agent.state_dim,), device=cfg.device)
        print(f"\nPre-collecting {cfg.precollect_episodes} episodes...")
        start_pre = time.time()
        states_collected = []
        for _ in range(cfg.precollect_episodes):
            inst = random.choice(instance_list)
            env = MOFJSP_Env(cfg, inst)
            state = env.reset()
            states_collected.append(state)
            done = False
            step = 0
            while not done and step < 100:
                mask = env.get_action_mask()
                valid = np.nonzero(mask)[0]
                if len(valid) == 0:
                    break
                action = np.random.choice(valid)
                next_state, _, done, _ = env.step(action)
                states_collected.append(next_state)
                state = next_state
                step += 1
        if states_collected:
            states_tensor = torch.from_numpy(np.stack(states_collected)).to(cfg.device)
            state_norm.update(states_tensor)
        print(f"Pre-collection completed, time {time.time() - start_pre:.2f}s, collected {len(states_collected)} state samples.\n")
        # Every checkpoint written from here carries these statistics, so an evaluation can rebuild
        # the normalisation the weights were trained with.
        agent.norm_extra = _norm_extra(state_norm)
    else:
        state_norm = None

    # The transitions of the environments are handed to the replay buffer through the n-step window
    # of this run (see Config.n_step).
    n_step_window = NStepAccumulator(cfg.n_step, cfg.gamma, cfg.num_envs)
    env_manager = EnvManager(cfg, instance_list, cfg.num_envs, agent, state_norm)

    episode_count = 0
    normalized_rewards = []
    reward_ema = None
    final_info = None
    step_counter = 0
    last_log_interval = 0

    # Continue an interrupted run from its rolling checkpoint.  The optimiser moments, the
    # episode counter and the validation record are restored, so a restart repeats at most
    # checkpoint_interval episodes instead of the whole run.
    val_curve = list(agent.val_curve)
    val_reference = list(agent.val_reference) or None
    best_val = float(agent.best_val)
    stale_checks = 0
    converged = False
    if agent._load_checkpoint(agent.rollout_path):
        episode_count = int(agent.episode_counter)
        # The selected model is the one with the best validation objective of the run, and the
        # checkpoint carries that objective.  Reading it before the load left the local counter at
        best_val = float(agent.best_val)
        print(f"Continuing from episode {episode_count}/{cfg.epochs} (best_val={best_val:.6f})")
        # The patience of the early stop survives a restart.  A queue that restarts a run after
        # every shutdown would otherwise grant it a fresh patience window and train it past its
        stale_checks = 0
        running_best = float('inf')
        for _episode, _score in val_curve:
            if _score < running_best * (1.0 - cfg.val_tol):
                running_best = _score
                stale_checks = 0
            else:
                stale_checks += 1
        # The statistics of the checkpoint describe the states the restored weights were trained on.
        # The normaliser continues from them instead of the pre-collected sample: the two are not
        # the same distribution, and the greedy decode is sensitive to the scale of its input.
        saved_norm = getattr(agent, 'checkpoint_norm', None)
        if cfg.use_state_norm and state_norm is not None and saved_norm:
            state_norm.mean = torch.tensor(saved_norm['norm_mean'], device=cfg.device, dtype=torch.float32)
            state_norm.var = torch.tensor(saved_norm['norm_var'], device=cfg.device, dtype=torch.float32)
            state_norm.count = float(saved_norm.get('norm_count', 1.0))
            agent.norm_extra = _norm_extra(state_norm)
            print(f"State normaliser restored (count={state_norm.count:.1f})")
        # An early stop is the decision of the run, not an interruption of it: the validation
        # objective of the model stopped improving, and more episodes would only be trained past
        if agent.converged:
            print("The rolling checkpoint already holds a converged run; nothing to train.")
            return
    else:
        print(f"No rolling checkpoint {agent.rollout_path} found, training starts from scratch.")

    # The milestones of a resumed run are not repeated: both counters start at the interval the
    # restored episode counter already falls into.
    last_val_interval = episode_count // cfg.val_interval
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
    last_checkpoint_interval = episode_count // cfg.checkpoint_interval

    try:
        while episode_count < cfg.epochs and not converged:
            # Update curriculum stage.  It advances on a fixed episode scale instead of on the
            # epoch budget, so the schedule does not change when the budget is changed.
            progress = min(1.0, episode_count / max(1, cfg.curriculum_episodes))
            env_manager.curriculum_sampler.update_stage(progress)

            experiences = env_manager.step_all()
            step_counter += 1
            agent.update_epsilon()

            for env_index, exp in enumerate(experiences):
                state, action, reward, next_state, done, mask, next_mask, info = exp
                state_gpu = torch.from_numpy(state).to(cfg.device)
                next_state_gpu = torch.from_numpy(next_state).to(cfg.device)
                mask_gpu = torch.from_numpy(mask).to(cfg.device)
                next_mask_gpu = torch.from_numpy(next_mask).to(cfg.device)
                action_t = torch.tensor(action, dtype=torch.long, device=cfg.device)
                reward_t = torch.tensor(reward, dtype=torch.float32, device=cfg.device)
                done_t = torch.tensor(done, dtype=torch.float32, device=cfg.device)

                n_step_window.push(agent.memory, env_index, state_gpu, action_t, reward_t,
                                   next_state_gpu, done_t, mask_gpu, next_mask_gpu)

                if done:
                    episode_count += 1
                    if info is not None and 'episode_reward' in info:
                        norm_reward = info['episode_reward'] / info['total_ops']
                        normalized_rewards.append(norm_reward)
                        if reward_ema is None:
                            reward_ema = norm_reward
                        else:
                            reward_ema = 0.99 * reward_ema + 0.01 * norm_reward
                    if info is not None and 'final_metrics' in info:
                        final_info = info

            # Periodically train
            if step_counter % cfg.steps_per_update == 0:
                agent.update()

            # Validation block.  It selects the reported weights and it stops the run as soon as
            # the average objective stops improving, i.e. the run is never trained past the point
            val_interval_index = episode_count // cfg.val_interval
            if (episode_count >= cfg.min_epochs and episode_count > 0
                    and val_interval_index > last_val_interval):
                last_val_interval = val_interval_index
                objectives = evaluate_validation(agent, validation_set, cfg,
                                                 state_norm=state_norm, seed=cfg.val_seed)
                scores = [weighted_objective(obj) for obj in objectives]
                if val_reference is not None and len(val_reference) != len(scores):
                    # A validation set that was resized after the checkpoint was written would
                    # otherwise be scored against the reference of the set it used to have, and
                    # a score of the new frame could never beat the best of the old one.
                    val_reference = None
                    best_val = float('inf')
                    stale_checks = 0
                if val_reference is None:
                    # The first check defines the reference of every instance, so the score is a
                    # relative objective that does not depend on the instance size which
                    # dominates the raw weighted value.
                    val_reference = [value if abs(value) > 1e-9 else 1.0 for value in scores]
                val_score = float(np.mean([value / ref
                                           for value, ref in zip(scores, val_reference)]))
                val_curve.append((episode_count, val_score))
                print(f"Validation {len(val_curve):3d} at episode {episode_count}: "
                      f"relative objective {val_score:.6f} (best {best_val:.6f})")
                if val_score < best_val * (1.0 - cfg.val_tol):
                    best_val = val_score
                    stale_checks = 0
                    agent.episode_counter = episode_count
                    agent.val_curve = list(val_curve)
                    agent.val_reference = list(val_reference)
                    agent.best_val = best_val
                    agent.converged = False
                    agent._save_model()
                else:
                    stale_checks += 1
                    if stale_checks >= cfg.val_patience:
                        print(f"Early stop after {episode_count} episodes: the validation "
                              f"objective did not improve for {stale_checks} checks "
                              f"(best={best_val:.6f}).")
                        converged = True

            # The rolling checkpoint has to survive the jumping counter as well: it is the file a
            # restarted run continues from, so a skipped milestone would cost the whole run.
            checkpoint_interval_index = episode_count // cfg.checkpoint_interval
            if episode_count > 0 and checkpoint_interval_index > last_checkpoint_interval:
                last_checkpoint_interval = checkpoint_interval_index
                agent.episode_counter = episode_count
                agent.val_curve = list(val_curve)
                agent.val_reference = list(val_reference) if val_reference else []
                agent.best_val = best_val
                agent.converged = False
                agent._save_checkpoint(agent.rollout_path)

            # Logging
            current_interval = episode_count // cfg.log_interval
            if current_interval > last_log_interval:
                last_log_interval = current_interval
                display_episode = current_interval * cfg.log_interval

                avg_norm = np.mean(normalized_rewards[-100:]) if normalized_rewards else 0
                # A resumed run has not finished an episode yet, so its moving average of the
                # return is still empty and the report below would fail to format it.
                report_reward = reward_ema if reward_ema is not None else float('nan')
                stats = agent.get_stats()
                lr = agent.optimizer.param_groups[0]['lr']
                metrics = ""
                if final_info and 'final_metrics' in final_info:
                    m = final_info['final_metrics']
                    metrics = f" | Cmax={m['Cmax']:.1f} LB={m['LB']:.2f} Tardy={m['Tardy']:.1f}"
                # Print current allowed sizes
                allowed_sizes = env_manager.curriculum_sampler.allowed_sizes
                print(
                    f"Episode {display_episode:6d}/{cfg.epochs} | Sizes: {allowed_sizes} | NormReward: {report_reward:8.4f} | "
                    f"Avg100: {avg_norm:8.4f} | Epsilon: {agent.epsilon:.4f} | AvgQ: {stats['avg_q']:6.2f} | "
                    f"Loss: {stats['avg_loss']:.4f} | Buffer: {stats['buffer_usage']*100:3.0f}% | LR: {lr:.2e}{metrics}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    agent.episode_counter = episode_count
    agent.val_curve = list(val_curve)
    agent.val_reference = list(val_reference) if val_reference else []
    agent.best_val = best_val
    agent.converged = bool(converged)
    agent._save_checkpoint(agent.rollout_path)
    print("Training completed." if not converged else "Training stopped early (converged).")
    print(f"Best validation objective: {best_val:.6f}")
    print(f"Rolling checkpoint saved to {agent.rollout_path}")

    # Validation curve: the relative objective of the fixed validation set.  It is the quantity
    # that selects the reported weights and that stops the run.
    if val_curve:
        plt.figure(figsize=(8, 3.2))
        plt.plot([e for e, _ in val_curve], [s for _, s in val_curve], marker='o', linewidth=1.2,
                 color='tab:red')
        plt.xlabel("Episode")
        plt.ylabel("Relative validation objective")
        plt.title("Validation curve (lower is better)")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "dqn_validation_curve.png"), dpi=150)

    # Plotting (with improved y-axis range)
    if normalized_rewards:
        rewards = np.array(normalized_rewards)
        episodes = np.arange(len(rewards))
        window = min(100, len(rewards) // 10)
        if window > 1:
            weights = np.ones(window) / window
            smooth_rewards = np.convolve(rewards, weights, mode='valid')
            smooth_episodes = episodes[window - 1:]
        else:
            smooth_rewards = rewards
            smooth_episodes = episodes
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        plt.plot(smooth_episodes, smooth_rewards, color='red', linewidth=2, label=f'Smooth (window={window})')
        # Use actual data range plus 5% margin
        y_min = np.min(rewards)
        y_max = np.max(rewards)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        plt.title("DQN Training Curve (Smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Normalized Reward")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "dqn_curve.png"), dpi=150)
        plt.show()
    else:
        print("No reward data to plot.")


if __name__ == "__main__":
    # Optional episode budget, mirroring the --episodes flag of the graph trainers, so that a run
    # can target the same number of episodes as the comparison methods.
    arguments = sys.argv[1:]
    budget = None
    for position, argument in enumerate(arguments):
        if argument == "--epochs":
            budget = int(arguments[position + 1])
    if budget is not None and budget > 0:
        Config.epochs = budget
        print(f"epochs overridden to {Config.epochs}")
    train()

