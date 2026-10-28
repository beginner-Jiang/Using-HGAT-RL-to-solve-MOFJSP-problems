"""
HGHRL Training Program
Fully implements PPO (using KL divergence, combined with hierarchical reinforcement learning) + HGAT (heterogeneous graph)
After training, plots the normalized curves of three objectives (Cmax, Load Balance, Tardy), KL divergence, and total loss variation trends
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

# Intra-op thread count: every training run is launched with a slice of the CPU
# budget, and the torch default (one thread per physical core) oversubscribes the
THREAD_BUDGET = int(os.environ.get('MOFJSP_HGAT_THREADS', '6') or 0)
if THREAD_BUDGET > 0:
    torch.set_num_threads(THREAD_BUDGET)
import re

# 1. Global Configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

# Multi-objective weights
W1 = 0.4  # Cmax

W2 = 0.3  # Load Balance
W3 = 0.3  # Tardy

# Competing operation pairs kept per machine.  The disjunctive edges of the manuscript connect
# every pair of operations that can be processed on the same machine, which is quadratic in the
DISJ_PAIRS_PER_MACHINE = 50

# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Instance, model and result directories, anchored at the project root so that
# no input and no output location depends on the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR = os.path.join(_PROJECT_ROOT, "DataSet", "mo_fjsp_instances")
FILE_PATTERN = "mo_fjsp_*_train.txt"
MODEL_DIR = os.path.join(_PROJECT_ROOT, "model", "hghrl")
MODEL_NAME = "hghrl_best.pth"
# Tag of the hyper-parameter recipe of this generation.  Every checkpoint records it, and a resumed
# run adopts the schedule and the KL penalty of a checkpoint only when the checkpoint belongs to the
RECIPE_TAG = "hetero-state-2026-10-17-legacy3"
TRAIN_FIGURE_DIR = os.path.join(_PROJECT_ROOT, "Figure_And_File", "HGHRL",
                                "Train")

torch.backends.cudnn.benchmark = True

# 2. Configuration class

class Config:
    # Episode budget of this wave.  The wave before it held its best validation at episode 1500 and
    # then degraded for four thousand episodes (the relative objective of the last check was 6.895
    n_episodes = 3000
    # Rolling checkpoint of an interrupted run: it also carries the optimiser
    # state and the episode counter, so that a restart continues instead of
    # starting from the first episode again.
    resume_name = 'hghrl_last.pth'
    checkpoint_interval = 100
    # Peak step size of the policy update.  The wave of the previous generation used 1e-3 and the
    # policy collapsed after its early checks (the relative validation objective doubled and the
    lr = 2e-4
    lr_decay = 0.5
    lr_decay_steps = 2000
    min_lr = 1e-5
    gamma = 0.99
    # Undiscounted return of the episode.  One episode of this method is the complete schedule of
    # an instance, i.e. 60 to 450 decisions, and the objective is only complete at its end: at
    gamma = 1.0
    gae_lambda = 1.0
    # Monte-Carlo advantage.  The lambda of the GAE is the window of the credit assignment: at
    # 0.95 the advantage of a decision only contains the temporal differences of the next twenty
    gae_lambda = 0.95
    # The target of the KL penalty, and the floor the adapted coefficient cannot fall below.  Measured
    # on the plateaued wave: the penalty was two orders of magnitude below its floor because the
    kl_target = 0.1
    beta_floor = 0.01
    beta_start = 0.2
    # Adaptation rate of the KL penalty.  The arms of this family use 0.05 and hold the policy
    # inside the trust region of kl_target; the 0.01 of the collapsed wave moved the penalty
    # five times slower than the drift it was supposed to correct.
    beta_lr = 0.05
    value_coef = 0.5
    entropy_coef = 0.01
    # Exploration kept alive for the whole budget.  The previous wave decayed this factor per update,
    # which halves the coefficient within one run; a policy whose exploration vanished stops visiting
    entropy_decay = 0.9995
    # Gradient clip of the shared recipe of this family.  The 20.0 of the collapsed wave clipped
    # the joint update of the encoder and of both policies hard enough to change the direction of
    max_grad_norm = 10.0
    # Back to the value of the generation that trained: fifty.  The ten was carried over from the
    # family when the value term of the GAT baseline had to be stopped from diverging, but it is
    max_grad_norm = 50.0
    # Capacity of this wave.  The reported model was written at episode 500 of its run and 3500
    # further episodes never improved it, while a simpler baseline of the same width kept
    gat_hidden_dim = 256
    gat_out_dim = 256
    policy_hidden_dim = 512
    value_clip = None
    # Enabled.  The value term of this family diverges without a bound: the measured GAT
    # baseline of the same recipe printed a total loss of 1.1e6 because its value estimates
    value_clip = 0.5
    # Back to None.  Enabling the clip exposed a latent defect of the update of this file: its
    # value-clip branch reads "values" from the enclosing scope, and that name is not bound on every
    value_clip = None
    reward_scaling = 5.0
    reward_clip = 2.0
    # Running mean and deviation of the graph embedding.  Every arm of this family trains with
    # the normalisation enabled and the collapsed wave trained without it, which fed the critic
    # and the policies an embedding whose scale follows the instance size.
    use_state_norm = True
    use_disjunctive_edges = True
    max_disj_edges_per_machine = 50
    sampling_temperature = 1.0
    max_steps_per_episode = 2000
    stats_window = 100
    max_edges_per_type = 30000
    # Instances collected in lockstep.  The update below still processes one episode at a time, so
    # this is a throughput knob only: the encoder, the policy and the critic process this many states
    num_envs = 6
    amp_enabled = False
    logits_clip = 50.0

    # Model selection and early stopping.  One episode is a noisy sample of the policy (its
    # instance scale and its operation count change from episode to episode), so the reported
    val_interval = 250
    # Four instances per size halve the noise of the six instance set that the previous
    # generation selected its weights on: the best checkpoint and the early stop of a run whose
    val_instances_per_size = 8
    # Validation opens before the peak of a fast converging method: the best checkpoint of the
    # previous generation was written at episode 1750, and the DAN baseline peaked even earlier.
    min_episodes = 1500
    # A stale check is one validation of 250 episodes, so a patience of six holds a run
    # 1500 episodes past its peak (the generation before it used eight, i.e. 2000).
    val_patience = 8
    val_tol = 0.02
    val_seed = 0
    # Smoothing window of the training curve that the figures report.
    select_window = 100
    # Floor of the entropy coefficient.  The decay alone drives the coefficient to 1e-4 over a
    # run of ten thousand episodes, i.e. the policy becomes deterministic before the curriculum
    entropy_min = 0.005

    # Curriculum learning parameters
    curriculum_enabled = True
    # The curriculum advances on a fixed episode scale instead of on the --episodes budget: a
    # stage boundary that moves with the budget would compare episodes drawn from different
    curriculum_episodes = 2500
    # Aligned with every baseline of the study.  This value decides when the curriculum introduces
    # the medium and the large instances, so it decides the training distribution itself: at 2500
    curriculum_episodes = 4000
    curriculum_stages = [
        (0.0, 0.3, ['small']),
        (0.3, 0.6, ['small', 'medium']),
        (0.6, 1.0, ['small', 'medium', 'large'])
    ]

cfg = Config()


# 3. Data reading and instance definition

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
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    if len(lines) < 3:
        raise ValueError(f"Instance file {file_path} has insufficient data lines")

    num_jobs, num_machines = map(int, lines[0].split())
    job_lines = lines[1:1 + num_jobs]
    due_dates_line = lines[-2]
    capabilities_line = lines[-1]
    due_dates = list(map(float, due_dates_line.split()))
    machine_capabilities = list(map(int, capabilities_line.split()))

    if len(due_dates) != num_jobs:
        print(f"Warning: number of due dates ({len(due_dates)}) does not match number of jobs ({num_jobs})")
    if len(machine_capabilities) != num_machines:
        print(f"Warning: number of machine capabilities ({len(machine_capabilities)}) does not match number of machines ({num_machines})")

    jobs = []
    for job_idx, line in enumerate(job_lines):
        nums = list(map(int, line.split()))
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
        # Normalised processing time of every operation on every machine: the first term of the
        # fixed-preference heuristic of the lower policy (Eq. 10 of the manuscript).  It is static
        self.pt_rows_norm = torch.where(self.proc_time_matrix > 99998.0,
                                        torch.zeros_like(self.proc_time_matrix),
                                        self.proc_time_matrix / (global_max_proc_time + EPS))
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

        # Disjunctive edges of the manuscript.  Two operations that can be processed on the same
        # machine compete for it, and this third edge type is what tells the encoder about that
        disj_src = []
        disj_dst = []
        for machine in range(self.n_machines):
            ops = self.machine_ops_list[machine].tolist()
            pairs = 0
            for first in range(len(ops)):
                for second in range(first + 1, len(ops)):
                    if pairs >= DISJ_PAIRS_PER_MACHINE:
                        break
                    disj_src.append(ops[first])
                    disj_dst.append(ops[second])
                    disj_src.append(ops[second])
                    disj_dst.append(ops[first])
                    pairs += 1
                if pairs >= DISJ_PAIRS_PER_MACHINE:
                    break
        self.disj_edges_src = torch.tensor(disj_src, dtype=torch.long, device=DEVICE)
        self.disj_edges_dst = torch.tensor(disj_dst, dtype=torch.long, device=DEVICE)
        self.disj_edges_feat = torch.full((len(disj_src), 1), 2.0, dtype=torch.float32,
                                          device=DEVICE)

    def get_pt(self, job, op, machine):
        return self.proc_time_matrix[self.op_index_map[(job, op)], machine].item()


# 4. Curriculum learning sampler

class CurriculumSampler:
    def __init__(self, instances: List[MOFJSPInstance], cfg: Config):
        self.instances = instances
        self.cfg = cfg
        self.instances_by_size = {'small': [], 'medium': [], 'large': []}
        for inst in instances:
            if inst.size in self.instances_by_size:
                self.instances_by_size[inst.size].append(inst)
        self.allowed_sizes = ['small']
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

    def sample(self) -> MOFJSPInstance:
        allowed_instances = []
        for sz in self.allowed_sizes:
            allowed_instances.extend(self.instances_by_size.get(sz, []))
        if not allowed_instances:
            allowed_instances = self.instances
        return random.choice(allowed_instances)


# 5. State normalization utility

class TorchRunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4, device=DEVICE):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.device = device

    def update(self, x):
        # A non-finite sample must not enter the statistics: the normaliser is not part of
        # the optimiser, so one NaN batch would make every later state NaN and the policy
        if not bool(torch.isfinite(x).all().item()):
            return
        if (not bool(torch.isfinite(self.mean).all().item())
                or not bool(torch.isfinite(self.var).all().item())):
            self.mean = torch.zeros_like(self.mean)
            self.var = torch.ones_like(self.var)
            self.count = 1e-4
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
        # The caller keeps running after an invalid state, so the normalisation has to stay
        # finite even when the sample or the statistics are not.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(self.var.sqrt(), nan=1.0, posinf=1.0, neginf=1.0)
        return (x - mean) / (std + 1e-8)


# 6. Heterogeneous graph environment

class HeteroGraphEnv:
    def __init__(self, instance: MOFJSPInstance, global_max_jobs, global_max_ops, global_max_machines,
                 global_max_proc_time, global_max_due_date, cfg, feature_mode="legacy"):
        self.inst = instance
        # "augmented" is the proposed method's compact table: position, due date, job id,
        # earliest start and remaining work (5 operation columns, 3 machine columns).
        self.feature_mode = feature_mode
        self.global_max_jobs = global_max_jobs
        # The operation-feature table of the environment is sized by the global maximum of the
        # TRAINING family, because the policies were normalised with that tuple.  A comparison
        self.global_max_ops = max(global_max_ops, instance.total_ops)
        self.global_max_machines = global_max_machines
        self.global_max_proc_time = global_max_proc_time
        self.global_max_due_date = global_max_due_date
        self.cfg = cfg

        self.global_max_proc_time_tensor = torch.tensor(global_max_proc_time, device=DEVICE)
        self.global_max_due_date_tensor = torch.tensor(global_max_due_date, device=DEVICE)

        self.total_op_nodes = instance.total_ops
        self.total_machine_nodes = instance.n_machines

        if feature_mode == "full":
            # Appendix B, Table B2: [min processing time, max processing time, step position,
            # remaining operations, due date, job id, earliest start].
            self.op_feat_dim = 7
            # Table B2 machine features: capability, machine id, load and availability.
            self.mac_feat_dim = 4
        elif feature_mode == "augmented":
            # Intermediate enhancement (position, due date, job id, readiness, remaining work).
            self.op_feat_dim = 5
            self.mac_feat_dim = 3
        else:
            # Original shared environment: position, due date, job id; load/availability/
            # capability for the machines.
            self.op_feat_dim = 3
            self.mac_feat_dim = 3

        self.op_predecessor = instance.op_predecessor
        self.job_op_to_idx = instance.job_op_to_idx
        self.op_mac_mask = instance.op_mac_mask
        self.proc_time_matrix = instance.proc_time_matrix
        self.due_dates_tensor = instance.due_dates_tensor
        self.ops_per_job_tensor = instance.ops_per_job_tensor
        self.op_idx_to_job = instance.op_idx_to_job
        self.op_idx_to_op = instance.op_idx_to_op

        self.seq_edges_src = instance.seq_edges_src
        self.seq_edges_dst = instance.seq_edges_dst
        self.alloc_edges_src = instance.alloc_edges_src
        self.alloc_edges_dst = instance.alloc_edges_dst
        self.alloc_edges_feat = instance.alloc_edges_feat
        self.alloc_rev_edges_src = instance.alloc_rev_edges_src
        self.alloc_rev_edges_dst = instance.alloc_rev_edges_dst
        self.alloc_rev_edges_feat = instance.alloc_rev_edges_feat

        self.machine_ops_list = instance.machine_ops_list

        self.static_op_feats = torch.zeros((self.global_max_ops, self.op_feat_dim), dtype=torch.float32, device=DEVICE)
        matrix = instance.proc_time_matrix[:instance.total_ops]
        eligible = matrix < 99998.0
        min_proc = torch.where(eligible, matrix,
                              torch.full_like(matrix, float("inf"))).min(dim=1).values
        max_proc = torch.where(eligible, matrix,
                              torch.zeros_like(matrix)).max(dim=1).values
        if feature_mode == "full":
            # Appendix B, Table B2.
            for job in range(instance.n_jobs):
                ops_in_job = instance.ops_per_job[job]
                due_time = instance.due_dates[job] / (global_max_due_date + EPS)
                job_id_norm = job / max(1, global_max_jobs - 1)
                for op in range(ops_in_job):
                    node_idx = instance.op_index_map[(job, op)]
                    self.static_op_feats[node_idx, 0] = min_proc[node_idx] / (global_max_proc_time + EPS)
                    self.static_op_feats[node_idx, 1] = max_proc[node_idx] / (global_max_proc_time + EPS)
                    self.static_op_feats[node_idx, 2] = op / max(1, ops_in_job - 1)
                    self.static_op_feats[node_idx, 3] = (ops_in_job - op) / ops_in_job
                    self.static_op_feats[node_idx, 4] = due_time
                    self.static_op_feats[node_idx, 5] = job_id_norm
                    # column 6 (earliest start) is written by _get_state at every decision
        elif feature_mode == "augmented":
            # Shared enhancement of the graph baselines: position, due date, job id,
            # readiness (column 3) and remaining work (column 4).
            for job in range(instance.n_jobs):
                ops_in_job = instance.ops_per_job[job]
                due_time = instance.due_dates[job] / (global_max_due_date + EPS)
                job_id_norm = job / max(1, global_max_jobs - 1)
                nodes = [instance.op_index_map[(job, op)] for op in range(ops_in_job)]
                job_total = float(sum(float(min_proc[node]) for node in nodes)) or 1.0
                running = job_total
                for op, node_idx in enumerate(nodes):
                    self.static_op_feats[node_idx, 0] = op / max(1, ops_in_job - 1)
                    self.static_op_feats[node_idx, 1] = due_time
                    self.static_op_feats[node_idx, 2] = job_id_norm
                    self.static_op_feats[node_idx, 4] = running / job_total
                    running -= float(min_proc[node_idx])
        else:
            # Original shared environment: position, due date and job id.
            for job in range(instance.n_jobs):
                ops_in_job = instance.ops_per_job[job]
                due_time = instance.due_dates[job] / (global_max_due_date + EPS)
                job_id_norm = job / max(1, global_max_jobs - 1)
                for op in range(ops_in_job):
                    node_idx = instance.op_index_map[(job, op)]
                    self.static_op_feats[node_idx, 0] = op / max(1, ops_in_job - 1)
                    self.static_op_feats[node_idx, 1] = due_time
                    self.static_op_feats[node_idx, 2] = job_id_norm

        self.static_mac_feats = torch.zeros((self.global_max_machines, self.mac_feat_dim), dtype=torch.float32, device=DEVICE)
        for m in range(instance.n_machines):
            self.static_mac_feats[m, 2] = instance.capability_tensor[m]
            if feature_mode == "full":
                self.static_mac_feats[m, 3] = m / max(1, instance.n_machines - 1)

        # Shared empty tensors for the edge type this model does not use.
        self.empty_src = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.empty_dst = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.empty_feat = torch.empty((0, 1), dtype=torch.float32, device=DEVICE)

        self.reset()

    def reset(self):
        self.machine_available_time = torch.zeros(self.inst.n_machines, dtype=torch.float32, device=DEVICE)
        self.machine_load = torch.zeros(self.inst.n_machines, dtype=torch.float32, device=DEVICE)
        self.job_next_op = torch.zeros(self.inst.n_jobs, dtype=torch.long, device=DEVICE)
        self.job_completion_time = torch.zeros(self.inst.n_jobs, dtype=torch.float32, device=DEVICE)
        self.current_time = torch.tensor(0.0, device=DEVICE)

        self.op_finish_time = torch.zeros(self.total_op_nodes, dtype=torch.float32, device=DEVICE)
        self.op_scheduled = torch.zeros(self.total_op_nodes, dtype=torch.bool, device=DEVICE)
        # Number of scheduled operations of this episode.  Counting them in python keeps the
        # termination test below free of a host synchronisation.
        self.scheduled_ops = 0

        self.op_feats_t = self.static_op_feats.clone()
        self.mac_feats_t = self.static_mac_feats.clone()

        # Constants that the step uses over and over.  Rebuilding them in place costs a host to
        # device transfer (for the scalar) and an allocation plus a fill kernel (for the vectors)
        # in every step of every parallel environment.
        self.zero_scalar = torch.zeros((), dtype=torch.float32, device=DEVICE)
        self.all_job_ids = torch.arange(self.inst.n_jobs, device=DEVICE)
        self.seq_edge_feat_full = torch.ones((self.seq_edges_src.shape[0], 1), dtype=torch.float32, device=DEVICE)
        self.disj_edges_src = self.inst.disj_edges_src
        self.disj_edges_dst = self.inst.disj_edges_dst
        self.disj_edges_feat = self.inst.disj_edges_feat

        return self._get_state()

    def step(self, action):
        job, op, machine = action
        op_idx = self.job_op_to_idx[job.long(), op.long()]
        p_time = self.proc_time_matrix[op_idx, machine.long()]

        pre_idx = self.op_predecessor[op_idx]
        # The cached scalar keeps the select on the device: torch.tensor(0.0, device=DEVICE)
        # copies a host value to the device in every step of every parallel environment.
        job_ready = torch.where(pre_idx != -1, self.op_finish_time[pre_idx], self.zero_scalar)
        machine_ready = self.machine_available_time[machine.long()]
        start = torch.maximum(job_ready, machine_ready)
        end = start + p_time

        # The load-balance variation dLB_t is the difference between the standard deviation of
        # the machine loads before and after this assignment: the pre-assignment standard deviation
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
        self.scheduled_ops += 1

        # Immediate reward

        norm_p = p_time / self.global_max_proc_time_tensor
        avg_load = torch.mean(self.machine_load) + 1e-8
        norm_LB = delta_LB / avg_load

        due = self.due_dates_tensor[job.long()]
        t_c = self.current_time
        n_i = self.ops_per_job_tensor[job.long()]
        # Operation urgency: zero while the operation finishes before its due date and
        # proportional to the lateness once the due date is exceeded. Using the lateness
        u_ij = torch.clamp(end - due, min=0) / (self.global_max_due_date_tensor * n_i + EPS)

        # Per-operation normalisation of the local reward.  Every operation is scheduled exactly
        # once, so the unnormalised sum of the first term is a constant, and the other two
        reward = - (W1 * norm_p + W2 * norm_LB + W3 * u_ij) / self.total_op_nodes
        reward /= self.cfg.reward_scaling
        reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        # The counter tracks the number of assignments and the mask tracks which operations
        # were assigned; a feasible action assigns the operation it selects exactly once, so both
        done = self.scheduled_ops >= self.total_op_nodes
        if done:
            # Final global penalty
            c_max = self.current_time
            avg_load = torch.mean(self.machine_load)
            lb = torch.sqrt(torch.mean((self.machine_load - avg_load) ** 2))
            t_tardy = torch.sum(torch.clamp(self.job_completion_time - self.due_dates_tensor, min=0))

            # Terminal reward: the weighted objective of the finished schedule, normalised by
            # the size of the instance, i.e. by the same operation count as the local reward,
            global_penalty = (W1 * c_max / (self.global_max_proc_time_tensor * self.total_op_nodes)
                              + W2 * lb / (avg_load + EPS)
                              + W3 * t_tardy / (self.global_max_due_date_tensor * self.total_op_nodes))
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

        predecessors = self.op_predecessor[:self.inst.total_ops]
        ready = torch.where(predecessors >= 0,
                            self.op_finish_time[predecessors.clamp(min=0)],
                            self.zero_scalar)
        # The compact table writes the real earliest start: the larger of the predecessor’s
        # finish and the earliest free eligible machine.  The previous readiness column
        if self.feature_mode == "augmented":
            n_ops = self.inst.total_ops
            n_mac = self.inst.n_machines
            available = self.machine_available_time[:n_mac].unsqueeze(0)
            eligible = self.op_mac_mask[:n_ops, :n_mac] > 0.5
            masked = torch.where(eligible, available.expand(n_ops, -1),
                                 available.new_full((n_ops, n_mac), float("inf")))
            earliest = torch.maximum(ready, masked.min(dim=1).values)
            # Current time plus the average work per machine: a fixed global bound made the
            # column too small, while the previous current-time-only divisor saturated it at 1.
            horizon = (self.current_time
                       + self.global_max_proc_time_tensor * float(self.total_op_nodes)
                       / float(max(1, self.inst.n_machines)) + 1.0)
            self.op_feats_t[:n_ops, 3] = torch.clamp(earliest / horizon, 0.0, 1.0)
        elif self.feature_mode == "full":
            self.op_feats_t[:self.inst.total_ops, 6] = torch.clamp(
                ready - self.current_time, min=0.0) / self.global_max_proc_time_tensor

        unsched_mask = ~self.op_scheduled

        # The edge arrays of an instance never change, only their visibility does.  The state
        # therefore carries the static arrays plus one visibility flag per edge, and the packer
        seq_valid = unsched_mask[self.seq_edges_src] & unsched_mask[self.seq_edges_dst]
        alloc_valid = unsched_mask[self.alloc_edges_src]
        alloc_rev_valid = unsched_mask[self.alloc_rev_edges_dst]
        # Competition stays visible while both of its operations are still unscheduled: once one
        # of them is placed, the pair no longer competes for the machine and the manuscript removes
        if getattr(self.cfg, 'use_disjunctive_edges', True):
            disj_valid = unsched_mask[self.disj_edges_src] & unsched_mask[self.disj_edges_dst]
        else:
            disj_valid = torch.zeros_like(self.disj_edges_src, dtype=torch.bool)

        edges_t = {
            'seq': (self.seq_edges_src, self.seq_edges_dst),
            'disj': (self.disj_edges_src, self.disj_edges_dst),
            'op_mac': (self.alloc_edges_src, self.alloc_edges_dst),
            'mac_op': (self.alloc_rev_edges_src, self.alloc_rev_edges_dst)
        }
        edge_feats_t = {
            'seq': self.seq_edge_feat_full,
            'disj': self.disj_edges_feat,
            'op_mac': self.alloc_edges_feat,
            'mac_op': self.alloc_rev_edges_feat
        }
        edge_valid_t = {
            'seq': seq_valid,
            'disj': disj_valid,
            'op_mac': alloc_valid,
            'mac_op': alloc_rev_valid
        }

        op_mask = torch.zeros(self.global_max_ops, dtype=torch.float32, device=DEVICE)
        next_op = self.job_next_op
        max_ops_per_job = self.ops_per_job_tensor
        # Selecting the unfinished jobs with a boolean mask has an unknown length on the host and
        # therefore synchronises the device.  Clamping the per-job operation counter keeps the
        next_op_clamped = torch.minimum(next_op, max_ops_per_job - 1)
        avail_op_indices = self.job_op_to_idx[self.all_job_ids, next_op_clamped]
        avail_op_flag = (next_op < max_ops_per_job).to(op_mask.dtype)
        op_mask.index_put_((avail_op_indices,), avail_op_flag)

        return {
            # The operation features are written once per episode and only read afterwards, so the
            # trajectory can share them; the machine features are updated in place at every step and
            'op_feats': self.op_feats_t.clone(),
            'mac_feats': self.mac_feats_t.clone(),
            'edges': edges_t,
            'edge_feats': edge_feats_t,
            'edge_valid': edge_valid_t,
            'op_mask': op_mask,
        }




# 7. Batch environment manager

class BatchEnv:
    def __init__(self, envs: List['HeteroGraphEnv']):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        next_states = []
        rewards = []
        dones = []
        for env, act in zip(self.envs, actions):
            ns, r, d = env.step(act)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
        return next_states, rewards, dones


# 8. Heterogeneous graph neural network

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
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.d_k = out_dim // num_heads
        self.out_dim = out_dim

        self.W_op = nn.Linear(in_dim_op, out_dim)
        self.W_mac = nn.Linear(in_dim_mac, out_dim)
        orthogonal_init(self.W_op, gain=nn.init.calculate_gain('relu'))
        orthogonal_init(self.W_mac, gain=nn.init.calculate_gain('relu'))

        self.att_seq = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))
        self.att_op_mac = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))
        self.att_mac_op = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))
        # Disjunctive (machine competition) message channel of the manuscript.  The environment
        # builds the competing operation pairs and prunes them per machine (see
        self.att_disj = nn.Parameter(torch.randn(num_heads, 2*self.d_k + 1))

        self.leaky = nn.LeakyReLU(0.2)
        self.max_edges = cfg.max_edges_per_type

    def forward(self, h_op, h_mac, edges, edge_feats):
        h_op = h_op.float()
        h_mac = h_mac.float()
        edge_feats_proc = {}
        for key, feat in edge_feats.items():
            if feat.numel() > 0:
                edge_feats_proc[key] = feat.float()
            else:
                edge_feats_proc[key] = feat

        device = h_op.device
        dtype = h_op.dtype

        z_op = self.W_op(h_op).view(-1, self.heads, self.d_k)
        z_mac = self.W_mac(h_mac).view(-1, self.heads, self.d_k)

        out_op = torch.zeros_like(z_op)
        out_mac = torch.zeros_like(z_mac)

        def apply_attn(src_emb, dst_emb, src_idx, dst_idx, feat_e, att_param, out_buffer, gate=False):
            if src_idx.numel() == 0:
                return
            if src_idx.numel() > self.max_edges:
                return
            dtype_out = out_buffer.dtype
            h_s = src_emb[src_idx].to(dtype_out)
            h_d = dst_emb[dst_idx].to(dtype_out)
            f_e = feat_e.to(dtype_out).unsqueeze(1).repeat(1, self.heads, 1)
            att_param = att_param.to(dtype_out)
            concat = torch.cat([h_s, h_d, f_e], dim=-1)
            scores = (concat * att_param).sum(dim=-1)
            scores = self.leaky(scores)
            if gate:
                # Gate of the conjunctive channel.  The exponential of a raw attention score
                # overflows to +inf once the score passes 88, and a softmax over a single
                weight = torch.sigmoid(scores).to(dtype_out)
            else:
                # The exponential of a raw attention score overflows to +inf once the score
                # passes 88: the denominator becomes inf as well, the normalised weight is
                scores = scores - scores.detach().max()
                alpha = torch.exp(scores).to(dtype_out)

                denom = torch.zeros(out_buffer.shape[0], self.heads, device=device, dtype=dtype_out)
                denom.index_add_(0, dst_idx, alpha)
                denom = denom[dst_idx] + torch.tensor(EPS, dtype=dtype_out, device=device)
                weight = alpha / denom
            weighted = h_s * weight.unsqueeze(-1)
            out_buffer.index_add_(0, dst_idx, weighted)

        if edges['seq'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['seq'][0], edges['seq'][1], edge_feats_proc['seq'], self.att_seq,
                       out_op, gate=True)
        if edges['op_mac'][0].numel() > 0:
            apply_attn(z_op, z_mac, edges['op_mac'][0], edges['op_mac'][1], edge_feats_proc['op_mac'], self.att_op_mac, out_mac)
        if edges['mac_op'][0].numel() > 0:
            apply_attn(z_mac, z_op, edges['mac_op'][0], edges['mac_op'][1], edge_feats_proc['mac_op'], self.att_mac_op, out_op)
        # The disjunctive channel is applied exactly as the three channels above, over the
        # competing pairs the environment reports.  Without it the encoder of this method read
        if edges.get('disj') is not None and edges['disj'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['disj'][0], edges['disj'][1], edge_feats_proc['disj'],
                       self.att_disj, out_op)

        res_op = (out_op.flatten(1) + self.W_op(h_op))
        res_mac = (out_mac.flatten(1) + self.W_mac(h_mac))
        return F.elu(res_op), F.elu(res_mac)

class HeteroGAT(nn.Module):
    """Heterogeneous encoder of the proposed method, three message-passing hops deep.

    One hop carries the operation features across the operation-machine edges, the second one
    brings the machine features back to the operations.  The third hop is the first that
    propagates the availability of a machine through an operation into the *other* machines that
    share operations with it, which is the information a load-balancing decision needs; the
    two-hop encoder of every earlier wave has to infer it from the pooled graph embedding alone.
    """
    def __init__(self, dim_op, dim_mac, hidden_dim, out_dim):
        super().__init__()
        self.l1 = HeteroGATLayer(dim_op, dim_mac, hidden_dim)
        self.l2 = HeteroGATLayer(hidden_dim, hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, state_dict):
        op_feats = state_dict['op_feats']
        mac_feats = state_dict['mac_feats']
        edges = state_dict['edges']
        edge_feats = state_dict['edge_feats']

        h_op1, h_mac1 = self.l1(op_feats, mac_feats, edges, edge_feats)
        h_op2, h_mac2 = self.l2(h_op1, h_mac1, edges, edge_feats)

        return h_op2, h_mac2


# 9. Hierarchical policy networks (improved numerical stability)

class UpperPolicy(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 2, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1)
        )

    def forward(self, g_emb, op_embs, mask=None):
        g_emb = g_emb.float()
        op_embs = op_embs.float()
        if g_emb.dim() == 1:
            N = op_embs.shape[0]
            g_rep = g_emb.unsqueeze(0).expand(N, -1)
            cat = torch.cat([g_rep, op_embs], dim=1)
            logits = self.net(cat).squeeze(-1)
            fill_value = -1e9
            # Clip the raw logits BEFORE masking: clamping after masked_fill would
            # lift the fill value (-1e9) up to -cfg.logits_clip and therefore let
            # infeasible actions keep probability mass in the softmax output.
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                # Use -1e9 instead of -inf to avoid NaN in softmax
                logits = logits.masked_fill(mask == 0, fill_value)
            # A row without a feasible action falls back to the uniform distribution; the test stays
            # on the device instead of branching on a host read.
            all_masked = (logits == fill_value).all()
            probs = F.softmax(logits, dim=0)
            if mask is not None:
                # Infeasible actions must keep exactly zero probability,
                # otherwise the epsilon floor below leaks mass onto them.
                probs = probs * (mask > 0).float()
            probs = probs / probs.sum().clamp(min=EPS)
            probs = torch.where(all_masked, torch.full_like(probs, 1.0 / N), probs)
        else:
            batch_size, N, _ = op_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, N, -1)
            cat = torch.cat([g_rep, op_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            fill_value = -1e9
            # Clip before masking (same reason as the single-sample branch)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, fill_value)
            # A row without a feasible action falls back to the uniform distribution.  The test is
            # evaluated for the whole batch at once: a per-row branch needs a host read per
            # environment, and such a read waits for the queue of the shared device.
            all_masked = (logits == fill_value).all(dim=1, keepdim=True)
            row_probs = F.softmax(logits, dim=1)
            if mask is not None:
                row_probs = row_probs * (mask > 0).float()
            row_probs = row_probs / row_probs.sum(dim=1, keepdim=True).clamp(min=EPS)
            probs = torch.where(all_masked, torch.full_like(row_probs, 1.0 / N), row_probs)
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
        g_emb = g_emb.float()
        selected_op_emb = selected_op_emb.float()
        mac_embs = mac_embs.float()
        if g_emb.dim() == 1:
            M = mac_embs.shape[0]
            g_rep = g_emb.unsqueeze(0).expand(M, -1)
            op_rep = selected_op_emb.unsqueeze(0).expand(M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=1)
            logits = self.net(cat).squeeze(-1)
            if heuristic is not None:
                # Fixed-preference heuristic term of the manuscript's lower policy (Eq. 10): the
                # machine that processes the selected operation fastest, carries the least load and
                logits = logits + heuristic
            fill_value = -1e9
            # Clip the raw logits BEFORE masking (see the note in UpperPolicy)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, fill_value)
            all_masked = (logits == fill_value).all()
            probs = F.softmax(logits, dim=0)
            if mask is not None:
                probs = probs * (mask > 0).float()
            probs = probs / probs.sum().clamp(min=EPS)
            probs = torch.where(all_masked, torch.full_like(probs, 1.0 / M), probs)
        else:
            batch_size, M, _ = mac_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, M, -1)
            op_rep = selected_op_emb.unsqueeze(1).expand(-1, M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            if heuristic is not None:
                # Fixed-preference heuristic term, see the single-sample branch above.
                logits = logits + heuristic
            fill_value = -1e9
            # Clip before masking (see the note in UpperPolicy)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, fill_value)
            all_masked = (logits == fill_value).all(dim=1, keepdim=True)
            row_probs = F.softmax(logits, dim=1)
            if mask is not None:
                row_probs = row_probs * (mask > 0).float()
            row_probs = row_probs / row_probs.sum(dim=1, keepdim=True).clamp(min=EPS)
            probs = torch.where(all_masked, torch.full_like(row_probs, 1.0 / M), row_probs)
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


def padded_op_mac_mask(inst, n_op: int, n_mac: int):
    """Machine mask of every operation in the padded layout of the packed batch.

    The packer lays the states out as consecutive blocks of n_op operation nodes, so the mask of a
    whole trajectory is one gather instead of one row-wise copy per state.  The tensor only depends
    on the padding of the instance and is therefore cached on it.
    """
    mask = getattr(inst, '_op_mac_mask_padded', None)
    if mask is None or mask.shape[0] != n_op or mask.shape[1] != n_mac:
        mask = torch.zeros((n_op, n_mac), dtype=torch.float32, device=DEVICE)
        mask[:inst.total_ops, :inst.n_machines] = inst.op_mac_mask
        inst._op_mac_mask_padded = mask
    return mask


# 10. Batch state packing

def batch_states(states: List[Dict], n_op: int, n_mac: int):
    """Pack the per-environment states into the batched graph layout of the encoders.

    A state carries the static edge arrays of its instance and one visibility flag per edge, so the
    visible edges of a whole batch are extracted with three device reads (one per edge type) instead
    of three reads per state and type.  The extracted edges, and their order, are the same as in a
    per-state selection.
    """
    batch_size = len(states)
    op_offsets = torch.arange(batch_size, device=DEVICE) * n_op
    mac_offsets = torch.arange(batch_size, device=DEVICE) * n_mac

    edge_types = ['seq', 'disj', 'op_mac', 'mac_op']
    batch_edges = {}
    batch_edge_feats = {}
    empty_edges = (torch.empty(0, dtype=torch.long, device=DEVICE),
                   torch.empty(0, dtype=torch.long, device=DEVICE))
    empty_feat = torch.empty((0, 1), dtype=torch.float32, device=DEVICE)

    # Every channel is packed by the same code.  The disjunctive channel used to be replaced by an
    # empty list at this point, which kept the machine-competition edges out of the packed graph of
    for k in edge_types:
        lengths = [s['edges'][k][0].numel() for s in states]
        if not any(lengths):
            batch_edges[k] = empty_edges
            batch_edge_feats[k] = empty_feat
            continue

        src_all = torch.cat([s['edges'][k][0] for s in states])
        dst_all = torch.cat([s['edges'][k][1] for s in states])
        feat_all = torch.cat([s['edge_feats'][k] for s in states], dim=0)
        valid_all = torch.cat([s['edge_valid'][k] for s in states])
        owner = torch.repeat_interleave(
            torch.arange(batch_size, device=DEVICE),
            torch.tensor(lengths, dtype=torch.long, device=DEVICE))

        # The only host read of this function: the visible edges of every state, in state order.
        keep = valid_all.nonzero().squeeze(1)
        owners = owner[keep]
        if k == 'seq':
            src_final = src_all[keep] + op_offsets[owners]
            dst_final = dst_all[keep] + op_offsets[owners]
        elif k == 'op_mac':
            src_final = src_all[keep] + op_offsets[owners]
            dst_final = dst_all[keep] + mac_offsets[owners]
        elif k == 'disj':
            # Competition runs between two operations, so both of its ends are operation nodes.
            # Without this branch the disjunctive edges would take the machine-to-operation branch
            # below and their source would be read out of the machine block of the batch.
            src_final = src_all[keep] + op_offsets[owners]
            dst_final = dst_all[keep] + op_offsets[owners]
        else:
            src_final = src_all[keep] + mac_offsets[owners]
            dst_final = dst_all[keep] + op_offsets[owners]
        feat_final = feat_all[keep]

        num_edges = src_final.shape[0]
        if num_edges > cfg.max_edges_per_type:
            perm = torch.randperm(num_edges, device=DEVICE)[:cfg.max_edges_per_type]
            src_final = src_final[perm]
            dst_final = dst_final[perm]
            feat_final = feat_final[perm]

        batch_edges[k] = (src_final, dst_final)
        batch_edge_feats[k] = feat_final

    batch_state_dict = {
        'op_feats': torch.cat([s['op_feats'] for s in states], dim=0),
        'mac_feats': torch.cat([s['mac_feats'] for s in states], dim=0),
        'edges': batch_edges,
        'edge_feats': batch_edge_feats,
    }

    state_slices = torch.zeros((batch_size, 4), dtype=torch.long, device=DEVICE)
    for i in range(batch_size):
        state_slices[i, 0] = i * n_op
        state_slices[i, 1] = (i + 1) * n_op
        state_slices[i, 2] = i * n_mac
        state_slices[i, 3] = (i + 1) * n_mac

    batch_op_mask = torch.cat([s['op_mask'] for s in states], dim=0)

    return batch_state_dict, state_slices, batch_op_mask




# 11. PPO Agent

class HierarchicalPPOAgent:
    def __init__(self, inst, gat, upper, lower, critic_u, critic_l, cfg, state_norm=None):
        self.inst = inst
        self.gat = gat
        self.upper = upper
        self.lower = lower
        self.critic_u = critic_u
        self.critic_l = critic_l
        self.cfg = cfg
        self.state_norm = state_norm

        self.gamma = cfg.gamma
        self.lam = cfg.gae_lambda
        self.clip = 0.2
        self.beta = cfg.beta_start
        self.beta_lr = cfg.beta_lr
        self.value_coef = cfg.value_coef
        self.entropy_coef = cfg.entropy_coef
        self.value_clip = cfg.value_clip
        self.max_grad_norm = cfg.max_grad_norm

        self.params = list(gat.parameters()) + list(upper.parameters()) + \
                      list(lower.parameters()) + list(critic_u.parameters()) + list(critic_l.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay)

        self.amp_enabled = cfg.amp_enabled
        self.scaler = torch.cuda.amp.GradScaler() if self.amp_enabled else None

        self.kl_ema = None
        self.kl_alpha = 0.9

        # Diagnostics of the sampler.  The full probability vector of an invalid row used
        # to be printed, and because the sampler retries at every step a diverged run grew
        self.invalid_report_count = 0
        # Optimizer steps that were rolled back because they left a non-finite weight.
        self.nan_rollbacks = 0

        self.stats_window = cfg.stats_window
        self.recent_policy_loss_u = deque(maxlen=self.stats_window)
        self.recent_policy_loss_l = deque(maxlen=self.stats_window)
        self.recent_value_loss = deque(maxlen=self.stats_window)
        self.recent_kl = deque(maxlen=self.stats_window)
        self.recent_beta = deque(maxlen=self.stats_window)
        self.recent_advantage = deque(maxlen=self.stats_window)
        self.recent_return = deque(maxlen=self.stats_window)

        self.update_counter = 0
        self.episode_counter = 0
        self.best_norm_reward = -float('inf')
        # Validation bookkeeping: the validation curve, the reference objective of every
        # validation instance and the best value seen so far survive an interruption.
        self.val_curve = []
        self.val_reference = []
        self.best_val = float('inf')
        self.converged = False

    def save(self, path):
        _atomic_path = path + ".tmp"
        torch.save({
            'gat': self.gat.state_dict(),
            'upper': self.upper.state_dict(),
            'lower': self.lower.state_dict(),
            'critic_u': self.critic_u.state_dict(),
            'critic_l': self.critic_l.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'beta': self.beta,
            'recipe': RECIPE_TAG,
            'entropy_coef': self.entropy_coef,
            'episode_counter': self.episode_counter,
            'best_norm_reward': self.best_norm_reward,
            'val_curve': list(self.val_curve),
            'val_reference': list(self.val_reference),
            'best_val': self.best_val,
            'converged': self.converged,
            # The normaliser is not part of the optimizer, so it is saved explicitly:
            # a resumed run would otherwise re-normalise with fresh statistics.
            'state_norm_mean': None if self.state_norm is None else self.state_norm.mean,
            'state_norm_var': None if self.state_norm is None else self.state_norm.var,
            'state_norm_count': None if self.state_norm is None else self.state_norm.count
        }, _atomic_path)
        os.replace(_atomic_path, path)
        print(f"Model saved to {path}")

    def _incompatible(self, checkpoint):
        """Return a description of the first tensor of ``checkpoint`` that does not fit this agent.

        A checkpoint of another capacity must not be loaded halfway (see :meth:`load`), so the
        shapes are compared before anything is written into the modules.  The empty string means
        that every tensor of every module fits the configured network.
        """
        for name, module in (('gat', self.gat), ('upper', self.upper), ('lower', self.lower),
                             ('critic_u', self.critic_u), ('critic_l', self.critic_l)):
            stored = checkpoint.get(name)
            if stored is None:
                return "%s is absent from the checkpoint" % name
            current = module.state_dict()
            if set(stored) != set(current):
                missing = sorted(set(current) - set(stored))[:1]
                unexpected = sorted(set(stored) - set(current))[:1]
                return "%s: keys differ (missing %s, unexpected %s)" % (name, missing, unexpected)
            for key, tensor in stored.items():
                if tuple(tensor.shape) != tuple(current[key].shape):
                    return "%s.%s: %s instead of %s" % (name, key, tuple(tensor.shape),
                                                       tuple(current[key].shape))
        return ""

    def load(self, path):
        if os.path.exists(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(path, map_location=DEVICE)
            # A checkpoint of another capacity must not be loaded halfway.  The wave that changes
            # the width or the depth of the encoder writes tensors whose shapes differ from the ones
            mismatched = self._incompatible(checkpoint)
            if mismatched:
                print("Checkpoint %s does not fit the configured capacity (%s); starting from scratch."
                      % (path, mismatched))
                return False
            self.gat.load_state_dict(checkpoint['gat'])
            self.upper.load_state_dict(checkpoint['upper'])
            self.lower.load_state_dict(checkpoint['lower'])
            self.critic_u.load_state_dict(checkpoint['critic_u'])
            self.critic_l.load_state_dict(checkpoint['critic_l'])

            self.gat = self.gat.to(DEVICE).float()
            self.upper = self.upper.to(DEVICE).float()
            self.lower = self.lower.to(DEVICE).float()
            self.critic_u = self.critic_u.to(DEVICE).float()
            self.critic_l = self.critic_l.to(DEVICE).float()

            self.params = list(self.gat.parameters()) + list(self.upper.parameters()) + \
                          list(self.lower.parameters()) + list(self.critic_u.parameters()) + list(self.critic_l.parameters())
            self.opt = torch.optim.Adam(self.params, lr=self.cfg.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay)

            # Restore the optimiser state when the checkpoint carries it (the
            # older checkpoints hold the weights only).
            if checkpoint.get('optimizer') is not None:
                try:
                    self.opt.load_state_dict(checkpoint['optimizer'])
                except Exception as exc:
                    print(f"Optimiser state not restored ({exc}), cold optimiser used.")
            # The schedule and the KL penalty are hyper-parameters of a recipe, not state.  A run that
            # resumed from a checkpoint of another recipe inherited beta=0.01 and entropy_coef=0.005
            if checkpoint.get('recipe') == RECIPE_TAG:
                self.beta = float(checkpoint.get('beta', self.beta))
                self.entropy_coef = float(checkpoint.get('entropy_coef', self.entropy_coef))
            else:
                # The stored optimiser carries the learning rate of that recipe as well, so the peak of
                # the configured schedule is restored on top of it; the scheduler is rebuilt with the
                # configured decay so a checkpoint of another recipe cannot carry its own in.
                for group in self.opt.param_groups:
                    group['lr'] = self.cfg.lr
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.opt, step_size=self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay)
                print("Checkpoint recipe %r differs from %r, keeping the configured recipe."
                      % (checkpoint.get('recipe'), RECIPE_TAG))
            self.episode_counter = checkpoint.get('episode_counter', 0)
            # Older checkpoints carry no record; -inf means that the next episode becomes the best.
            self.best_norm_reward = checkpoint.get('best_norm_reward', -float('inf'))
            self.val_curve = list(checkpoint.get('val_curve', []) or [])
            self.val_reference = list(checkpoint.get('val_reference', []) or [])
            self.best_val = float(checkpoint.get('best_val', float('inf')))
            self.converged = bool(checkpoint.get('converged', False))
            if self.state_norm is not None and checkpoint.get('state_norm_mean') is not None:
                self.state_norm.mean = checkpoint['state_norm_mean'].to(DEVICE).float()
                self.state_norm.var = checkpoint['state_norm_var'].to(DEVICE).float()
                self.state_norm.count = float(checkpoint.get('state_norm_count',
                                                           self.state_norm.count))

            print(f"Model loaded from {path} and converted to float32, beta={self.beta:.4f}, entropy_coef={self.entropy_coef:.4f}")
            return True
        else:
            print(f"Model file {path} not found, starting training from scratch.")
            return False

    def get_action_batch(self, states: List[Dict], insts: List[MOFJSPInstance], deterministic=False):
        batch_size = len(states)
        n_op_global = states[0]['op_feats'].shape[0]
        n_mac_global = states[0]['mac_feats'].shape[0]

        # Early detection of states with all-zero op_mask (finished environments).  The test is
        # vectorised into a single host read: one `.item()` per state drained the device queue
        # once per environment on every step, and several runs share the same GPU.
        op_mask_all = torch.stack([s['op_mask'] for s in states], dim=0)
        invalid_mask = op_mask_all.sum(dim=1) <= 0
        invalid_indices = torch.nonzero(invalid_mask).squeeze(1).tolist()

        if invalid_indices:
            actions = [None] * batch_size
            probs_list = [None] * batch_size
            indices_list = [None] * batch_size
            for i in invalid_indices:
                # Assign default action (0,0,0) and skip network computation
                actions[i] = (0, 0, 0)
                probs_list[i] = (1.0, 1.0)
                indices_list[i] = (0, 0)
                if self.invalid_report_count < 5:
                    print(f"Warning: State {i} has all-zero op_mask, assigning default action (0,0,0)")
                    self.invalid_report_count += 1
            valid_indices = [i for i in range(batch_size) if i not in invalid_indices]
            if valid_indices:
                valid_states = [states[i] for i in valid_indices]
                valid_insts = [insts[i] for i in valid_indices]
                valid_actions, valid_probs, valid_indices_list = self.get_action_batch(valid_states, valid_insts, deterministic)
                for idx, (a, p, il) in zip(valid_indices, valid_actions, zip(*valid_probs)):
                    actions[idx] = a
                    probs_list[idx] = p
                    indices_list[idx] = il
            return actions, probs_list, indices_list

        # Normal flow
        batch_state_dict, state_slices, batch_op_mask = batch_states(states, n_op_global, n_mac_global)

        with torch.no_grad():
            h_op_all, h_mac_all = self.gat(batch_state_dict)

            g_embs = []
            batch_h_op = []
            batch_h_mac = []
            batch_op_masks = []
            for i in range(batch_size):
                op_start, op_end, mac_start, mac_end = state_slices[i]
                h_op = h_op_all[op_start:op_end]
                h_mac = h_mac_all[mac_start:mac_end]
                g_emb = h_op.mean(dim=0)
                if self.state_norm is not None:
                    self.state_norm.update(g_emb.detach())
                    g_emb = self.state_norm.normalize(g_emb)
                g_embs.append(g_emb)
                batch_h_op.append(h_op)
                batch_h_mac.append(h_mac)
                batch_op_masks.append(states[i]['op_mask'])

            g_embs = torch.stack(g_embs, dim=0)
            batch_h_op = torch.stack(batch_h_op, dim=0)
            batch_h_mac = torch.stack(batch_h_mac, dim=0)
            batch_op_masks = torch.stack(batch_op_masks, dim=0)

            u_probs_batch = self.upper(g_embs, batch_h_op, batch_op_masks)

            # Ensure each row of probabilities is valid.  One vectorised test replaces the
            # per-row host reads, which stalled the step against the other runs on the device.
            row_sums_u = u_probs_batch.sum(dim=1)
            invalid_rows_u = (~torch.isfinite(row_sums_u)) | (row_sums_u < EPS)
            if bool(invalid_rows_u.any().item()):
                if self.invalid_report_count < 5:
                    rows = torch.nonzero(invalid_rows_u).squeeze(1).tolist()
                    print(f"Critical error: Upper probability distribution row(s) {rows[:8]} are "
                          f"invalid, row sums {row_sums_u[invalid_rows_u][:8].tolist()}")
                    self.invalid_report_count += 1
                raise RuntimeError("Invalid probability distribution, training halted")

            if deterministic:
                op_indices = torch.argmax(u_probs_batch, dim=1)
            else:
                dist = torch.distributions.Categorical(u_probs_batch)
                op_indices = dist.sample()

            selected_op_embs = batch_h_op[torch.arange(batch_size), op_indices]

            batch_mac_masks_op = torch.zeros((batch_size, n_mac_global), device=DEVICE)
            for i in range(batch_size):
                inst = insts[i]
                op_idx_local = op_indices[i]
                # The padded rows of the global graph carry no machine mask.  Clamping the
                # index and scaling by the validity flag keeps the guard on the device: the
                # host-side comparison stalled the queue once per environment and step.
                valid = (op_idx_local < inst.total_ops).to(batch_mac_masks_op.dtype)
                safe_idx = op_idx_local.clamp(max=inst.total_ops - 1)
                batch_mac_masks_op[i, :inst.n_machines] = inst.op_mac_mask[safe_idx] * valid

            # Fixed-preference heuristic term of the lower policy (Eq. 10).  The processing times
            # are the static table of the instance, the load and the availability of a machine are
            lower_mac_feats = batch_state_dict['mac_feats'].reshape(batch_size, n_mac_global, -1)
            lower_pt_rows = torch.zeros((batch_size, n_mac_global), device=DEVICE)
            for i in range(batch_size):
                inst = insts[i]
                op_idx_local = op_indices[i]
                valid = (op_idx_local < inst.total_ops).to(lower_pt_rows.dtype)
                safe_idx = op_idx_local.clamp(max=inst.total_ops - 1)
                lower_pt_rows[i, :inst.n_machines] = inst.pt_rows_norm[safe_idx] * valid
            lower_heuristic = -(W1 * lower_pt_rows + W2 * lower_mac_feats[:, :, 0]
                                + W3 * lower_mac_feats[:, :, 1])

            l_probs_batch = self.lower(g_embs, selected_op_embs, batch_h_mac, batch_mac_masks_op,
                                      lower_heuristic)

            row_sums_l = l_probs_batch.sum(dim=1)
            invalid_rows_l = (~torch.isfinite(row_sums_l)) | (row_sums_l < EPS)
            if bool(invalid_rows_l.any().item()):
                if self.invalid_report_count < 5:
                    rows = torch.nonzero(invalid_rows_l).squeeze(1).tolist()
                    print(f"Critical error: Lower probability distribution row(s) {rows[:8]} are "
                          f"invalid, row sums {row_sums_l[invalid_rows_l][:8].tolist()}")
                    self.invalid_report_count += 1
                raise RuntimeError("Invalid probability distribution, training halted")

            if deterministic:
                mac_indices = torch.argmax(l_probs_batch, dim=1)
            else:
                dist_l = torch.distributions.Categorical(l_probs_batch)
                mac_indices = dist_l.sample()

            actions = []
            probs_list = []
            indices_list = []
            for i in range(batch_size):
                job = insts[i].op_idx_to_job[op_indices[i]]
                op = insts[i].op_idx_to_op[op_indices[i]]
                machine = mac_indices[i]
                actions.append((job, op, machine))
                probs_list.append((u_probs_batch[i, op_indices[i]], l_probs_batch[i, mac_indices[i]]))
                indices_list.append((op_indices[i], mac_indices[i]))

        return actions, probs_list, indices_list

    def update(self, trajectory, inst):
        T = len(trajectory)
        if T == 0:
            return None

        states = [t['state'] for t in trajectory]
        rewards = torch.stack([t['reward'] for t in trajectory]).to(DEVICE)
        old_u_probs = torch.stack([t['old_probs'][0] for t in trajectory]).detach()
        old_l_probs = torch.stack([t['old_probs'][1] for t in trajectory]).detach()
        op_indices = torch.stack([t['indices'][0] for t in trajectory])
        mac_indices = torch.stack([t['indices'][1] for t in trajectory])

        n_op_global = states[0]['op_feats'].shape[0]
        n_mac_global = states[0]['mac_feats'].shape[0]

        # States per update step.  The update keeps a whole trajectory on the device, so the chunk
        # size only decides how many kernel launches one pass costs; the device of this run holds
        # a single training now, so the chunks are doubled.
        chunk_size = 32

        # The packer keeps one block of n_op_global operation nodes and one block of n_mac_global
        # machine nodes per state, so the packed tensors are plain (T, nodes, dim) views.  The packed
        batch_state_dict, _, batch_op_mask = batch_states(states, n_op_global, n_mac_global)
        batch_op_mask_view = batch_op_mask.view(T, n_op_global)

        # Advantage calculation
        with torch.no_grad():
            h_op_all, h_mac_all = self.gat(batch_state_dict)

            g_embs = h_op_all.reshape(T, n_op_global, -1).mean(dim=1)
            if self.state_norm is not None:
                g_embs = self.state_norm.normalize(g_embs)

            values = self.critic_u(g_embs).squeeze()
            # Generalized advantage estimation, in the form the rest of the family runs: the same
            # backwards recursion as HGNN, DAN and RESCHED, i.e. the implementation of the two
            next_values = torch.cat([values[1:], torch.zeros(1, device=DEVICE)])
            non_terminal = torch.cat([torch.ones(T - 1, device=DEVICE),
                                     torch.zeros(1, device=DEVICE)])
            deltas = rewards + self.gamma * next_values * non_terminal - values
            advantages = torch.empty_like(deltas)
            running = torch.zeros((), device=DEVICE)
            for step in reversed(range(T)):
                running = (deltas[step] + self.gamma * self.lam * non_terminal[step] * running)
                advantages[step] = running
            returns = values + advantages

            # Advantage normalization.  The degenerate case without spread divides by one, which
            # keeps the branch on the device instead of reading the deviation back to the host.
            adv_std = advantages.std()
            scale = torch.where(adv_std < 1e-8, torch.ones_like(adv_std), adv_std + EPS)
            advantages = (advantages - advantages.mean()) / scale
            advantages = torch.clamp(advantages, -10.0, 10.0)
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
            advantages = advantages.detach()
            returns = returns.detach()

            # The lower level is updated with the advantage of the upper level.  A wave that trained
            # a value head of the lower level as well (its own advantage for the machine it picks,

            # The packed edges are kept for the update pass.  The cache is not emptied either: a
            # full flush hands the blocks back to the driver, the allocator has to obtain them again
            del h_op_all, h_mac_all, g_embs, next_values, deltas

        # Chunk metrics stay on the device: reading each of them separately waits for the device
        # queue once per chunk and per metric.
        total_metrics = torch.zeros(4, device=DEVICE)
        valid_chunks = 0

        # A trajectory normally fits into a single chunk, and its packed edges are available, so the
        # update pass below reuses them instead of packing the same states a second time.
        single_pass = T <= chunk_size

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_len = chunk_end - chunk_start

            chunk_op_indices = op_indices[chunk_start:chunk_end]
            chunk_mac_indices = mac_indices[chunk_start:chunk_end]
            chunk_advantages = advantages[chunk_start:chunk_end]
            chunk_returns = returns[chunk_start:chunk_end]
            chunk_old_u_probs = old_u_probs[chunk_start:chunk_end]
            chunk_old_l_probs = old_l_probs[chunk_start:chunk_end]

            if single_pass:
                chunk_batch_state_dict = batch_state_dict
                batch_op_masks = batch_op_mask_view
            else:
                chunk_states = states[chunk_start:chunk_end]
                chunk_batch_state_dict, _, chunk_op_masks = batch_states(chunk_states, n_op_global, n_mac_global)
                batch_op_masks = chunk_op_masks.view(chunk_len, n_op_global)

            h_op_all, h_mac_all = self.gat(chunk_batch_state_dict)

            g_embs_chunk = h_op_all.reshape(chunk_len, n_op_global, -1).mean(dim=1)
            if self.state_norm is not None:
                g_embs_chunk = self.state_norm.normalize(g_embs_chunk)

            batch_h_op = h_op_all.reshape(chunk_len, n_op_global, -1)
            batch_h_mac = h_mac_all.reshape(chunk_len, n_mac_global, -1)

            curr_u_probs_batch = self.upper(g_embs_chunk, batch_h_op, batch_op_masks)
            curr_u_probs = curr_u_probs_batch[torch.arange(chunk_len), chunk_op_indices]
            curr_u_probs = torch.clamp(curr_u_probs, EPS, 1.0 - EPS)
            chunk_old_u_probs_clamped = torch.clamp(chunk_old_u_probs, EPS, 1.0 - EPS)

            # One gather from the padded mask of the instance replaces the row-wise copy, which
            # issued a handful of host calls per state of the trajectory.
            padded_mac_mask = padded_op_mac_mask(inst, n_op_global, n_mac_global)
            chunk_valid = (chunk_op_indices < inst.total_ops).to(padded_mac_mask.dtype)
            chunk_safe_indices = chunk_op_indices.clamp(max=inst.total_ops - 1)
            batch_mac_masks_op = padded_mac_mask[chunk_safe_indices] * chunk_valid.unsqueeze(1)

            # The same fixed-preference heuristic term as in the rollout (Eq. 10).  The update has
            # to rebuild it from the stored states, otherwise the distribution it scores the
            chunk_mac_feats = chunk_batch_state_dict['mac_feats'].reshape(chunk_len,
                                                                       n_mac_global, -1)
            # The processing time table of an instance spans its own machines, while the feature
            # tensor of a state is padded to the global machine count, so the rows are copied into
            # a padded block instead of being added to a block of a different width.
            chunk_pt_rows = torch.zeros((chunk_len, n_mac_global), device=DEVICE)
            chunk_pt_rows[:, :inst.n_machines] = (inst.pt_rows_norm[chunk_safe_indices]
                                                  * chunk_valid.unsqueeze(1))
            chunk_heuristic = -(W1 * chunk_pt_rows + W2 * chunk_mac_feats[:, :, 0]
                                + W3 * chunk_mac_feats[:, :, 1])

            selected_op_embs = batch_h_op[torch.arange(chunk_len), chunk_op_indices]
            curr_l_probs_batch = self.lower(g_embs_chunk, selected_op_embs, batch_h_mac,
                                           batch_mac_masks_op, chunk_heuristic)
            curr_l_probs = curr_l_probs_batch[torch.arange(chunk_len), chunk_mac_indices]
            curr_l_probs = torch.clamp(curr_l_probs, EPS, 1.0 - EPS)
            chunk_old_l_probs_clamped = torch.clamp(chunk_old_l_probs, EPS, 1.0 - EPS)

            ratio_u = curr_u_probs / (chunk_old_u_probs_clamped + EPS)
            ratio_l = curr_l_probs / (chunk_old_l_probs_clamped + EPS)

            ratio_u = torch.clamp(ratio_u, 0.1, 10.0)
            ratio_l = torch.clamp(ratio_l, 0.1, 10.0)

            surr1_u = ratio_u * chunk_advantages
            surr2_u = torch.clamp(ratio_u, 1 - self.clip, 1 + self.clip) * chunk_advantages
            policy_loss_u = -torch.min(surr1_u, surr2_u).mean()

            surr1_l = ratio_l * chunk_advantages
            surr2_l = torch.clamp(ratio_l, 1 - self.clip, 1 + self.clip) * chunk_advantages
            policy_loss_l = -torch.min(surr1_l, surr2_l).mean()

            # Standard KL divergence KL(old||new), ensuring numerical stability
            kl_u = (chunk_old_u_probs_clamped * (torch.log(chunk_old_u_probs_clamped + EPS) - torch.log(curr_u_probs + EPS))).mean()
            kl_l = (chunk_old_l_probs_clamped * (torch.log(chunk_old_l_probs_clamped + EPS) - torch.log(curr_l_probs + EPS))).mean()
            kl = kl_u + kl_l
            kl = torch.clamp(kl, min=0.0, max=10.0)

            v_pred = self.critic_u(g_embs_chunk.float()).squeeze()
            chunk_returns_f32 = chunk_returns.float()
            if v_pred.dim() == 0:
                v_pred = v_pred.unsqueeze(0)
            if chunk_returns_f32.dim() == 0:
                chunk_returns_f32 = chunk_returns_f32.unsqueeze(0)

            if self.value_clip is not None:
                old_v = values[chunk_start:chunk_end].detach().float()
                if old_v.dim() == 0:
                    old_v = old_v.unsqueeze(0)
                v_pred_clipped = old_v + torch.clamp(v_pred - old_v, -self.value_clip, self.value_clip)
                value_loss_unclipped = F.mse_loss(v_pred, chunk_returns_f32, reduction='none')
                value_loss_clipped = F.mse_loss(v_pred_clipped, chunk_returns_f32, reduction='none')
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = F.mse_loss(v_pred, chunk_returns_f32)


            # Policy entropy must be evaluated on the whole action distribution,
            # not on the -log p of the sampled action only
            entropy_u = -(curr_u_probs_batch * torch.log(curr_u_probs_batch + EPS)).sum(dim=-1).mean()
            entropy_l = -(curr_l_probs_batch * torch.log(curr_l_probs_batch + EPS)).sum(dim=-1).mean()
            entropy = entropy_u + entropy_l

            total_loss = (policy_loss_u + policy_loss_l + self.value_coef * value_loss
                          + self.beta * kl - self.entropy_coef * entropy)

            if not bool(torch.isfinite(total_loss).item()):
                print(f"Warning: Loss is NaN/Inf in chunk {chunk_start}-{chunk_end}, skipping this chunk.")
                del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
                del curr_u_probs_batch, curr_l_probs_batch, selected_op_embs
                del ratio_u, ratio_l, surr1_u, surr2_u, surr1_l, surr2_l
                del policy_loss_u, policy_loss_l, kl, v_pred, value_loss, total_loss
                torch.cuda.empty_cache()
                continue

            self.opt.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            if not bool(torch.isfinite(grad_norm).item()):
                print(f"Warning: Gradient norm is NaN/Inf, skipping this chunk update.")
                self.opt.zero_grad()
            else:
                # Divergence guard.  A finite loss does not exclude a step that turns a
                # weight into NaN or Inf: the checks above only reject the loss and the
                snapshot = [param.detach().clone() for param in self.params]
                self.opt.step()
                if not all(bool(torch.isfinite(param).all().item()) for param in self.params):
                    with torch.no_grad():
                        for param, saved in zip(self.params, snapshot):
                            param.copy_(saved)
                    self.opt.state.clear()
                    self.nan_rollbacks += 1
                    print(f"Warning: the optimizer step left a non-finite parameter, the "
                          f"update of chunk {chunk_start}-{chunk_end} was rolled back "
                          f"({self.nan_rollbacks} rollbacks so far).")
                del snapshot

            total_metrics += torch.stack([kl, policy_loss_u, policy_loss_l, value_loss]).detach() * chunk_len
            valid_chunks += 1

            del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
            del curr_u_probs_batch, curr_l_probs_batch, selected_op_embs
            del ratio_u, ratio_l, surr1_u, surr2_u, surr1_l, surr2_l
            del policy_loss_u, policy_loss_l, kl, v_pred, value_loss, total_loss

            # CUDA housekeeping only: the pipeline also runs on a CPU-only build,
            # where torch.cuda.synchronize() raises instead of being a no-op.
            self.update_counter += 1
            if self.update_counter % 200 == 0 and DEVICE.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        if valid_chunks == 0:
            print("Warning: All update chunks were skipped, no valid update performed.")
            return None

        avg_kl, avg_policy_loss_u, avg_policy_loss_l, avg_value_loss = (total_metrics / T).tolist()

        if self.kl_ema is None:
            self.kl_ema = avg_kl
        else:
            self.kl_ema = self.kl_alpha * self.kl_ema + (1 - self.kl_alpha) * avg_kl
        kl_error = self.kl_ema - self.cfg.kl_target
        self.beta *= (1 + self.beta_lr * kl_error)
        # The floor is a config value: a coefficient that decays to 0.01 leaves the trust region of the
        # recipe inactive, which is the state the plateaued wave was measured in.
        self.beta = np.clip(self.beta, float(getattr(self.cfg, 'beta_floor', 0.01)), 10.0)

        self.entropy_coef *= self.cfg.entropy_decay
        self.entropy_coef = max(self.entropy_coef, getattr(self.cfg, 'entropy_min', 0.0))

        self.recent_policy_loss_u.append(avg_policy_loss_u)
        self.recent_policy_loss_l.append(avg_policy_loss_l)
        self.recent_value_loss.append(avg_value_loss)
        self.recent_kl.append(avg_kl)
        self.recent_beta.append(self.beta)
        # One host read for both means instead of one per value.
        adv_mean, ret_mean = torch.stack([advantages.mean(), returns.mean()]).tolist()
        self.recent_advantage.append(adv_mean)
        self.recent_return.append(ret_mean)

        stats = {
            'policy_loss_u': avg_policy_loss_u,
            'policy_loss_l': avg_policy_loss_l,
            'value_loss': avg_value_loss,
            'kl': avg_kl,
            'beta': self.beta,
            'advantage': adv_mean,
            'return': ret_mean,
            'total_loss': (
                        avg_policy_loss_u + avg_policy_loss_l + self.value_coef * avg_value_loss + self.beta * avg_kl)
        }
        return stats


# 12. Plotting helper functions (new)

def plot_three_goals(cmax_list, lb_list, tardy_list, save_path):
    """Plot the three objective values across episodes (original values)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'green', 'red']
    titles = ['Cmax (Max Completion Time)', 'Load Balance (Std)', 'Total Tardiness']
    data_lists = [cmax_list, lb_list, tardy_list]
    for ax, data, title, color in zip(axes, data_lists, titles, colors):
        ax.plot(data, alpha=0.3, color=color, label='Raw')
        window = min(50, len(data))
        if window > 1:
            smooth = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), smooth, color=color, linewidth=2, label=f'MA({window})')
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if data:
            y_min, y_max = min(data), max(data)
            margin = 0.1 * (y_max - y_min) if y_max != y_min else 1.0
            ax.set_ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Three objective curves (original values) saved to {save_path}")

def plot_three_goals_normalized(cmax_list, lb_list, tardy_list, save_path):
    """Plot the three objective values across episodes (normalized to [0,1])"""
    # Compute Min-Max normalization factors for each objective
    def normalize(data):
        data = np.array(data)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val < 1e-8:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    cmax_norm = normalize(cmax_list)
    lb_norm = normalize(lb_list)
    tardy_norm = normalize(tardy_list)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'green', 'red']
    titles = ['Cmax (Normalized)', 'Load Balance (Normalized)', 'Total Tardiness (Normalized)']
    data_lists = [cmax_norm, lb_norm, tardy_norm]
    for ax, data, title, color in zip(axes, data_lists, titles, colors):
        ax.plot(data, alpha=0.3, color=color, label='Raw')
        window = min(50, len(data))
        if window > 1:
            smooth = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), smooth, color=color, linewidth=2, label=f'MA({window})')
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Normalized Value")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Three objective curves (normalized) saved to {save_path}")

def _set_curve_ylim(*series):
    """Y range of a curve panel that covers the raw series and its moving average.

    The range was taken from the raw series alone, so a transient spike of the raw data set the
    scale of the panel and compressed the moving average, which is an order of magnitude smaller
    after the spike, onto the axis: the smoothed line then reads as a constant.  A series that
    spans more than two orders of magnitude is drawn on a log axis, which is the only scale on
    which both the transient and the average are legible.
    """
    values = [float(value) for one in series for value in one if np.isfinite(value)]
    if not values:
        return
    low, high = min(values), max(values)
    if low > 0 and high / low > 100.0:
        plt.yscale('log')
        plt.ylim(low * 0.5, high * 2.0)
    else:
        margin = 0.1 * (high - low) if high != low else 0.01
        plt.ylim(low - margin, high + margin)


def plot_kl_divergence(kl_list, save_path):
    """Plot the KL divergence curve separately"""
    plt.figure(figsize=(10, 6))
    plt.plot(kl_list, alpha=0.3, color='orange', label='Raw')
    window = min(50, len(kl_list))
    if window > 1:
        smooth = np.convolve(kl_list, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(kl_list)), smooth, color='orange', linewidth=2, label=f'MA({window})')
    plt.title("KL Divergence")
    plt.xlabel("Episode")
    plt.ylabel("KL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _set_curve_ylim(kl_list, smooth if window > 1 else [])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"KL divergence curve saved to {save_path}")

def plot_loss(loss_list, save_path):
    """Plot the total loss curve separately"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, alpha=0.3, color='red', label='Raw')
    window = min(50, len(loss_list))
    if window > 1:
        smooth = np.convolve(loss_list, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(loss_list)), smooth, color='red', linewidth=2, label=f'MA({window})')
    plt.title("Total Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _set_curve_ylim(loss_list, smooth if window > 1 else [])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Total loss curve saved to {save_path}")


# 13. Main training function

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


# Rollouts of one validation instance: the first is deterministic and the others are sampled, and
# the instance enters the objective with its best of that set - the protocol the comparison
VALIDATION_ROLLOUTS = 4


def evaluate_validation(agent, validation, max_jobs, max_total_ops, max_machines,
                        max_proc_time, max_due_date, seed=0):
    """Rollouts of the validation instances; returns the best [Cmax, LB, Tardy] per instance.

    The objective is the same statistic the comparison reports (the best of one deterministic and
    fifty sampled rollouts), so the selected weights are the best of the run under the reported
    protocol and not the draw of a single greedy decode.
    """
    agent.gat.eval()
    agent.upper.eval()
    agent.lower.eval()
    objectives = []
    for inst in validation:
        best = None
        for run_index in range(VALIDATION_ROLLOUTS):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                                 max_proc_time, max_due_date, agent.cfg)
            state = env.reset()
            steps = 0
            done = False
            while not done and steps < agent.cfg.max_steps_per_episode:
                actions, _, _ = agent.get_action_batch([state], [inst],
                                                       deterministic=(run_index == 0))
                state, _, done = env.step(actions[0])
                steps += 1
            cmax = float(env.job_completion_time.max().item())
            lb = float(env.machine_load.std().item())
            tardy = float(torch.clamp(env.job_completion_time - env.due_dates_tensor,
                                      min=0).sum().item())
            candidate = [cmax, lb, tardy]
            if best is None or (W1 * cmax + W2 * lb + W3 * tardy) < (W1 * best[0]
                                                                   + W2 * best[1]
                                                                   + W3 * best[2]):
                best = candidate
        objectives.append(best)
    agent.gat.train()
    agent.upper.train()
    agent.lower.train()
    return objectives


def train():
    cfg.use_disjunctive_edges = True
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    latest_model_path = os.path.join(MODEL_DIR, cfg.resume_name)

    print(f"Loading all instances from directory {INSTANCE_DIR}...")
    instance_tuples, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = load_all_instances(INSTANCE_DIR, FILE_PATTERN)
    print(f"Loaded {len(instance_tuples)} instances")
    print(f"Global max jobs: {max_jobs}, max machines: {max_machines}, max total operations: {max_total_ops}, max processing time: {max_proc_time}, max due date: {max_due_date}, max machine capability: {max_capability}")

    # Build instance objects
    instance_list = []
    for jobs, caps, dues, fname, size in instance_tuples:
        inst = MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability)
        instance_list.append(inst)

    # Curriculum learning sampler
    curriculum_sampler = CurriculumSampler(instance_list, cfg)

    # The reported weights are selected on a fixed validation set, one that the curriculum
    # sampler cannot move: the sampler changes the instance distribution over the run, so the
    val_tuples = load_all_instances(INSTANCE_DIR, "mo_fjsp_*_val.txt")[0]
    pool_tuples = load_all_instances(INSTANCE_DIR, "mo_fjsp_*_train.txt")[0]
    validation_instances = []
    for jobs, caps, dues, fname, size in val_tuples:
        validation_instances.append(MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time,
                                                   max_capability))
    pool_instances = []
    for jobs, caps, dues, fname, size in pool_tuples:
        pool_instances.append(MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time,
                                                   max_capability))
    validation_set = build_validation_set(validation_instances, per_size=8)
    # Two instances of every size group of the training pool next to the eight held-out ones:
    # the wave that selected on the held-out split alone produced checkpoints that measured
    print(f"Validation set: {[inst.file_name for inst in validation_set]}")

    global_max_ops = max(inst.total_ops for inst in instance_list)
    global_max_machines = max(inst.n_machines for inst in instance_list)
    print(f"Global max operations: {global_max_ops}, global max machines: {global_max_machines}")

    # Final method keeps the original 3/3 table; the compact and full tables remain for the
    # feature-ablation record only.
    dim_op = 3
    dim_mac = 3
    gat = HeteroGAT(dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim).to(DEVICE)
    upper = UpperPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
    lower = LowerPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
    critic_u = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
    critic_l = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)

    state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=DEVICE) if cfg.use_state_norm else None

    dummy_inst = instance_list[0]
    agent = HierarchicalPPOAgent(dummy_inst, gat, upper, lower, critic_u, critic_l, cfg, state_norm)
    # Continue an interrupted run from its rolling checkpoint; the best model is
    # used only as the fallback, because it lags behind by every episode that
    # did not improve the best normalised return.
    resume_path = latest_model_path if os.path.exists(latest_model_path) else best_model_path
    agent.load(resume_path)
    # An early stop is the decision of the run, not an interruption of it: the validation
    # objective stopped improving, and more episodes would only be trained past the peak of the
    if agent.converged:
        print(f"The rolling checkpoint already holds a converged run "
              f"({agent.episode_counter} episodes); nothing to train.")
        return

    num_envs = cfg.num_envs
    envs = []
    env_insts = []
    for _ in range(num_envs):
        inst = curriculum_sampler.sample()
        env = HeteroGraphEnv(inst, max_jobs, global_max_ops, global_max_machines, max_proc_time, max_due_date, cfg)
        envs.append(env)
        env_insts.append(inst)

    batch_env = BatchEnv(envs)
    states = batch_env.reset()

    traj_buffers = [[] for _ in range(num_envs)]
    ep_rewards = [torch.tensor(0.0, device=DEVICE) for _ in range(num_envs)]
    step_counts = [0 for _ in range(num_envs)]

    # Only the missing episodes are trained: the counter stored in the rolling
    # checkpoint is restored so that a restart does not repeat the whole run.
    total_episodes = min(agent.episode_counter, cfg.n_episodes)
    converged = False
    normalized_rewards = []
    kl_episodes = []
    loss_episodes = []
    # Carried by the agent: a restart continues the record of the earlier episodes instead of
    # overwriting the best model with the first episode of the new run.
    best_norm_reward = agent.best_norm_reward
    reward_ema = None
    # Validation bookkeeping of the interrupted run: the curve continues and the reference of
    # every validation instance is kept, so the score stays comparable across a restart.
    val_curve = list(agent.val_curve)
    val_reference = list(agent.val_reference) or None
    best_val = float(agent.best_val)
    # The patience of the early stop survives a restart.  Without this rebuild every restart of
    # the queue opened a fresh patience window and the run was trained far past its peak - the
    stale_checks = 0
    running_best = float('inf')
    for _episode, _score in val_curve:
        if _score < running_best * (1.0 - cfg.val_tol):
            running_best = _score
            stale_checks = 0
        else:
            stale_checks += 1

    # New lists for three objectives
    cmax_list = []
    lb_list = []
    tardy_list = []

    pbar = tqdm(total=cfg.n_episodes, initial=total_episodes, desc="Training", unit="ep")
    # Reference of the validation objective.  It is the weighted objective of a deterministic
    # earliest-completion dispatch of the same instance (dispatch_reference.py) instead of the
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from dispatch_reference import reference_objectives
    val_reference = [weighted_objective(obj) for obj in reference_objectives(
        [inst.file_name for inst in validation_set], INSTANCE_DIR)]
    # A curve written by the earlier frame holds a ratio to the first check of that run and is
    # not comparable with this one, so the run starts a fresh curve instead of mixing the two.
    if val_curve and abs(val_curve[0][1] - 1.0) < 1e-12:
        print("The restored validation curve is in the old frame and is reset.")
        val_curve = []
        best_val = float('inf')
        stale_checks = 0

    log_interval = 100
    empty_cache_interval = 50
    # Episodes between two rolling checkpoints of an interrupted run.
    checkpoint_interval = max(1, int(getattr(cfg, 'checkpoint_interval', 100)))
    # Sampling failures that happened one after another without an episode in between.
    sampling_failures = 0

    while total_episodes < cfg.n_episodes and not converged:
        # Fixed curriculum scale: see Config.curriculum_episodes.
        progress = min(1.0, total_episodes / max(1, cfg.curriculum_episodes))
        curriculum_sampler.update_stage(progress)

        # Check for None states
        for i in range(num_envs):
            if states[i] is None:
                print(f"Detected state {i} is None, forcibly resetting environment")
                new_inst = curriculum_sampler.sample()
                envs[i] = HeteroGraphEnv(new_inst, max_jobs, global_max_ops, global_max_machines, max_proc_time, max_due_date, cfg)
                batch_env.envs[i] = envs[i]
                env_insts[i] = new_inst
                states[i] = envs[i].reset()
                traj_buffers[i] = []
                ep_rewards[i] = torch.tensor(0.0, device=DEVICE)
                step_counts[i] = 0

        try:
            actions, probs_list, indices_list = agent.get_action_batch(states, env_insts)
            next_states, rewards, dones = batch_env.step(actions)
        except Exception as e:
            sampling_failures += 1
            print(f"Exception occurred during sampling or step: {e}, resetting all environments and continuing")
            for j in range(num_envs):
                new_inst = curriculum_sampler.sample()
                envs[j] = HeteroGraphEnv(new_inst, max_jobs, global_max_ops, global_max_machines, max_proc_time, max_due_date, cfg)
                batch_env.envs[j] = envs[j]
                env_insts[j] = new_inst
                states[j] = envs[j].reset()
                traj_buffers[j] = []
                ep_rewards[j] = torch.tensor(0.0, device=DEVICE)
                step_counts[j] = 0
            if sampling_failures >= 20:
                # The environments are new but the policy produced no valid distribution
                # in twenty consecutive attempts, so the weights are the broken part.  The
                print("Restoring the rolling checkpoint after %d consecutive sampling failures."
                      % sampling_failures)
                agent.load(resume_path)
                sampling_failures = 0
                # A failure that survives several restores is a defect of the code or of the
                # configuration and not a corrupted policy, so retrying it only spins the loop
                agent.restore_cycles = getattr(agent, "restore_cycles", 0) + 1
                if agent.restore_cycles >= 3:
                    raise RuntimeError(
                        "sampling failed twenty consecutive times in %d restore cycles"
                        " without a single successful step: %s" % (agent.restore_cycles, e))
            continue
        sampling_failures = 0
        # A restore cycle counts only while no step succeeds: a single valid step tells the run
        # that the policy and the environment agree again, and the counter is cleared.
        agent.restore_cycles = 0

        for i in range(num_envs):
            traj_buffers[i].append({
                'state': states[i],
                'action': actions[i],
                'reward': rewards[i],
                'old_probs': probs_list[i],
                'indices': indices_list[i]
            })
            ep_rewards[i] += rewards[i]
            step_counts[i] += 1

            if dones[i]:
                env = envs[i]
                cmax = float(env.job_completion_time.max().cpu().numpy())
                lb = float(env.machine_load.std().cpu().numpy())
                tardy = float(torch.sum(torch.clamp(env.job_completion_time - env.due_dates_tensor, min=0)).cpu().numpy())
                cmax_list.append(cmax)
                lb_list.append(lb)
                tardy_list.append(tardy)

                ep_reward_val = ep_rewards[i].item()
                norm_reward = ep_reward_val / envs[i].inst.total_ops
                normalized_rewards.append(norm_reward)
                total_episodes += 1
                pbar.update(1)

                if traj_buffers[i]:
                    stats = agent.update(traj_buffers[i], env_insts[i])
                    if stats is not None:
                        kl_episodes.append(stats['kl'])
                        loss_episodes.append(stats['total_loss'])
                    else:
                        kl_episodes.append(0.0)
                        loss_episodes.append(0.0)
                else:
                    stats = None
                    kl_episodes.append(0.0)
                    loss_episodes.append(0.0)

                # The scheduler advances once per episode so that its own counter reaches the
                # step size after that many episodes.  Stepping it only on the multiples of the
                agent.scheduler.step()
                for param_group in agent.opt.param_groups:
                    param_group['lr'] = max(param_group['lr'], cfg.min_lr)

                if total_episodes % log_interval == 0 and stats is not None:
                    lr = agent.opt.param_groups[0]['lr']
                    print(f"\nEpisode {total_episodes}: LR={lr:.2e}, Beta={stats['beta']:.3f}, "
                          f"Adv={stats['advantage']:.4f}, KL={stats['kl']:.4f}, "
                          f"Entropy={agent.entropy_coef:.4f}, AllowedSizes={curriculum_sampler.allowed_sizes}")

                # The training curve is smoothed over a window instead of being compared
                # episode by episode: one episode is a noisy sample of the policy.
                window = max(1, int(getattr(cfg, 'select_window', 100)))
                if len(normalized_rewards) >= window:
                    best_norm_reward = max(best_norm_reward,
                                           sum(normalized_rewards[-window:]) / window)
                    agent.best_norm_reward = best_norm_reward

                # Validation block.  It selects the reported weights and it stops the run as
                # soon as the average objective stops improving, i.e. the run is never trained
                # past the point where the policy keeps getting better.
                if (total_episodes >= cfg.min_episodes
                        and total_episodes % max(1, int(cfg.val_interval)) == 0):
                    objectives = evaluate_validation(agent, validation_set, max_jobs,
                                                     global_max_ops, global_max_machines,
                                                     max_proc_time, max_due_date,
                                                     seed=cfg.val_seed)
                    scores = [weighted_objective(obj) for obj in objectives]
                    if val_reference is None:
                        # The first check defines the reference of every instance, so the score
                        # is a relative objective that does not depend on the instance size
                        # which dominates the raw weighted value.
                        val_reference = [value if abs(value) > 1e-9 else 1.0 for value in scores]
                    val_score = float(np.mean([value / ref
                                               for value, ref in zip(scores, val_reference)]))
                    val_curve.append((total_episodes, val_score))
                    print(f"Validation {len(val_curve):3d} at episode {total_episodes}: "
                          f"relative objective {val_score:.6f} (best {best_val:.6f})")
                    if val_score < best_val * (1.0 - cfg.val_tol):
                        best_val = val_score
                        stale_checks = 0
                        agent.episode_counter = total_episodes
                        agent.val_curve = list(val_curve)
                        agent.val_reference = list(val_reference)
                        agent.best_val = best_val
                        agent.converged = False
                        agent.save(best_model_path)
                    else:
                        stale_checks += 1
                        if stale_checks >= cfg.val_patience:
                            print(f"Early stop after {total_episodes} episodes: the "
                                  f"validation objective did not improve for "
                                  f"{stale_checks} checks (best={best_val:.6f}).")
                            converged = True

                # Rolling checkpoint: it bounds the work that a restart repeats
                # to checkpoint_interval episodes.
                if total_episodes % checkpoint_interval == 0:
                    agent.episode_counter = total_episodes
                    agent.best_norm_reward = best_norm_reward
                    agent.save(latest_model_path)

                if reward_ema is None:
                    reward_ema = norm_reward
                else:
                    reward_ema = 0.9 * reward_ema + 0.1 * norm_reward

                pbar.set_postfix({
                    'Cmax': f"{cmax:.0f}",
                    'LB': f"{lb:.2f}",
                    'Tardy': f"{tardy:.0f}",
                    'KL': f"{stats['kl']:.4f}" if stats else 'N/A',
                    'Beta': f"{stats['beta']:.3f}" if stats else 'N/A',
                    'Steps': step_counts[i],
                    'Sizes': f"{curriculum_sampler.allowed_sizes}"
                })

                new_inst = curriculum_sampler.sample()
                envs[i] = HeteroGraphEnv(new_inst, max_jobs, global_max_ops, global_max_machines, max_proc_time, max_due_date, cfg)
                batch_env.envs[i] = envs[i]
                env_insts[i] = new_inst
                next_states[i] = envs[i].reset()
                traj_buffers[i] = []
                ep_rewards[i] = torch.tensor(0.0, device=DEVICE)
                step_counts[i] = 0

        states = next_states

        if total_episodes % empty_cache_interval == 0 and DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    pbar.close()
    # The finished run is checkpointed as well, so that a later restart of the
    # same command finds every episode already trained.
    agent.episode_counter = total_episodes
    agent.val_curve = list(val_curve)
    agent.val_reference = list(val_reference) if val_reference else []
    agent.best_val = best_val
    agent.converged = bool(converged)
    agent.save(latest_model_path)
    print("Training completed." if not converged else "Training stopped early (converged).")
    print(f"Best validation objective: {best_val:.6f}")
    print(f"Best smoothed normalised reward: {best_norm_reward:.4f}")

    # Save plots
    os.makedirs(TRAIN_FIGURE_DIR, exist_ok=True)
    # The three-goal plots reduce over the objective history that the validation writes.  A run
    # that ends before its first validation therefore has an empty history, and numpy rejects the
    if cmax_list:
        plot_three_goals(cmax_list, lb_list, tardy_list,
                         os.path.join(TRAIN_FIGURE_DIR, "three_goals.png"))
        plot_three_goals_normalized(cmax_list, lb_list, tardy_list,
                                    os.path.join(TRAIN_FIGURE_DIR, "three_goals_normalized.png"))
    plot_kl_divergence(kl_episodes,
                       os.path.join(TRAIN_FIGURE_DIR, "kl_divergence.png"))
    plot_loss(loss_episodes, os.path.join(TRAIN_FIGURE_DIR, "total_loss.png"))

if __name__ == "__main__":
    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR, exist_ok=True)
        example_file = os.path.join(INSTANCE_DIR, "mo_fjsp_000_small_train.txt")
        with open(example_file, 'w') as f:
            f.write("2 2\n")
            f.write("2 1 1 10 1 2 20 1 1 5\n")
            f.write("2 1 2 15 1 1 10 1 2 10\n")
            f.write("100 100\n")
            f.write("1 1\n")
            f.write("# Instance ID: 0\n# Size type: small\n")
        print(f"Generated example instance file: {example_file}")


    # Optional episode budget, mirroring the --episodes flag of the baseline trainers, so that a
    # run can target the same number of episodes as the comparison methods.
    arguments = sys.argv[1:]
    for position, argument in enumerate(arguments):
        if argument == "--episodes":
            cfg.n_episodes = int(arguments[position + 1])
        elif argument == "--lr":
            cfg.lr = float(arguments[position + 1])

    train()

