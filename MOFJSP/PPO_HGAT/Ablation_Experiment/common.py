"""
PPO+HGAT Ablation Study Common Module
Contains all shared classes, functions, and configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

# Intra-op thread count: every training run is launched with a slice of the CPU
# budget, and the torch default (one thread per physical core) oversubscribes the
# machine while several runs share it.
THREAD_BUDGET = int(os.environ.get('TORCH_NUM_THREADS', '0') or 0)
if THREAD_BUDGET > 0:
    torch.set_num_threads(THREAD_BUDGET)
import sys
import glob
import time
import warnings
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque
import re
import shutil

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
W1, W2, W3 = 0.4, 0.3, 0.3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Training instance directory (under project root)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # MOFJSP directory
TRAIN_INSTANCE_DIR = os.path.join(PROJECT_ROOT, "DataSet", "mo_fjsp_instances")
TRAIN_PATTERN = "mo_fjsp_*_train.txt"
# Held-out split of the same directory: the twenty validation files every arm selects on.
VAL_PATTERN = "mo_fjsp_*_val.txt"

# Comparison dataset directory
COMPARISON_DIR = os.path.join(PROJECT_ROOT, "DataSet", "comparison_instances")

# Base model save path
BASE_MODEL_DIR = os.path.join(CURRENT_DIR, "model")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# A5 full model path (pre-trained)
A5_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "ppo_hgat", "ppo_hgat_best.pth")

# Experiment definitions.
# Every variant shares the single Config object below, so its network width cannot be inherited
EXPERIMENTS = [
    {
        'name': 'A1 (Homogeneous+Edge Feature+Hierarchical PPO)',
        'use_hgat': False,
        'use_homogeneous': True,
        'use_edge_feat': True,
        'use_mlp_encoder': False,
        'use_hierarchical': True,
        'model_dir': os.path.join(BASE_MODEL_DIR, 'A1'),
        'gat_hidden_dim': 256,
        'gat_out_dim': 256,
        'policy_hidden_dim': 512,
        # Aligned with the main training recipe of A5 so the ablation compares the
        # architectures and not two different optimisers: lr 2e-4, gamma 1.0,
        # gradient clip 50 and no value clip.
        'lr': 2e-4,
        'gamma': 1.0,
        'max_grad_norm': 50.0,
        'value_clip': None
    },
    {
        'name': 'A2 (Heterogeneous+No Edge Feature+Hierarchical PPO)',
        'use_hgat': True,
        'use_homogeneous': False,
        'use_edge_feat': False,
        'use_mlp_encoder': False,
        'use_hierarchical': True,
        'model_dir': os.path.join(BASE_MODEL_DIR, 'A2'),
        'gat_hidden_dim': 256,
        'gat_out_dim': 256,
        'policy_hidden_dim': 512,
        # Aligned with the A5 optimiser of this wave: the ablation compares the
        # heterogeneous encoder without edge features against the full model, not
        # two different optimisers.
        'lr': 2e-4,
        'gamma': 1.0,
        'max_grad_norm': 50.0,
        'value_clip': None
    },
    {
        'name': 'A3 (Heterogeneous+Edge Feature+MLP+Hierarchical PPO)',
        'use_hgat': False,
        'use_homogeneous': False,
        'use_edge_feat': True,
        'use_mlp_encoder': True,
        'use_hierarchical': True,
        'model_dir': os.path.join(BASE_MODEL_DIR, 'A3'),
        'gat_hidden_dim': 256,
        'gat_out_dim': 256,
        'policy_hidden_dim': 512,
        # The MLP encoder arm needs its own step size and trust region: the shared
        # recipe degraded it from the first check on.  The discount, gradient clip
        # and value clip follow A5 so the remaining difference is the encoder.
        'lr': 3e-4,
        'kl_target': 0.03,
        'beta_start': 0.5,
        'gamma': 1.0,
        'max_grad_norm': 50.0,
        'value_clip': None
    },
    {
        'name': 'A4 (Heterogeneous+Edge Feature+GAT+Joint PPO)',
        'use_hgat': True,
        'use_homogeneous': False,
        'use_edge_feat': True,
        'use_mlp_encoder': False,
        'use_hierarchical': False,
        'model_dir': os.path.join(BASE_MODEL_DIR, 'A4'),
        'gat_hidden_dim': 256,
        'gat_out_dim': 256,
        'policy_hidden_dim': 512,
        # Same optimiser as A5; the joint policy keeps its single environment
        # because its action space is the full operation-machine pair.
        'lr': 2e-4,
        'gamma': 1.0,
        'max_grad_norm': 50.0,
        'value_clip': None
    },
    {
        'name': 'A5 (Heterogeneous+Edge Feature+GAT+Hierarchical PPO) Full Model',
        'use_hgat': True,
        'use_homogeneous': False,
        'use_edge_feat': True,
        'use_mlp_encoder': False,
        'use_hierarchical': True,
        'model_dir': os.path.join(BASE_MODEL_DIR, 'A5'),
        'gat_hidden_dim': 256,
        'gat_out_dim': 256,
        'policy_hidden_dim': 512,
        'pretrained_path': A5_MODEL_PATH   # Pre-trained model path, for direct loading
    }
]

torch.backends.cudnn.benchmark = True


# Configuration class
class Config:
    n_episodes = 10000
    lr = 5e-4                      # Reduce learning rate to prevent gradient explosion
    lr_decay = 0.5
    lr_decay_steps = 2000
    min_lr = 1e-5
    gamma = 0.99
    gae_lambda = 0.95
    # Same correction as the proposed method of this wave.  The measured divergence per update is
    # around 0.003 to 0.01, so a target of 0.1 leaves the penalty inactive (the coefficient decays
    kl_target = 0.1
    beta_start = 0.2
    beta_lr = 0.05
    # Floor of the adapted KL coefficient, see the correction above.  The previous waves clipped it
    # at 0.01, which is the level at which the penalty stops mattering.
    beta_floor = 0.01
    value_coef = 0.5
    entropy_coef = 0.01
    # Slowed down together with the proposed method of this wave: at the old rate the coefficient
    # reached its floor within one run and the arms stopped exploring before the end.
    entropy_decay = 0.9995
    # Restored to the family norm of ten (see the note in HGNN.py): the loosened fifty is what
    # let the value term of the GAT baseline of this family diverge to a loss of 1.1e6.
    max_grad_norm = 10.0
    gat_hidden_dim = 128
    gat_out_dim = 128
    policy_hidden_dim = 128
    value_clip = None
    # Enabled for the same reason as in PPO_HGAT.py: the value term of this family diverges
    # without a bound (the GAT baseline printed a total loss of 1.1e6) and its gradient then
    # dominates the encoder that the policy shares with the critic.
    value_clip = 0.5
    # Back to None for the same reason as in PPO_HGAT.py: this file shares the update structure of
    # the proposed method, whose value-clip branch reads "values" from the enclosing scope - a name
    value_clip = 0.5
    # The method itself is trained without the clip, on a measurement of its own recipe of the same
    # day: two of its waves of 2026-10-12 ran in the same frame, on the same twenty held-out instances
    reward_scaling = 5.0
    reward_clip = 2.0
# The disjunctive (machine competition) edges are part of the state this wave describes and the
# environment builds them unconditionally, capped per machine (DISJ_PAIRS_PER_MACHINE pairs),
    sampling_temperature = 1.0
    logits_clip = 50.0              # Logit clip applied before action masking (see UpperPolicy)
    max_steps_per_episode = 2000
    stats_window = 100
    max_edges_per_type = 30000
    # Collected instances.  The value was ten, which is above what a single run of this family was
    # measured to hold once the curriculum reaches the large instances (sixteen environments at
    num_envs = 6
    # Mixed precision is off.  The proposed method trains in full precision (see PPO_HGAT.Config),
    # so the arms have to as well, and with half precision the policy update of every arm that does
    amp_enabled = False

    # Model selection and early stopping.  One episode is a noisy sample of the policy (its
    # instance scale and its operation count change from episode to episode), so the reported
    val_interval = 250
    # Four instances per size reduce the sampling noise of a check: with two instances the
    # reported objective swung by ten percent between two checks of one run.
    val_instances_per_size = 8
    val_seed = 0
    # A stale check is one validation of 250 episodes, so the patience below allows at most 2000
    # episodes past the peak before the run is closed.
    val_patience = 8
    val_tol = 0.02
    select_window = 100
    # No early stop before this budget: the window has to open before the peak of the run.
    # The guard is the one of the trainer, i.e. PPO_HGAT.Config.min_episodes = 1500; the value here
    min_episodes = 1500
    # The curriculum advances on a fixed episode scale instead of on the episode budget, so the
    # schedule does not change when the budget of the run is changed.
    curriculum_episodes = 2500
    # Aligned with every baseline of the study, for the reason recorded in PPO_HGAT.py: the scale
    # decides when the curriculum introduces the medium and the large instances, i.e. the training
    curriculum_episodes = 4000
    # Floor of the entropy coefficient: the decay alone would make the policy deterministic long
    # before the end of the run.
    entropy_min = 0.005

    # Curriculum learning parameters (enabled by default)
    curriculum_enabled = True
    curriculum_stages = [
        (0.0, 0.3, ['small']),
        (0.3, 0.6, ['small', 'medium']),
        (0.6, 1.0, ['small', 'medium', 'large'])
    ]

cfg = Config()

# Base values of the shared optimiser.  An arm that overrides one of them is applied to these
# values and not to the current content of ``cfg``: the arms are applied one after another to the
RECIPE_TAG = "ablation-a5align-2026-10-18"
BASE_LR = cfg.lr
BASE_KL_TARGET = cfg.kl_target
BASE_BETA_START = cfg.beta_start
BASE_GAMMA = cfg.gamma
BASE_MAX_GRAD_NORM = cfg.max_grad_norm
BASE_VALUE_CLIP = cfg.value_clip


def apply_experiment_dims(exp_config):
    """Pin the network width of an experiment to the value declared in EXPERIMENTS.

    The width has to be applied right before the networks of a variant are built, because all
    variants share the single cfg object above and would otherwise silently inherit whatever
    default it happens to hold.
    """
    cfg.gat_hidden_dim = exp_config.get('gat_hidden_dim', cfg.gat_hidden_dim)
    cfg.gat_out_dim = exp_config.get('gat_out_dim', cfg.gat_out_dim)
    cfg.policy_hidden_dim = exp_config.get('policy_hidden_dim', cfg.policy_hidden_dim)
    # Optimiser overrides of a single arm.  The arms share one cfg object and one recipe, and an
    # arm may override the step size and the trust region when its own curve shows that the shared
    cfg.lr = exp_config.get('lr', BASE_LR)
    cfg.kl_target = exp_config.get('kl_target', BASE_KL_TARGET)
    cfg.beta_start = exp_config.get('beta_start', BASE_BETA_START)
    # The reward and the value path are part of the recipe, not of the architecture.  An arm
    # that must be comparable with the main model has to use the same discount, gradient clip
    # and value-clip switch; otherwise the comparison mixes two optimisers.
    cfg.gamma = exp_config.get('gamma', BASE_GAMMA)
    cfg.max_grad_norm = exp_config.get('max_grad_norm', BASE_MAX_GRAD_NORM)
    cfg.value_clip = exp_config.get('value_clip', BASE_VALUE_CLIP)


# Check training data
def check_training_data():
    """Check if training data exists; if not, raise an error."""
    if not os.path.exists(TRAIN_INSTANCE_DIR):
        raise FileNotFoundError(
            f"Training instance directory does not exist: {TRAIN_INSTANCE_DIR}\n"
            f"Please run Generate_DataSet.py in the project root to generate training data."
        )
    files = glob.glob(os.path.join(TRAIN_INSTANCE_DIR, TRAIN_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No files matching {TRAIN_PATTERN} found in directory {TRAIN_INSTANCE_DIR}.\n"
            f"Please run Generate_DataSet.py in the project root to generate training data."
        )
    print(f"Found {len(files)} training instance files, continuing training.")


# Data reading and instance definition
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

        # Build index mappings
        self.op_index_map = {}
        op_idx = 0
        for job in range(self.n_jobs):
            for op in range(self.ops_per_job[job]):
                self.op_index_map[(job, op)] = op_idx
                op_idx += 1

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
        # fixed-preference heuristic of the lower policy (Eq. 10 of the manuscript).  The fill
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
        seq_src, seq_dst = [], []
        for job in range(self.n_jobs):
            for op in range(self.ops_per_job[job] - 1):
                u = self.op_index_map[(job, op)]
                v = self.op_index_map[(job, op+1)]
                seq_src.append(u)
                seq_dst.append(v)
        self.seq_edges_src = torch.tensor(seq_src, dtype=torch.long, device=DEVICE)
        self.seq_edges_dst = torch.tensor(seq_dst, dtype=torch.long, device=DEVICE)

        alloc_src, alloc_dst, alloc_feat = [], [], []
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

        # Disjunctive edges of the manuscript: two operations that can be processed on the same
        # machine compete for it.  The set is capped per machine (the graph sampling the
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

    def get_pt(self, job, op, machine):
        return self.proc_time_matrix[self.op_index_map[(job, op)], machine].item()


# Curriculum learning sampler
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


# ---------------------------- State normalization ----------------------------
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


# ---------------------------- Heterogeneous graph environment ----------------------------
class HeteroGraphEnv:
    def __init__(self, instance: MOFJSPInstance, global_max_jobs, global_max_ops, global_max_machines,
                 global_max_proc_time, global_max_due_date, cfg, feature_mode="legacy"):
        # The ablation arms share the original 3/3 feature table.  The evaluation entry point
        # passes the mode inferred from the checkpoint, so accept the argument and reject a
        # table this environment cannot build instead of silently scoring the wrong graph.
        if feature_mode not in ("legacy", None):
            raise ValueError("The ablation environment supports the 3/3 legacy feature table only")
        self.feature_mode = feature_mode
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
        self.due_dates_tensor = instance.due_dates_tensor
        self.ops_per_job_tensor = instance.ops_per_job_tensor
        self.op_idx_to_job = instance.op_idx_to_job
        self.op_idx_to_op = instance.op_idx_to_op

        self.seq_edges_src = instance.seq_edges_src
        self.disj_edges_src = instance.disj_edges_src
        self.disj_edges_dst = instance.disj_edges_dst
        self.disj_edges_feat = instance.disj_edges_feat
        self.seq_edges_dst = instance.seq_edges_dst
        self.alloc_edges_src = instance.alloc_edges_src
        self.alloc_edges_dst = instance.alloc_edges_dst
        self.alloc_edges_feat = instance.alloc_edges_feat
        self.alloc_rev_edges_src = instance.alloc_rev_edges_src
        self.alloc_rev_edges_dst = instance.alloc_rev_edges_dst
        self.alloc_rev_edges_feat = instance.alloc_rev_edges_feat

        self.machine_ops_list = instance.machine_ops_list

        # Static operation features
        self.static_op_feats = torch.zeros((self.global_max_ops, self.op_feat_dim), dtype=torch.float32, device=DEVICE)
        for job in range(instance.n_jobs):
            ops_in_job = instance.ops_per_job[job]
            due_time = instance.due_dates[job] / (global_max_due_date + EPS)
            job_id_norm = job / max(1, global_max_jobs - 1)
            for op in range(ops_in_job):
                node_idx = instance.op_index_map[(job, op)]
                self.static_op_feats[node_idx, 0] = op / max(1, ops_in_job - 1)
                self.static_op_feats[node_idx, 1] = due_time
                self.static_op_feats[node_idx, 2] = job_id_norm

        # Static machine features (capability, machine id; load and availability are dynamic)
        self.static_mac_feats = torch.zeros((self.global_max_machines, self.mac_feat_dim), dtype=torch.float32, device=DEVICE)
        for m in range(instance.n_machines):
            self.static_mac_feats[m, 2] = instance.capability_tensor[m]

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

        self.op_feats_t = self.static_op_feats.clone()
        self.mac_feats_t = self.static_mac_feats.clone()

        # Constants the step and the state builder use over and over.  Rebuilding them in place
        # costs a host to device transfer (for the scalar) and an allocation plus a fill kernel
        # (for the vectors) in every step of every parallel environment.
        self.zero_scalar = torch.zeros((), dtype=torch.float32, device=DEVICE)
        self.all_job_ids = torch.arange(self.inst.n_jobs, device=DEVICE)
        self.seq_edge_feat_full = torch.ones((self.seq_edges_src.shape[0], 1),
                                            dtype=torch.float32, device=DEVICE)
        # Number of scheduled operations of this episode.  Counting them in python keeps the
        # termination test free of a host synchronisation.
        self.scheduled_ops = 0

        return self._get_state()

    def step(self, action):
        if isinstance(action, tuple) and len(action) == 3:
            job, op, machine = action
            op_idx = self.job_op_to_idx[job.long(), op.long()]
        else:
            op_idx, machine = action
            job = self.op_idx_to_job[op_idx]
            op = self.op_idx_to_op[op_idx]

        p_time = self.proc_time_matrix[op_idx, machine.long()]

        pre_idx = self.op_predecessor[op_idx]
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

        done = self.scheduled_ops >= self.total_op_nodes
        if done:
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


        unsched_mask = ~self.op_scheduled

        # The edge arrays of an instance never change, only their visibility does.  The state
        # therefore carries the static arrays plus one visibility flag per edge, and the packer
        seq_valid = unsched_mask[self.seq_edges_src] & unsched_mask[self.seq_edges_dst]
        alloc_valid = unsched_mask[self.alloc_edges_src]
        alloc_rev_valid = unsched_mask[self.alloc_rev_edges_dst]
        # Competition stays visible while both of its operations are still unscheduled.
        disj_valid = unsched_mask[self.disj_edges_src] & unsched_mask[self.disj_edges_dst]

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
            'op_feats': self.op_feats_t.clone(),
            'mac_feats': self.mac_feats_t.clone(),
            'edges': edges_t,
            'edge_feats': edge_feats_t,
            'edge_valid': edge_valid_t,
            'op_mask': op_mask,
        }




# Batch environment manager
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


# ---------------------------- Orthogonal initialization ----------------------------
def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


# Network modules

# Heterogeneous GAT layer (supports edge feature switch and optional LayerNorm)
class HeteroGATLayer(nn.Module):
    def __init__(self, in_dim_op, in_dim_mac, out_dim, num_heads=4, use_edge_feat=True, use_layer_norm=True):
        super().__init__()
        self.heads = num_heads
        assert out_dim % num_heads == 0
        self.d_k = out_dim // num_heads
        self.out_dim = out_dim
        self.use_edge_feat = use_edge_feat
        self.use_layer_norm = use_layer_norm

        self.W_op = nn.Linear(in_dim_op, out_dim)
        self.W_mac = nn.Linear(in_dim_mac, out_dim)
        orthogonal_init(self.W_op, gain=nn.init.calculate_gain('relu'))
        orthogonal_init(self.W_mac, gain=nn.init.calculate_gain('relu'))

        att_dim = 2 * self.d_k + (1 if use_edge_feat else 0)
        self.att_seq = nn.Parameter(torch.randn(num_heads, att_dim))
        self.att_op_mac = nn.Parameter(torch.randn(num_heads, att_dim))
        self.att_mac_op = nn.Parameter(torch.randn(num_heads, att_dim))
        # Disjunctive (machine competition) channel of the heterogeneous arms.  The environment
        # builds the competing operation pairs and reports them in the state, but this shared
        self.att_disj = nn.Parameter(torch.randn(num_heads, att_dim))

        self.leaky = nn.LeakyReLU(0.2)
        self.max_edges = cfg.max_edges_per_type

        if use_layer_norm:
            self.ln_op = nn.LayerNorm(out_dim)
            self.ln_mac = nn.LayerNorm(out_dim)
        else:
            self.ln_op = nn.Identity()
            self.ln_mac = nn.Identity()

    def forward(self, h_op, h_mac, edges, edge_feats):
        h_op = h_op.float()
        h_mac = h_mac.float()
        device = h_op.device
        dtype = h_op.dtype

        z_op = self.W_op(h_op).view(-1, self.heads, self.d_k)
        z_mac = self.W_mac(h_mac).view(-1, self.heads, self.d_k)

        out_op = torch.zeros_like(z_op)
        out_mac = torch.zeros_like(z_mac)

        edge_feats_proc = {}
        for key, feat in edge_feats.items():
            if feat.numel() > 0:
                edge_feats_proc[key] = feat.to(dtype)
            else:
                edge_feats_proc[key] = feat

        def apply_attn(src_emb, dst_emb, src_idx, dst_idx, feat_e, att_param, out_buffer, gate=False):
            if src_idx.numel() == 0:
                return
            if src_idx.numel() > self.max_edges:
                return
            dtype_out = out_buffer.dtype
            h_s = src_emb[src_idx].to(dtype_out)
            h_d = dst_emb[dst_idx].to(dtype_out)
            if self.use_edge_feat and feat_e.numel() > 0:
                f_e = feat_e.to(dtype_out).unsqueeze(1).repeat(1, self.heads, 1)
                concat = torch.cat([h_s, h_d, f_e], dim=-1)
            else:
                concat = torch.cat([h_s, h_d], dim=-1)
            att_param = att_param.to(dtype_out)
            scores = (concat * att_param).sum(dim=-1)
            scores = self.leaky(scores)
            if gate:
                # Sigmoid gate of the conjunctive channel.  Every operation has exactly one
                # predecessor and a softmax over a single incoming edge normalises to one, so the
                weight = torch.sigmoid(scores).to(dtype_out)
                weighted = (h_s * weight.unsqueeze(-1)).to(dtype_out)
                out_buffer.index_add_(0, dst_idx, weighted)
                return
            # Numerical stability: the largest score of the batch is subtracted before the
            # exponential, because the exponential of a raw score overflows once the score
            scores = scores - scores.detach().max()
            alpha = torch.exp(scores).to(dtype_out)

            denom = torch.zeros(out_buffer.shape[0], self.heads, device=device, dtype=dtype_out)
            denom.index_add_(0, dst_idx, alpha)
            denom = denom[dst_idx] + torch.tensor(EPS, dtype=dtype_out, device=device)
            norm_alpha = alpha / denom
            weighted = h_s * norm_alpha.unsqueeze(-1)
            weighted = weighted.to(dtype_out)  # Ensure type matches out_buffer
            out_buffer.index_add_(0, dst_idx, weighted)

        if edges['seq'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['seq'][0], edges['seq'][1], edge_feats_proc['seq'], self.att_seq,
                       out_op, gate=True)
        if edges['op_mac'][0].numel() > 0:
            apply_attn(z_op, z_mac, edges['op_mac'][0], edges['op_mac'][1], edge_feats_proc['op_mac'], self.att_op_mac, out_mac)
        if edges['mac_op'][0].numel() > 0:
            apply_attn(z_mac, z_op, edges['mac_op'][0], edges['mac_op'][1], edge_feats_proc['mac_op'], self.att_mac_op, out_op)
        # Same disjunctive channel as the proposed method: the arms of the study must consume the
        # heterogeneous state of the manuscript, otherwise the ablation measures a conjunctive
        # encoder against a heterogeneous one.
        if edges.get('disj') is not None and edges['disj'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['disj'][0], edges['disj'][1], edge_feats_proc['disj'],
                       self.att_disj, out_op)

        res_op = (out_op.flatten(1) + self.W_op(h_op))
        res_mac = (out_mac.flatten(1) + self.W_mac(h_mac))

        # Apply optional layer normalization
        res_op = self.ln_op(res_op)
        res_mac = self.ln_mac(res_mac)

        return F.elu(res_op), F.elu(res_mac)


class HeteroGAT(nn.Module):
    def __init__(self, dim_op, dim_mac, hidden_dim, out_dim, use_edge_feat=True, use_layer_norm=True):
        super().__init__()
        self.l1 = HeteroGATLayer(dim_op, dim_mac, hidden_dim, use_edge_feat=use_edge_feat, use_layer_norm=use_layer_norm)
        self.l2 = HeteroGATLayer(hidden_dim, hidden_dim, out_dim, use_edge_feat=use_edge_feat, use_layer_norm=use_layer_norm)
        self.out_dim = out_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, state_dict):
        op_feats = state_dict['op_feats']
        mac_feats = state_dict['mac_feats']
        edges = state_dict['edges']
        edge_feats = state_dict['edge_feats']

        h_op1, h_mac1 = self.l1(op_feats, mac_feats, edges, edge_feats)
        h_op2, h_mac2 = self.l2(h_op1, h_mac1, edges, edge_feats)

        return h_op2, h_mac2


# Homogeneous GAT (supports optional LayerNorm)
class HomogeneousGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, use_edge_feat=True, use_layer_norm=True):
        super().__init__()
        self.heads = num_heads
        assert out_dim % num_heads == 0
        self.d_k = out_dim // num_heads
        self.out_dim = out_dim
        self.use_edge_feat = use_edge_feat
        self.use_layer_norm = use_layer_norm

        self.W = nn.Linear(in_dim, out_dim)
        orthogonal_init(self.W, gain=nn.init.calculate_gain('relu'))

        att_dim = 2 * self.d_k + (1 if use_edge_feat else 0)
        self.att = nn.Parameter(torch.randn(num_heads, att_dim))

        self.leaky = nn.LeakyReLU(0.2)
        self.max_edges = cfg.max_edges_per_type

        if use_layer_norm:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = nn.Identity()

    def forward(self, h, edge_index, edge_feat):
        h = h.float()
        device = h.device
        dtype = h.dtype
        num_nodes = h.shape[0]

        z = self.W(h).view(-1, self.heads, self.d_k)

        src, dst = edge_index
        if src.numel() == 0:
            return F.elu(z.flatten(1))

        h_s = z[src]
        h_d = z[dst]

        if self.use_edge_feat and edge_feat.numel() > 0:
            f_e = edge_feat.to(dtype).unsqueeze(1).repeat(1, self.heads, 1)
            concat = torch.cat([h_s, h_d, f_e], dim=-1)
        else:
            concat = torch.cat([h_s, h_d], dim=-1)

        att_param = self.att.to(dtype)
        scores = (concat * att_param.unsqueeze(0)).sum(dim=-1)
        scores = self.leaky(scores)
        # See the note in the layer above: the largest score is subtracted before the
        # exponential so that a raw score past 88 cannot overflow into NaN.
        scores = scores - scores.detach().max()
        alpha = torch.exp(scores)

        out = torch.zeros(num_nodes, self.heads, self.d_k, device=device, dtype=dtype)
        denom = torch.zeros(num_nodes, self.heads, device=device, dtype=dtype)
        denom.index_add_(0, dst, alpha)
        denom = denom[dst] + EPS
        norm_alpha = alpha / denom
        weighted = h_s * norm_alpha.unsqueeze(-1)
        weighted = weighted.to(dtype)
        out.index_add_(0, dst, weighted)

        out = out.flatten(1)
        out = out + self.W(h)  # Residual
        out = self.ln(out)
        return F.elu(out)


class HomogeneousGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, use_edge_feat=True, use_layer_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HomogeneousGATLayer(in_dim, hidden_dim, num_heads, use_edge_feat, use_layer_norm))
        for _ in range(num_layers - 2):
            self.layers.append(HomogeneousGATLayer(hidden_dim, hidden_dim, num_heads, use_edge_feat, use_layer_norm))
        self.layers.append(HomogeneousGATLayer(hidden_dim, out_dim, num_heads, use_edge_feat, use_layer_norm))
        self.out_dim = out_dim

    def forward(self, h, edge_index, edge_feat):
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
        return h


# MLP encoder
class MLPEncoder(nn.Module):
    def __init__(self, dim_op, dim_mac, hidden_dim, out_dim):
        super().__init__()
        self.op_net = nn.Sequential(
            orthogonal_init(nn.Linear(dim_op, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, out_dim), gain=nn.init.calculate_gain('relu'))
        )
        self.mac_net = nn.Sequential(
            orthogonal_init(nn.Linear(dim_mac, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, out_dim), gain=nn.init.calculate_gain('relu'))
        )
        self.out_dim = out_dim

    def forward(self, state_dict):
        op_feats = state_dict['op_feats']
        mac_feats = state_dict['mac_feats']
        h_op = self.op_net(op_feats)
        h_mac = self.mac_net(mac_feats)
        return h_op, h_mac


# Hierarchical policies
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
            # Clip the raw logits BEFORE masking: clamping after masked_fill would lift
            # the fill value up to the clip floor and let infeasible actions keep
            # probability mass in the softmax output.
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, fill_value)
            if (logits == fill_value).all():
                # Every action of this state is infeasible: fall back to a uniform
                # distribution so that Categorical stays well defined.
                probs = torch.ones_like(logits) / logits.shape[0]
            else:
                probs = F.softmax(logits, dim=0)
                if mask is not None:
                    # Infeasible actions must keep exactly zero probability.
                    probs = probs * (mask > 0).float()
                probs = probs / probs.sum().clamp(min=EPS)
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
            probs = torch.zeros_like(logits)
            for i in range(batch_size):
                row = logits[i]
                if (row == fill_value).all():
                    # All actions of this state are infeasible: uniform fallback
                    probs[i] = (torch.ones_like(row) / row.shape[0]).to(probs.dtype)
                else:
                    row_probs = F.softmax(row, dim=0)
                    if mask is not None:
                        # Infeasible actions must keep exactly zero probability
                        row_probs = row_probs * (mask[i] > 0).float()
                    probs[i] = (row_probs / row_probs.sum().clamp(min=EPS)).to(probs.dtype)
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
                # Fixed-preference heuristic term of the manuscript's lower policy (Eq. 10):
                # processing time, load and waiting time of every candidate machine, so the arm
                # reads the same assignment score as the full method.
                logits = logits + heuristic
            fill_value = -1e9
            # Clip the raw logits BEFORE masking: clamping after masked_fill would lift
            # the fill value up to the clip floor and let infeasible actions keep
            # probability mass in the softmax output.
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, fill_value)
            if (logits == fill_value).all():
                # Every action of this state is infeasible: fall back to a uniform
                # distribution so that Categorical stays well defined.
                probs = torch.ones_like(logits) / logits.shape[0]
            else:
                probs = F.softmax(logits, dim=0)
                if mask is not None:
                    # Infeasible actions must keep exactly zero probability.
                    probs = probs * (mask > 0).float()
                probs = probs / probs.sum().clamp(min=EPS)
        else:
            batch_size, M, _ = mac_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, M, -1)
            op_rep = selected_op_emb.unsqueeze(1).expand(-1, M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            if heuristic is not None:
                logits = logits + heuristic
            fill_value = -1e9
            # Clip before masking (same reason as the single-sample branch)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, fill_value)
            probs = torch.zeros_like(logits)
            for i in range(batch_size):
                row = logits[i]
                if (row == fill_value).all():
                    # All actions of this state are infeasible: uniform fallback
                    probs[i] = (torch.ones_like(row) / row.shape[0]).to(probs.dtype)
                else:
                    row_probs = F.softmax(row, dim=0)
                    if mask is not None:
                        # Infeasible actions must keep exactly zero probability
                        row_probs = row_probs * (mask[i] > 0).float()
                    probs[i] = (row_probs / row_probs.sum().clamp(min=EPS)).to(probs.dtype)
        return probs


# Joint policy (flat PPO)
class JointPolicy(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_op_global, n_mac_global):
        super().__init__()
        self.n_op_global = n_op_global
        self.n_mac_global = n_mac_global
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 3, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1)
        )
        self.chunk_size = 50

    def forward(self, g_emb, h_op, h_mac, op_mask, mac_mask_per_op):
        # Force conversion to float32 to avoid half-precision issues from AMP
        g_emb = g_emb.float()
        h_op = h_op.float()
        h_mac = h_mac.float()
        op_mask = op_mask.float()
        mac_mask_per_op = mac_mask_per_op.float()

        single = (g_emb.dim() == 1)
        if single:
            g_emb = g_emb.unsqueeze(0)
            h_op = h_op.unsqueeze(0)
            h_mac = h_mac.unsqueeze(0)
            op_mask = op_mask.unsqueeze(0)
            mac_mask_per_op = mac_mask_per_op.unsqueeze(0)

        batch_size = g_emb.shape[0]
        n_mac = h_mac.shape[1]
        n_op_global = h_op.shape[1]

        # Assert input dimensions match initialization
        assert n_op_global == self.n_op_global, f"Expected n_op_global={self.n_op_global}, got {n_op_global}"
        assert n_mac == self.n_mac_global, f"Expected n_mac_global={self.n_mac_global}, got {n_mac}"

        full_logits = torch.full((batch_size, n_op_global * n_mac), -float('inf'), device=g_emb.device, dtype=torch.float32)

        for b in range(batch_size):
            eligible_op_indices = torch.nonzero(op_mask[b] > 0).squeeze(1)
            if eligible_op_indices.numel() == 0:
                # ========== Modification point 1: throw detailed exception when no eligible operation ==========
                error_msg = (f"JointPolicy: op_mask for state {b} is all zeros, cannot sample.\n"
                             f"op_mask: {op_mask[b].cpu().numpy()}\n"
                             f"Current instance info: total ops={h_op.shape[1]}, machines={n_mac}\n"
                             f"Possible cause: state has finished but environment not properly reset, or op_mask construction error.")
                raise RuntimeError(error_msg)

            num_eligible = eligible_op_indices.shape[0]
            h_op_eligible = h_op[b, eligible_op_indices, :]
            mac_mask_eligible = mac_mask_per_op[b, eligible_op_indices, :]

            logits_chunks = []
            for start in range(0, n_mac, self.chunk_size):
                end = min(start + self.chunk_size, n_mac)
                mac_chunk = h_mac[b, start:end, :]

                g_exp = g_emb[b].unsqueeze(0).unsqueeze(0)
                op_exp = h_op_eligible.unsqueeze(1)
                mac_exp = mac_chunk.unsqueeze(0)

                g_part = g_exp.expand(num_eligible, end-start, -1)
                op_part = op_exp.expand(-1, end-start, -1)
                mac_part = mac_exp.expand(num_eligible, -1, -1)

                pair_emb = torch.cat([g_part, op_part, mac_part], dim=-1)
                logits_chunk = self.net(pair_emb).squeeze(-1)
                logits_chunks.append(logits_chunk)

            logits_eligible = torch.cat(logits_chunks, dim=-1)
            logits_eligible = logits_eligible.masked_fill(mac_mask_eligible == 0, -float('inf'))

            op_global = eligible_op_indices.unsqueeze(1)
            mac_global = torch.arange(n_mac, device=g_emb.device).unsqueeze(0)
            flat_indices = (op_global * n_mac + mac_global).reshape(-1)
            full_logits[b, flat_indices] = logits_eligible.reshape(-1).to(full_logits.dtype)

        # For each batch, if all logits are -inf (should not happen because empty mask handled), throw exception
        for b in range(batch_size):
            if (full_logits[b] == -float('inf')).all():
                error_msg = (f"JointPolicy: all logits for state {b} are -inf, cannot sample.\n"
                             f"op_mask: {op_mask[b].cpu().numpy()}\n"
                             f"Possible cause: all operation-machine combinations are masked.")
                raise RuntimeError(error_msg)

        # Remove clamp, use subtracting max for stable softmax
        full_logits = full_logits - full_logits.max(dim=-1, keepdim=True)[0]
        probs_flat = F.softmax(full_logits, dim=-1)

        if single:
            probs_flat = probs_flat.squeeze(0)
        return probs_flat


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


# Batch state packing
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
            # Competition runs between two operation nodes, so both ends take the operation
            # offset; without this branch the composition would be read as machine to operation.
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




# PPO Agent
class PPOAgent:
    def __init__(self, inst, encoder, upper, lower, critic_u, critic_l, cfg,
                 state_norm=None, joint_policy=None, use_hierarchical=True):
        self.inst = inst
        self.encoder = encoder
        self.upper = upper
        self.lower = lower
        self.critic_u = critic_u
        self.critic_l = critic_l
        self.joint_policy = joint_policy
        self.cfg = cfg
        self.state_norm = state_norm
        self.use_hierarchical = use_hierarchical

        self.gamma = cfg.gamma
        self.lam = cfg.gae_lambda
        self.clip = 0.2
        self.beta = cfg.beta_start
        self.beta_lr = cfg.beta_lr
        self.value_coef = cfg.value_coef
        self.entropy_coef = cfg.entropy_coef
        self.value_clip = cfg.value_clip
        self.max_grad_norm = cfg.max_grad_norm

        # Parameter list
        self.params = list(encoder.parameters())
        if self.use_hierarchical:
            self.params += list(upper.parameters()) + list(lower.parameters()) + list(critic_u.parameters()) + list(critic_l.parameters())
        else:
            self.params += list(joint_policy.parameters()) + list(critic_u.parameters())

        self.opt = torch.optim.Adam(self.params, lr=cfg.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay)

        self.kl_ema = None
        self.kl_alpha = 0.9

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
        self.loaded_metadata = {}

        # AMP
        self.scaler = torch.amp.GradScaler('cuda') if cfg.amp_enabled and DEVICE.type == 'cuda' else None

    def save(self, path, extra=None):
        payload = {
            'encoder': self.encoder.state_dict(),
            'upper': self.upper.state_dict() if self.use_hierarchical else None,
            'lower': self.lower.state_dict() if self.use_hierarchical else None,
            'joint_policy': self.joint_policy.state_dict() if not self.use_hierarchical else None,
            'critic_u': self.critic_u.state_dict(),
            'critic_l': self.critic_l.state_dict() if self.use_hierarchical else None,
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            # beta is a numpy scalar: storing it as a Python float keeps the checkpoint readable by
            # every interpreter regardless of the numpy major version it was written with.
            'beta': float(self.beta),
            'recipe': RECIPE_TAG,
            'entropy_coef': self.entropy_coef,
            'episode_counter': self.episode_counter,
            'scaler': self.scaler.state_dict() if self.scaler else None,
            # Running statistics of the embedding normaliser.  The proposed method trains with
            # the normalisation enabled, so an evaluation that builds the agent without restoring
            # these three values would feed the encoder an embedding of a different scale.
            'state_norm_mean': None if self.state_norm is None else self.state_norm.mean,
            'state_norm_var': None if self.state_norm is None else self.state_norm.var,
            'state_norm_count': None if self.state_norm is None else self.state_norm.count,
            # Architecture width, so that an evaluation run can verify that the checkpoint matches
            # the network it is loaded into (the ablation variants differ in width).
            'gat_hidden_dim': self.cfg.gat_hidden_dim,
            'gat_out_dim': self.cfg.gat_out_dim,
            'policy_hidden_dim': self.cfg.policy_hidden_dim
        }
        if extra:
            payload.update(extra)
        _atomic_path = path + ".tmp"
        torch.save(payload, _atomic_path)
        os.replace(_atomic_path, path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(path, map_location=DEVICE)

            # Helper function: attempt to extract state_dict of a specific module from checkpoint
            def _extract_state_dict(possible_keys):
                if isinstance(checkpoint, dict):
                    for key in possible_keys:
                        if key in checkpoint:
                            return checkpoint[key]
                # If the checkpoint itself is a state_dict (e.g., directly saved model parameters)
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    return checkpoint
                return None

            # Load encoder
            encoder_keys = ['encoder', 'gat', 'model', 'state_dict']
            encoder_sd = _extract_state_dict(encoder_keys)
            if encoder_sd is None:
                # If still not found, try using the whole checkpoint (may be a single state_dict)
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    encoder_sd = checkpoint
                else:
                    raise KeyError(f"Cannot find encoder state_dict from checkpoint, available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")
            self.encoder.load_state_dict(encoder_sd)

            # Load other modules (if present)
            if self.use_hierarchical:
                if 'upper' in checkpoint:
                    self.upper.load_state_dict(checkpoint['upper'])
                if 'lower' in checkpoint:
                    self.lower.load_state_dict(checkpoint['lower'])
                if 'critic_l' in checkpoint:
                    self.critic_l.load_state_dict(checkpoint['critic_l'])
            else:
                if 'joint_policy' in checkpoint:
                    self.joint_policy.load_state_dict(checkpoint['joint_policy'])
            if 'critic_u' in checkpoint:
                self.critic_u.load_state_dict(checkpoint['critic_u'])

            self.encoder = self.encoder.to(DEVICE).float()
            if self.use_hierarchical:
                self.upper = self.upper.to(DEVICE).float()
                self.lower = self.lower.to(DEVICE).float()
                self.critic_l = self.critic_l.to(DEVICE).float()
            else:
                self.joint_policy = self.joint_policy.to(DEVICE).float()
            self.critic_u = self.critic_u.to(DEVICE).float()

            # Recreate optimizer
            self.params = list(self.encoder.parameters())
            if self.use_hierarchical:
                self.params += list(self.upper.parameters()) + list(self.lower.parameters()) + list(self.critic_u.parameters()) + list(self.critic_l.parameters())
            else:
                self.params += list(self.joint_policy.parameters()) + list(self.critic_u.parameters())
            self.opt = torch.optim.Adam(self.params, lr=self.cfg.lr, weight_decay=1e-5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay)
            # A checkpoint written by save() carries the optimiser moments and the
            # scheduler state; the older checkpoints hold the weights only and are
            # accepted with a cold optimiser.
            if checkpoint.get('optimizer') is not None:
                try:
                    self.opt.load_state_dict(checkpoint['optimizer'])
                except Exception as exc:
                    print(f"Optimiser state not restored ({exc}), cold optimiser used.")
            if checkpoint.get('scheduler') is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                except Exception as exc:
                    print(f"Scheduler state not restored ({exc}).")

            # The schedule and the KL penalty are hyper-parameters of a recipe, not state: the config
            # is the authority, and a checkpoint of another recipe cannot carry its own values in.
            if checkpoint.get('recipe') == RECIPE_TAG:
                self.beta = checkpoint.get('beta', self.beta)
                self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            else:
                # The stored optimiser carries the learning rate of that recipe as well, so the peak
                # of the configured schedule is restored on top of it.
                for group in self.opt.param_groups:
                    group['lr'] = self.cfg.lr
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.opt, step_size=self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay)
                print("Checkpoint recipe %r differs from %r, keeping the configured recipe."
                      % (checkpoint.get('recipe'), RECIPE_TAG))
            self.episode_counter = checkpoint.get('episode_counter', 0)

            # Restore the running statistics of the embedding normaliser next to the weights, see
            # the note in save: they are part of the function the checkpoint encodes.
            if self.state_norm is not None and checkpoint.get('state_norm_mean') is not None:
                self.state_norm.mean = checkpoint['state_norm_mean'].to(DEVICE).float()
                self.state_norm.var = checkpoint['state_norm_var'].to(DEVICE).float()
                self.state_norm.count = float(checkpoint.get('state_norm_count',
                                                             self.state_norm.count))

            # Keep the architecture width and the bookkeeping stored next to the weights so that a
            # resumed run can restore them instead of restarting the bookkeeping from scratch.
            self.loaded_metadata = {key: checkpoint.get(key) for key in
                                    ('gat_hidden_dim', 'gat_out_dim', 'policy_hidden_dim',
                                     'best_norm_reward', 'episode_counter',
                                     'val_curve', 'val_reference', 'best_val', 'converged')}

            # Load scaler state if not None
            if self.scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler'])

            print(f"Model loaded from {path}, beta={self.beta:.4f}, entropy_coef={self.entropy_coef:.4f}")
            return True
        else:
            print(f"Model file {path} not found, starting training from scratch.")
            return False

    def get_action_batch(self, states: List[Dict], insts: List[MOFJSPInstance], deterministic=False):
        batch_size = len(states)
        n_op_global = states[0]['op_feats'].shape[0]
        n_mac_global = states[0]['mac_feats'].shape[0]

        batch_state_dict, state_slices, batch_op_mask = batch_states(states, n_op_global, n_mac_global)

        # Force all features to float32
        batch_state_dict['op_feats'] = batch_state_dict['op_feats'].float()
        batch_state_dict['mac_feats'] = batch_state_dict['mac_feats'].float()
        for k in batch_state_dict['edge_feats']:
            if batch_state_dict['edge_feats'][k].numel() > 0:
                batch_state_dict['edge_feats'][k] = batch_state_dict['edge_feats'][k].float()

        with torch.no_grad():
            if hasattr(self.encoder, 'out_dim') and isinstance(self.encoder, HomogeneousGAT):
                num_op = batch_state_dict['op_feats'].shape[0]
                num_mac = batch_state_dict['mac_feats'].shape[0]
                all_feats = torch.cat([batch_state_dict['op_feats'], batch_state_dict['mac_feats']], dim=0)
                all_edges = []
                all_edge_feats = []
                src_seq = batch_state_dict['edges']['seq'][0]
                dst_seq = batch_state_dict['edges']['seq'][1]
                if src_seq.numel() > 0:
                    all_edges.append(torch.stack([src_seq, dst_seq]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['seq'])
                src_om = batch_state_dict['edges']['op_mac'][0]
                dst_om = batch_state_dict['edges']['op_mac'][1] + num_op
                if src_om.numel() > 0:
                    all_edges.append(torch.stack([src_om, dst_om]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['op_mac'])
                src_mo = batch_state_dict['edges']['mac_op'][0] + num_op
                dst_mo = batch_state_dict['edges']['mac_op'][1]
                if src_mo.numel() > 0:
                    all_edges.append(torch.stack([src_mo, dst_mo]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['mac_op'])
                # The competing operation pairs of the manuscript belong to the edge set of this
                # arm as well.  The arm ablates the typing of the edges and not the presence of an
                src_dj = batch_state_dict['edges']['disj'][0]
                dst_dj = batch_state_dict['edges']['disj'][1]
                if src_dj.numel() > 0:
                    all_edges.append(torch.stack([src_dj, dst_dj]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['disj'])

                if all_edges:
                    edge_index = torch.cat(all_edges, dim=1)
                    edge_feat = torch.cat(all_edge_feats, dim=0)
                else:
                    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                    edge_feat = torch.empty(0, 1, device=DEVICE)

                all_emb = self.encoder(all_feats, edge_index, edge_feat)
                h_op_all = all_emb[:num_op]
                h_mac_all = all_emb[num_op:]
            else:
                h_op_all, h_mac_all = self.encoder(batch_state_dict)

            # Check for NaN
            if torch.isnan(h_op_all).any() or torch.isnan(h_mac_all).any():
                raise ValueError("Encoder output contains NaN")

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
                if torch.isnan(g_emb).any():
                    raise ValueError("g_emb contains NaN after normalization")
                g_embs.append(g_emb)
                batch_h_op.append(h_op)
                batch_h_mac.append(h_mac)
                batch_op_masks.append(states[i]['op_mask'])

            g_embs = torch.stack(g_embs, dim=0)
            batch_h_op = torch.stack(batch_h_op, dim=0)
            batch_h_mac = torch.stack(batch_h_mac, dim=0)
            batch_op_masks = torch.stack(batch_op_masks, dim=0)

            if self.use_hierarchical:
                u_probs_batch = self.upper(g_embs, batch_h_op, batch_op_masks)

                if torch.isnan(u_probs_batch).any():
                    raise ValueError("u_probs_batch contains NaN")

                if deterministic:
                    op_indices = torch.argmax(u_probs_batch, dim=1)
                else:
                    dist = torch.distributions.Categorical(u_probs_batch)
                    op_indices = dist.sample()

                selected_op_embs = batch_h_op[torch.arange(batch_size), op_indices]

                batch_mac_masks_op = torch.zeros((batch_size, n_mac_global), device=DEVICE)
                for i in range(batch_size):
                    inst_i = insts[i]
                    op_idx_local = op_indices[i]
                    if op_idx_local < inst_i.total_ops:
                        batch_mac_masks_op[i, :inst_i.n_machines] = inst_i.op_mac_mask[op_idx_local]

                # Fixed-preference heuristic term of the manuscript's lower policy (Eq. 10).  The
                # processing times come from the instance, the load and the availability of a
                mac_feats_batch = batch_state_dict['mac_feats'].reshape(batch_size,
                                                                    n_mac_global, -1)
                pt_rows = torch.zeros((batch_size, n_mac_global), device=DEVICE)
                for i in range(batch_size):
                    inst_i = insts[i]
                    op_idx_local = op_indices[i]
                    if op_idx_local < inst_i.total_ops:
                        pt_rows[i, :inst_i.n_machines] = inst_i.pt_rows_norm[op_idx_local]
                lower_heuristic = -(W1 * pt_rows + W2 * mac_feats_batch[:, :, 0]
                                    + W3 * mac_feats_batch[:, :, 1])

                l_probs_batch = self.lower(g_embs, selected_op_embs, batch_h_mac,
                                          batch_mac_masks_op, lower_heuristic)

                if torch.isnan(l_probs_batch).any():
                    raise ValueError("l_probs_batch contains NaN")

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
            else:
                mac_mask_per_op = torch.zeros((batch_size, n_op_global, n_mac_global), device=DEVICE)
                for i in range(batch_size):
                    inst_i = insts[i]
                    for op_idx in range(inst_i.total_ops):
                        mac_mask_per_op[i, op_idx, :inst_i.n_machines] = inst_i.op_mac_mask[op_idx]

                probs_flat = self.joint_policy(g_embs, batch_h_op, batch_h_mac, batch_op_masks, mac_mask_per_op)

                if torch.isnan(probs_flat).any():
                    raise ValueError("probs_flat contains NaN")

                # Safety check: ensure probs_flat shape matches expectation
                expected_num_actions = n_op_global * n_mac_global
                if probs_flat.shape[1] != expected_num_actions:
                    print(f"Warning: probs_flat shape {probs_flat.shape} does not match expected {expected_num_actions}, using actual shape for computation")
                    # Recompute n_op_global and n_mac_global (round)
                    actual_num_actions = probs_flat.shape[1]
                    if actual_num_actions % n_mac_global != 0:
                        raise ValueError(f"probs_flat columns {actual_num_actions} not divisible by n_mac_global {n_mac_global}")
                    n_op_global = actual_num_actions // n_mac_global

                # After sampling, strictly validate action validity
                max_attempts = 5
                for attempt in range(max_attempts):
                    if deterministic:
                        flat_indices = torch.argmax(probs_flat, dim=1)
                    else:
                        dist = torch.distributions.Categorical(probs_flat)
                        flat_indices = dist.sample()

                    op_indices = flat_indices // n_mac_global
                    mac_indices = flat_indices % n_mac_global

                    # Validity check
                    valid = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
                    for i in range(batch_size):
                        inst_i = insts[i]
                        op_ok = (op_indices[i] < inst_i.total_ops)
                        mac_ok = (mac_indices[i] < inst_i.n_machines)
                        mask_ok = False
                        if op_ok and mac_ok:
                            mask_ok = (inst_i.op_mac_mask[op_indices[i], mac_indices[i]] > 0.5)
                        valid[i] = op_ok and mac_ok and mask_ok

                    if valid.all():
                        break  # All actions valid, exit retry loop
                    else:
                        # Log details of invalid actions
                        for i in range(batch_size):
                            if not valid[i]:
                                print(f"Invalid action: env {i}, instance {insts[i].file_name}, "
                                      f"op_idx={op_indices[i]}, mac_idx={mac_indices[i]}, "
                                      f"op_ok={op_indices[i] < insts[i].total_ops}, "
                                      f"mac_ok={mac_indices[i] < insts[i].n_machines}, "
                                      f"mask_ok={mask_ok if op_ok and mac_ok else False}")
                                # Print current op_mask and mac_mask_per_op for debugging
                                print(f"op_mask[{i}]: {states[i]['op_mask'].cpu().numpy()}")
                                print(f"mac_mask_per_op[{i}, {op_indices[i]}]: {mac_mask_per_op[i, op_indices[i]].cpu().numpy()}")

                        if attempt == max_attempts - 1:
                            # Still invalid on last attempt, throw exception
                            error_msg = f"Cannot sample a valid action after {max_attempts} attempts, please check state representation."
                            raise RuntimeError(error_msg)

                        # Zero out probabilities for invalid actions and renormalize
                        for i in range(batch_size):
                            if not valid[i]:
                                probs_flat[i, flat_indices[i]] = 0.0
                        # Renormalize
                        row_sums = probs_flat.sum(dim=1, keepdim=True)
                        probs_flat = torch.where(row_sums > 0, probs_flat / row_sums, torch.ones_like(probs_flat) / probs_flat.shape[1])

                actions = []
                probs_list = []
                indices_list = []
                for i in range(batch_size):
                    actions.append((op_indices[i], mac_indices[i]))
                    probs_list.append(probs_flat[i, flat_indices[i]])
                    indices_list.append(flat_indices[i])

        return actions, probs_list, indices_list

    def update(self, trajectory, inst):
        T = len(trajectory)
        if T == 0:
            return None

        states = [t['state'] for t in trajectory]
        rewards = torch.stack([t['reward'] for t in trajectory]).to(DEVICE)

        if self.use_hierarchical:
            old_u_probs = torch.stack([t['old_probs'][0] for t in trajectory]).detach()
            old_l_probs = torch.stack([t['old_probs'][1] for t in trajectory]).detach()
            op_indices = torch.stack([t['indices'][0] for t in trajectory])
            mac_indices = torch.stack([t['indices'][1] for t in trajectory])
        else:
            old_probs = torch.stack([t['old_probs'] for t in trajectory]).detach()
            flat_indices = torch.stack([t['indices'] for t in trajectory])

        n_op_global = states[0]['op_feats'].shape[0]
        n_mac_global = states[0]['mac_feats'].shape[0]

        chunk_size = 16

        # Advantage calculation
        with torch.no_grad():
            batch_state_dict, state_slices, _ = batch_states(states, n_op_global, n_mac_global)

            # Force all features to float32
            batch_state_dict['op_feats'] = batch_state_dict['op_feats'].float()
            batch_state_dict['mac_feats'] = batch_state_dict['mac_feats'].float()
            for k in batch_state_dict['edge_feats']:
                if batch_state_dict['edge_feats'][k].numel() > 0:
                    batch_state_dict['edge_feats'][k] = batch_state_dict['edge_feats'][k].float()

            if hasattr(self.encoder, 'out_dim') and isinstance(self.encoder, HomogeneousGAT):
                num_op = batch_state_dict['op_feats'].shape[0]
                num_mac = batch_state_dict['mac_feats'].shape[0]
                all_feats = torch.cat([batch_state_dict['op_feats'], batch_state_dict['mac_feats']], dim=0)
                all_edges = []
                all_edge_feats = []
                src_seq = batch_state_dict['edges']['seq'][0]
                dst_seq = batch_state_dict['edges']['seq'][1]
                if src_seq.numel() > 0:
                    all_edges.append(torch.stack([src_seq, dst_seq]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['seq'])
                src_om = batch_state_dict['edges']['op_mac'][0]
                dst_om = batch_state_dict['edges']['op_mac'][1] + num_op
                if src_om.numel() > 0:
                    all_edges.append(torch.stack([src_om, dst_om]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['op_mac'])
                src_mo = batch_state_dict['edges']['mac_op'][0] + num_op
                dst_mo = batch_state_dict['edges']['mac_op'][1]
                if src_mo.numel() > 0:
                    all_edges.append(torch.stack([src_mo, dst_mo]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['mac_op'])
                # The competing operation pairs of the manuscript belong to the edge set of this
                # arm as well.  The arm ablates the typing of the edges and not the presence of an
                src_dj = batch_state_dict['edges']['disj'][0]
                dst_dj = batch_state_dict['edges']['disj'][1]
                if src_dj.numel() > 0:
                    all_edges.append(torch.stack([src_dj, dst_dj]))
                    all_edge_feats.append(batch_state_dict['edge_feats']['disj'])

                if all_edges:
                    edge_index = torch.cat(all_edges, dim=1)
                    edge_feat = torch.cat(all_edge_feats, dim=0)
                else:
                    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                    edge_feat = torch.empty(0, 1, device=DEVICE)

                all_emb = self.encoder(all_feats, edge_index, edge_feat)
                h_op_all = all_emb[:num_op]
                h_mac_all = all_emb[num_op:]
            else:
                h_op_all, h_mac_all = self.encoder(batch_state_dict)

            g_embs = []
            for i in range(T):
                op_start, op_end, mac_start, mac_end = state_slices[i]
                h_op = h_op_all[op_start:op_end]
                g_emb = h_op.mean(dim=0)
                g_embs.append(g_emb)
            g_embs = torch.stack(g_embs, dim=0)
            if self.state_norm is not None:
                g_embs = self.state_norm.normalize(g_embs)

            values = self.critic_u(g_embs).squeeze()
            next_values = torch.cat([values[1:], torch.zeros(1, device=DEVICE)])
            deltas = rewards + self.gamma * next_values - values
            factor = self.gamma * self.lam
            rev_deltas = deltas.flip(0)
            exp_factors = factor ** torch.arange(T, device=DEVICE)
            rev_advantages = torch.cumsum(rev_deltas * exp_factors, dim=0) / exp_factors
            advantages = rev_advantages.flip(0)
            returns = values + advantages

            adv_std = advantages.std()
            if adv_std < 1e-8:
                advantages = advantages - advantages.mean()
            else:
                advantages = (advantages - advantages.mean()) / (adv_std + EPS)
            advantages = torch.clamp(advantages, -10.0, 10.0)
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
            advantages = advantages.detach()
            returns = returns.detach()

            # `values` is kept for the value-clip branch of the chunk loop below, for the reason
            # recorded in PPO_HGAT.py: deleting it made the clip unusable and the arms then trained
            # without a bound on the value term.
            del h_op_all, h_mac_all, batch_state_dict, state_slices, g_embs, next_values, deltas, rev_deltas, rev_advantages
            # The lower level of every arm is updated with the advantage of the upper level, exactly
            # as the full model of this method is, so an arm and the full model share one trainer.
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        total_kl = torch.tensor(0.0, device=DEVICE)
        total_policy_loss_u = torch.tensor(0.0, device=DEVICE)
        total_policy_loss_l = torch.tensor(0.0, device=DEVICE)
        total_value_loss = torch.tensor(0.0, device=DEVICE)
        valid_chunks = 0

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_len = chunk_end - chunk_start

            chunk_states = states[chunk_start:chunk_end]
            chunk_advantages = advantages[chunk_start:chunk_end]
            chunk_returns = returns[chunk_start:chunk_end]

            if self.use_hierarchical:
                chunk_op_indices = op_indices[chunk_start:chunk_end]
                chunk_mac_indices = mac_indices[chunk_start:chunk_end]
                chunk_old_u_probs = old_u_probs[chunk_start:chunk_end]
                chunk_old_l_probs = old_l_probs[chunk_start:chunk_end]
            else:
                chunk_flat_indices = flat_indices[chunk_start:chunk_end]
                chunk_old_probs = old_probs[chunk_start:chunk_end]

            chunk_batch_state_dict, chunk_state_slices, _ = batch_states(chunk_states, n_op_global, n_mac_global)

            # Force all features to float32
            chunk_batch_state_dict['op_feats'] = chunk_batch_state_dict['op_feats'].float()
            chunk_batch_state_dict['mac_feats'] = chunk_batch_state_dict['mac_feats'].float()
            for k in chunk_batch_state_dict['edge_feats']:
                if chunk_batch_state_dict['edge_feats'][k].numel() > 0:
                    chunk_batch_state_dict['edge_feats'][k] = chunk_batch_state_dict['edge_feats'][k].float()

            if hasattr(self.encoder, 'out_dim') and isinstance(self.encoder, HomogeneousGAT):
                num_op = chunk_batch_state_dict['op_feats'].shape[0]
                num_mac = chunk_batch_state_dict['mac_feats'].shape[0]
                all_feats = torch.cat([chunk_batch_state_dict['op_feats'], chunk_batch_state_dict['mac_feats']], dim=0)
                all_edges = []
                all_edge_feats = []
                src_seq = chunk_batch_state_dict['edges']['seq'][0]
                dst_seq = chunk_batch_state_dict['edges']['seq'][1]
                if src_seq.numel() > 0:
                    all_edges.append(torch.stack([src_seq, dst_seq]))
                    all_edge_feats.append(chunk_batch_state_dict['edge_feats']['seq'])
                src_om = chunk_batch_state_dict['edges']['op_mac'][0]
                dst_om = chunk_batch_state_dict['edges']['op_mac'][1] + num_op
                if src_om.numel() > 0:
                    all_edges.append(torch.stack([src_om, dst_om]))
                    all_edge_feats.append(chunk_batch_state_dict['edge_feats']['op_mac'])
                src_mo = chunk_batch_state_dict['edges']['mac_op'][0] + num_op
                dst_mo = chunk_batch_state_dict['edges']['mac_op'][1]
                if src_mo.numel() > 0:
                    all_edges.append(torch.stack([src_mo, dst_mo]))
                    all_edge_feats.append(chunk_batch_state_dict['edge_feats']['mac_op'])
                # The competing operation pairs of the manuscript belong to the edge set of this
                # arm as well.  The arm ablates the typing of the edges and not the presence of an
                src_dj = chunk_batch_state_dict['edges']['disj'][0]
                dst_dj = chunk_batch_state_dict['edges']['disj'][1]
                if src_dj.numel() > 0:
                    all_edges.append(torch.stack([src_dj, dst_dj]))
                    all_edge_feats.append(chunk_batch_state_dict['edge_feats']['disj'])

                if all_edges:
                    edge_index = torch.cat(all_edges, dim=1)
                    edge_feat = torch.cat(all_edge_feats, dim=0)
                else:
                    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                    edge_feat = torch.empty(0, 1, device=DEVICE)

                all_emb = self.encoder(all_feats, edge_index, edge_feat)
                h_op_all = all_emb[:num_op]
                h_mac_all = all_emb[num_op:]
            else:
                h_op_all, h_mac_all = self.encoder(chunk_batch_state_dict)

            g_embs_chunk = []
            for i in range(chunk_len):
                op_start, op_end, mac_start, mac_end = chunk_state_slices[i]
                h_op = h_op_all[op_start:op_end]
                g_emb = h_op.mean(dim=0)
                g_embs_chunk.append(g_emb)
            g_embs_chunk = torch.stack(g_embs_chunk, dim=0)
            if self.state_norm is not None:
                g_embs_chunk = self.state_norm.normalize(g_embs_chunk)

            batch_h_op = []
            batch_h_mac = []
            batch_op_masks = []
            for i in range(chunk_len):
                op_start, op_end, mac_start, mac_end = chunk_state_slices[i]
                h_op = h_op_all[op_start:op_end]
                h_mac = h_mac_all[mac_start:mac_end]
                batch_h_op.append(h_op)
                batch_h_mac.append(h_mac)
                batch_op_masks.append(chunk_states[i]['op_mask'])

            batch_h_op = torch.stack(batch_h_op, dim=0)
            batch_h_mac = torch.stack(batch_h_mac, dim=0)
            batch_op_masks = torch.stack(batch_op_masks, dim=0)

            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                if self.use_hierarchical:
                    curr_u_probs_batch = self.upper(g_embs_chunk, batch_h_op, batch_op_masks)
                    curr_u_probs = curr_u_probs_batch[torch.arange(chunk_len), chunk_op_indices]
                    curr_u_probs = torch.clamp(curr_u_probs, EPS, 1.0 - EPS)
                    chunk_old_u_probs_clamped = torch.clamp(chunk_old_u_probs, EPS, 1.0 - EPS)

                    batch_mac_masks_op = torch.zeros((chunk_len, n_mac_global), device=DEVICE)
                    for j in range(chunk_len):
                        idx = chunk_start + j
                        if op_indices[idx] < inst.total_ops:
                            batch_mac_masks_op[j, :inst.n_machines] = inst.op_mac_mask[op_indices[idx]]

                    # The same heuristic term rebuilt from the stored states: an update that
                    # omitted it would score the sampled machines under a different policy than
                    # the one they were sampled from and would corrupt the PPO ratio.
                    chunk_mac_feats = chunk_batch_state_dict['mac_feats'].reshape(chunk_len,
                                                                                n_mac_global,
                                                                                -1)
                    chunk_pt_rows = torch.zeros((chunk_len, n_mac_global), device=DEVICE)
                    for j in range(chunk_len):
                        if chunk_op_indices[j] < inst.total_ops:
                            chunk_pt_rows[j, :inst.n_machines] = inst.pt_rows_norm[chunk_op_indices[j]]
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

                    kl_u = (chunk_old_u_probs_clamped * (torch.log(chunk_old_u_probs_clamped + EPS) - torch.log(curr_u_probs + EPS))).mean()
                    kl_l = (chunk_old_l_probs_clamped * (torch.log(chunk_old_l_probs_clamped + EPS) - torch.log(curr_l_probs + EPS))).mean()
                    kl = kl_u + kl_l
                    kl = torch.clamp(kl, min=0.0, max=10.0)

                    policy_loss = policy_loss_u + policy_loss_l
                else:
                    mac_mask_per_op = torch.zeros((chunk_len, n_op_global, n_mac_global), device=DEVICE)
                    for j in range(chunk_len):
                        inst_j = inst
                        for op_idx in range(inst_j.total_ops):
                            mac_mask_per_op[j, op_idx, :inst_j.n_machines] = inst_j.op_mac_mask[op_idx]

                    curr_probs_flat = self.joint_policy(g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks, mac_mask_per_op)
                    curr_probs = curr_probs_flat[torch.arange(chunk_len), chunk_flat_indices]
                    curr_probs = torch.clamp(curr_probs, EPS, 1.0 - EPS)
                    chunk_old_probs_clamped = torch.clamp(chunk_old_probs, EPS, 1.0 - EPS)

                    ratio = curr_probs / (chunk_old_probs_clamped + EPS)
                    ratio = torch.clamp(ratio, 0.1, 10.0)

                    surr1 = ratio * chunk_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * chunk_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    kl = (chunk_old_probs_clamped * (torch.log(chunk_old_probs_clamped + EPS) - torch.log(curr_probs + EPS))).mean()
                    kl = torch.clamp(kl, min=0.0, max=10.0)

                    policy_loss_u = policy_loss
                    policy_loss_l = 0.0

                policy_loss_u = policy_loss_u if isinstance(policy_loss_u, torch.Tensor) else torch.tensor(policy_loss_u, device=DEVICE)
                policy_loss_l = policy_loss_l if isinstance(policy_loss_l, torch.Tensor) else torch.tensor(policy_loss_l, device=DEVICE)
                kl = kl if isinstance(kl, torch.Tensor) else torch.tensor(kl, device=DEVICE)

                v_pred = self.critic_u(g_embs_chunk.float()).squeeze()
                if v_pred.dim() == 0:
                    v_pred = v_pred.unsqueeze(0)
                chunk_returns_f32 = chunk_returns.float()
                if chunk_returns_f32.dim() == 0:
                    chunk_returns_f32 = chunk_returns_f32.unsqueeze(0)

                try:
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
                except Exception as e:
                    print(f"Error computing value_loss: {e}, skipping this chunk.")
                    del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue

                if 'value_loss' not in locals():
                    print("value_loss not defined, skipping this chunk.")
                    del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue

                value_loss = value_loss if isinstance(value_loss, torch.Tensor) else torch.tensor(value_loss, device=DEVICE)

                if self.use_hierarchical:
                    entropy_u = -(curr_u_probs * torch.log(curr_u_probs + EPS)).mean()
                    entropy_l = -(curr_l_probs * torch.log(curr_l_probs + EPS)).mean()
                    entropy = entropy_u + entropy_l
                else:
                    entropy = -(curr_probs * torch.log(curr_probs + EPS)).mean()

                total_loss = (policy_loss + self.value_coef * value_loss
                              + self.beta * kl - self.entropy_coef * entropy)

                if torch.abs(total_loss) > 1e8:
                    print(f"Total loss too large ({total_loss.item():.2e}), skipping this chunk.")
                    del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: Loss is NaN/Inf in chunk {chunk_start}-{chunk_end}, skipping this chunk.")
                    del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue

            self.opt.zero_grad()
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Gradient norm is NaN/Inf, skipping this chunk update.")
                    self.opt.zero_grad()
                else:
                    self.scaler.step(self.opt)
                # update() has to close every unscale_() call: skipping it here would make the
                # next chunk of this optimiser raise "unscale_() has already been called".
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Gradient norm is NaN/Inf, skipping this chunk update.")
                    self.opt.zero_grad()
                else:
                    self.opt.step()

            total_kl += kl.detach() * chunk_len
            total_policy_loss_u += policy_loss_u.detach() * chunk_len
            total_policy_loss_l += policy_loss_l.detach() * chunk_len
            total_value_loss += value_loss.detach() * chunk_len
            valid_chunks += 1

            del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            self.update_counter += 1
            if self.update_counter % 10 == 0 and DEVICE.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        if valid_chunks == 0:
            print("Warning: All update chunks were skipped, no valid update performed.")
            return None

        avg_kl = total_kl / T
        avg_policy_loss_u = total_policy_loss_u / T
        avg_policy_loss_l = total_policy_loss_l / T
        avg_value_loss = total_value_loss / T

        if self.kl_ema is None:
            self.kl_ema = avg_kl.item()
        else:
            self.kl_ema = self.kl_alpha * self.kl_ema + (1 - self.kl_alpha) * avg_kl.item()

        kl_error = self.kl_ema - self.cfg.kl_target
        self.beta *= (1 + self.beta_lr * kl_error)
        # The floor is a config value, see beta_floor: a coefficient at 0.01 leaves the trust region
        # of the arms inactive, which is the state the curves of the previous wave were trained in.
        self.beta = np.clip(self.beta, float(getattr(self.cfg, 'beta_floor', 0.01)), 10.0)

        self.entropy_coef *= self.cfg.entropy_decay
        self.entropy_coef = max(self.entropy_coef, getattr(self.cfg, 'entropy_min', 0.0))

        def ensure_tensor(x):
            if not isinstance(x, torch.Tensor):
                return torch.tensor(x, device=DEVICE)
            return x

        stats = {
            'policy_loss_u': ensure_tensor(avg_policy_loss_u),
            'policy_loss_l': ensure_tensor(avg_policy_loss_l),
            'value_loss': ensure_tensor(avg_value_loss),
            'kl': ensure_tensor(avg_kl),
            'beta': ensure_tensor(self.beta),
            'advantage': ensure_tensor(advantages.mean()),
            'return': ensure_tensor(returns.mean()),
            'total_loss': ensure_tensor(avg_policy_loss_u + avg_policy_loss_l + self.value_coef * avg_value_loss + self.beta * avg_kl)
        }
        return stats


# Training function (for a single experiment). Returns True when the experiment finished and a
# checkpoint was written, False when the experiment was skipped or aborted by an error.
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


def evaluate_validation(agent, validation, max_jobs, max_total_ops, max_machines,
                        max_proc_time, max_due_date, seed=0):
    """Greedy rollouts of the validation instances; returns [Cmax, LB, Tardy] per instance.

    Greedy decoding is the canonical inference of the policy and it dominates the sampled
    trajectories, so the validation objective is measured on it instead of on a sample.
    """
    agent.encoder.eval()
    objectives = []
    # One deterministic rollout plus k-1 sampled ones, the best of the set scores the arm: the
    # protocol the comparison and the ablation evaluator report every method with.  Selecting on
    VALIDATION_ROLLOUTS = 4
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
            if best is None or weighted_objective(candidate) < weighted_objective(best):
                best = candidate
        objectives.append(best)
    agent.encoder.train()
    return objectives


def train_experiment(exp_config):
    print("=" * 50)
    print(f"Starting training for experiment: {exp_config['name']}")
    print("=" * 50)

    # The width comes from the experiment definition and is restored afterwards, so that training
    # one variant can never leak its own network size into another variant.
    original_dims = (cfg.gat_hidden_dim, cfg.gat_out_dim, cfg.policy_hidden_dim)
    apply_experiment_dims(exp_config)
    print(f"Network width of {exp_config['name']}: gat_hidden_dim={cfg.gat_hidden_dim}, "
          f"gat_out_dim={cfg.gat_out_dim}, policy_hidden_dim={cfg.policy_hidden_dim}")

    try:
        check_training_data()

        model_dir = exp_config['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best.pth")
        # Rolling checkpoint of an interrupted run: it also carries the optimiser
        # moments and the episode counter, so a restart continues instead of
        # restarting from the last best episode.
        latest_path = os.path.join(model_dir, "latest.pth")

        if exp_config.get('pretrained_path') and os.path.exists(exp_config['pretrained_path']):
            if not os.path.exists(model_path):
                shutil.copy(exp_config['pretrained_path'], model_path)
                print(f"A5 pre-trained model copied to {model_path}, skipping training.")
            else:
                print(f"A5 model already exists, skipping training.")
            return

        # resume_existing=True continues an interrupted run from its checkpoint instead of
        # skipping the experiment; the default keeps the original skipping behaviour.
        resume_existing = bool(exp_config.get('resume_existing', False))
        if os.path.exists(model_path) and not resume_existing:
            print(f"Model file already exists: {model_path}, skipping training.")
            return

        instance_tuples, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = \
            load_all_instances(TRAIN_INSTANCE_DIR, TRAIN_PATTERN)
        print(f"Loaded {len(instance_tuples)} training instances")
        print(f"Global max jobs: {max_jobs}, max machines: {max_machines}, max total operations: {max_total_ops}, max processing time: {max_proc_time}, max due date: {max_due_date}, max machine capability: {max_capability}")

        instance_list = []
        for jobs, caps, dues, fname, size in instance_tuples:
            inst = MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability)
            instance_list.append(inst)

        curriculum_sampler = CurriculumSampler(instance_list, cfg)
        # The reported weights are selected on a fixed validation set, one that the curriculum
        # sampler cannot move: the sampler changes the instance distribution over the run, so the
        val_tuples = load_all_instances(TRAIN_INSTANCE_DIR, VAL_PATTERN)[0]
        pool_tuples = load_all_instances(TRAIN_INSTANCE_DIR, TRAIN_PATTERN)[0]
        validation_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time,
                                               max_capability)
                                for (jobs, caps, dues, fname, size) in val_tuples]
        pool_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time,
                                               max_capability)
                                for (jobs, caps, dues, fname, size) in pool_tuples]

        validation_set = build_validation_set(validation_instances, per_size=8)
        # Two instances of every size group of the training pool next to the eight held-out ones:
        # the wave that selected on the held-out split alone produced checkpoints that measured
        print(f"Validation set: {[inst.file_name for inst in validation_set]}")

        global_max_ops = max(inst.total_ops for inst in instance_list)
        global_max_machines = max(inst.n_machines for inst in instance_list)
        print(f"Global max operations: {global_max_ops}, global max machines: {global_max_machines}")

        dim_op, dim_mac = 3, 3

        if exp_config['use_homogeneous']:
            encoder = HomogeneousGAT(
                in_dim=dim_op,
                hidden_dim=cfg.gat_hidden_dim,
                out_dim=cfg.gat_out_dim,
                num_layers=2,
                num_heads=4,
                use_edge_feat=exp_config['use_edge_feat']
            ).to(DEVICE)
        elif exp_config['use_mlp_encoder']:
            encoder = MLPEncoder(dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim).to(DEVICE)
        else:
            encoder = HeteroGAT(dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim,
                                use_edge_feat=exp_config['use_edge_feat']).to(DEVICE)

        if exp_config['use_hierarchical']:
            upper = UpperPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
            lower = LowerPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
            critic_u = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
            critic_l = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
            joint_policy = None
        else:
            upper = None
            lower = None
            critic_u = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
            critic_l = None
            joint_policy = JointPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim,
                                       global_max_ops, global_max_machines).to(DEVICE)

        state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=DEVICE)

        dummy_inst = instance_list[0]
        agent = PPOAgent(dummy_inst, encoder, upper, lower, critic_u, critic_l, cfg,
                         state_norm, joint_policy, use_hierarchical=exp_config['use_hierarchical'])

        resume_source = latest_path if os.path.exists(latest_path) else model_path
        restored_best = -float('inf')
        restored_episodes = 0
        if os.path.exists(resume_source):
            agent.load(resume_source)
            metadata = getattr(agent, 'loaded_metadata', {}) or {}
            if metadata.get('best_norm_reward') is not None:
                restored_best = float(metadata['best_norm_reward'])
            if metadata.get('episode_counter') is not None:
                restored_episodes = int(metadata['episode_counter'] or 0)
            agent.episode_counter = restored_episodes
            print(f"Loaded {resume_source}, continuing training "
                  f"(episode_counter={restored_episodes}, best_norm_reward={restored_best}).")
        else:
            print(f"No existing model found, starting training from scratch.")

        # Adjust number of parallel environments based on experiment type
        if exp_config['use_hierarchical']:
            num_envs = cfg.num_envs
        else:
            num_envs = 1  # Single environment for joint policy

        envs = []
        env_insts = []
        for _ in range(num_envs):
            inst = curriculum_sampler.sample()
            env = HeteroGraphEnv(inst, max_jobs, global_max_ops, global_max_machines,
                                 max_proc_time, max_due_date, cfg)
            envs.append(env)
            env_insts.append(inst)

        batch_env = BatchEnv(envs)
        states = batch_env.reset()

        traj_buffers = [[] for _ in range(num_envs)]
        ep_rewards = [torch.tensor(0.0, device=DEVICE) for _ in range(num_envs)]
        step_counts = [0 for _ in range(num_envs)]

        # Only the missing episodes are trained: the counter of the rolling
        # checkpoint is restored so that a restart does not repeat the run.
        total_episodes = min(restored_episodes, cfg.n_episodes)
        normalized_rewards = []
        kl_episodes = []
        loss_episodes = []
        best_norm_reward = restored_best
        reward_ema = None
        # Validation bookkeeping of the interrupted run: the curve continues and the reference of
        # every validation instance is kept, so the score stays comparable across a restart.
        resumed_metadata = getattr(agent, 'loaded_metadata', {}) or {}
        val_curve = [tuple(entry) for entry in (resumed_metadata.get('val_curve') or [])]
        val_reference = list(resumed_metadata.get('val_reference') or []) or None
        best_val = float(resumed_metadata.get('best_val') or float('inf'))
        # The patience of the early stop survives a restart.  Without this rebuild every restart
        # of the queue opened a fresh patience window and an arm that had already reached its
        stale_checks = 0
        running_best = float('inf')
        for _episode, _score in val_curve:
            if _score < running_best * (1.0 - cfg.val_tol):
                running_best = _score
                stale_checks = 0
            else:
                stale_checks += 1
        # An early stop is the decision of the run, not an interruption of it, so the flag of the
        # rolling checkpoint is restored as well: an arm that already reached its own plateau must
        # not be trained through another patience window past its peak.
        converged = bool(resumed_metadata.get('converged', False))

        # Reference of the validation objective.  It is the weighted objective of a deterministic
        # earliest-completion dispatch of the same instance (dispatch_reference.py) instead of the
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from dispatch_reference import reference_objectives
        val_reference = [weighted_objective(obj) for obj in reference_objectives(
            [inst.file_name for inst in validation_set], TRAIN_INSTANCE_DIR)]
        # A curve written by the earlier frame holds a ratio to the first check of that run and
        # is not comparable with this one, so the arm starts a fresh curve instead of mixing.
        if val_curve and abs(val_curve[0][1] - 1.0) < 1e-12:
            print("The restored validation curve is in the old frame and is reset.")
            val_curve = []
            best_val = float('inf')
            stale_checks = 0

        pbar = tqdm(total=cfg.n_episodes, initial=total_episodes,
                    desc=f"Training {exp_config['name']}", unit="ep")

        log_interval = 100
        empty_cache_interval = 50
        # Episodes between two rolling checkpoints of an interrupted run.
        checkpoint_interval = max(1, int(getattr(cfg, 'checkpoint_interval', 100)))

        while total_episodes < cfg.n_episodes and not converged:
            # Fixed curriculum scale: see Config.curriculum_episodes.
            progress = min(1.0, total_episodes / max(1, cfg.curriculum_episodes))
            curriculum_sampler.update_stage(progress)

            next_states = states
            try:
                actions, probs_list, indices_list = agent.get_action_batch(states, env_insts)
                next_states, rewards, dones = batch_env.step(actions)
            except Exception as e:
                print(f"Error during training step: {e}, resetting all environments and continuing.")
                for j in range(num_envs):
                    new_inst = curriculum_sampler.sample()
                    envs[j] = HeteroGraphEnv(new_inst, max_jobs, global_max_ops, global_max_machines,
                                             max_proc_time, max_due_date, cfg)
                    batch_env.envs[j] = envs[j]
                    env_insts[j] = new_inst
                    states[j] = envs[j].reset()
                    traj_buffers[j] = []
                    ep_rewards[j] = torch.tensor(0.0, device=DEVICE)
                    step_counts[j] = 0
                continue

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
                    ep_reward_val = ep_rewards[i].item()
                    norm_reward = ep_reward_val / envs[i].inst.total_ops
                    normalized_rewards.append(norm_reward)
                    total_episodes += 1
                    pbar.update(1)

                    if traj_buffers[i]:
                        try:
                            stats = agent.update(traj_buffers[i], env_insts[i])
                        except Exception as e:
                            print(f"Error during model update: {e}, skipping this update.")
                            stats = None
                        if stats is not None:
                            kl_episodes.append(stats['kl'].item())
                            loss_episodes.append(stats['total_loss'].item())
                        else:
                            kl_episodes.append(0.0)
                            loss_episodes.append(0.0)
                    else:
                        stats = None
                        kl_episodes.append(0.0)
                        loss_episodes.append(0.0)

                    # The scheduler advances once per episode so that its own counter reaches the step
                    # size after that many episodes.  Stepping it only on the multiples of the step size
                    agent.scheduler.step()
                    for param_group in agent.opt.param_groups:
                        param_group['lr'] = max(param_group['lr'], cfg.min_lr)

                    if total_episodes % log_interval == 0 and stats is not None:
                        lr = agent.opt.param_groups[0]['lr']
                        beta_val = stats['beta'].item()
                        adv_val = stats['advantage'].item()
                        kl_val = stats['kl'].item()
                        print(f"\nEpisode {total_episodes}: LR={lr:.2e}, Beta={beta_val:.3f}, "
                              f"Adv={adv_val:.4f}, KL={kl_val:.4f}, "
                              f"Entropy={agent.entropy_coef:.4f}, AllowedSizes={curriculum_sampler.allowed_sizes}")

                    # The training curve is smoothed over a window instead of being compared
                    # episode by episode: one episode is a noisy sample of the policy.
                    if len(normalized_rewards) >= cfg.select_window:
                        best_norm_reward = max(best_norm_reward,
                                               float(np.mean(normalized_rewards[-cfg.select_window:])))

                    # Validation block.  It selects the reported weights and it stops the run as
                    # soon as the average objective stops improving, i.e. the run is never trained
                    # past the point where the policy keeps getting better.
                    if (total_episodes >= cfg.min_episodes
                            and total_episodes % max(1, cfg.val_interval) == 0):
                        objectives = evaluate_validation(
                            agent, validation_set, max_jobs, global_max_ops, global_max_machines,
                            max_proc_time, max_due_date, seed=int(getattr(cfg, 'val_seed', 0)))
                        scores = [weighted_objective(obj) for obj in objectives]
                        if val_reference is None:
                            # The first check defines the reference of every instance, so the
                            # score is a relative objective that does not depend on the instance
                            # size which dominates the raw weighted value.
                            val_reference = [value if abs(value) > 1e-9 else 1.0
                                             for value in scores]
                        val_score = float(np.mean([value / ref
                                                   for value, ref in zip(scores, val_reference)]))
                        val_curve.append((total_episodes, val_score))
                        print(f"Validation {len(val_curve):3d} at episode {total_episodes}: "
                              f"relative objective {val_score:.6f} (best {best_val:.6f})")
                        if val_score < best_val * (1.0 - cfg.val_tol):
                            best_val = val_score
                            stale_checks = 0
                            agent.save(model_path, extra={
                                'best_norm_reward': float(best_norm_reward),
                                'val_curve': [tuple(entry) for entry in val_curve],
                                'val_reference': list(val_reference),
                                'best_val': float(best_val),
                                'converged': False})
                        else:
                            stale_checks += 1
                            if stale_checks >= cfg.val_patience:
                                print(f"Early stop after {total_episodes} episodes: the "
                                      f"validation objective did not improve for "
                                      f"{stale_checks} checks (best={best_val:.6f}).")
                                converged = True

                    # Rolling checkpoint: it bounds the work a restart repeats to
                    # checkpoint_interval episodes.
                    if total_episodes % checkpoint_interval == 0:
                        agent.episode_counter = total_episodes
                        agent.save(latest_path, extra={
                            'best_norm_reward': float(best_norm_reward),
                            'val_curve': [tuple(entry) for entry in val_curve],
                            'val_reference': list(val_reference) if val_reference else [],
                            'best_val': float(best_val),
                            'converged': False})

                    if reward_ema is None:
                        reward_ema = norm_reward
                    else:
                        reward_ema = 0.9 * reward_ema + 0.1 * norm_reward

                    pbar.set_postfix({
                        'NormR': f"{norm_reward:.4f}",
                        'KL': f"{stats['kl'].item() if stats else 'N/A':.4f}" if stats else 'N/A',
                        'Beta': f"{stats['beta'].item() if stats else 'N/A':.3f}" if stats else 'N/A',
                        'L(θ)': f"{stats['total_loss'].item() if stats else 'N/A':.3f}" if stats else 'N/A',
                        'Steps': step_counts[i],
                        'Sizes': f"{curriculum_sampler.allowed_sizes}"
                    })

                    new_inst = curriculum_sampler.sample()
                    envs[i] = HeteroGraphEnv(new_inst, max_jobs, global_max_ops, global_max_machines,
                                             max_proc_time, max_due_date, cfg)
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

        # The finished run is checkpointed as well: a later restart of the same
        # experiment finds every episode already trained.
        agent.episode_counter = total_episodes
        agent.save(latest_path, extra={
            'best_norm_reward': float(best_norm_reward),
            'val_curve': [tuple(entry) for entry in val_curve],
            'val_reference': list(val_reference) if val_reference else [],
            'best_val': float(best_val),
            'converged': bool(converged)})
        print("Experiment %s training completed." % exp_config['name']
              if not converged else "Experiment %s training stopped early (converged)."
              % exp_config['name'])
        print(f"Best validation objective: {best_val:.6f}")
        print(f"Best smoothed normalised reward: {best_norm_reward:.4f}")
        return True

    except Exception as e:
        print(f"\nSevere error during training experiment {exp_config['name']}: {e}")
        print("Skipping this experiment and continuing with the next.")
        import traceback
        traceback.print_exc()
        # Report the failure to the caller; a completion marker printed by a wrapper around this
        # function must not be trusted when training did not actually finish.
        return False
    finally:
        # Restore the widths of the shared Config object for whoever runs next.
        cfg.gat_hidden_dim, cfg.gat_out_dim, cfg.policy_hidden_dim = original_dims


if __name__ == "__main__":
    # This file is not meant to be run directly; it is imported by train.py and evaluate.py
    pass


