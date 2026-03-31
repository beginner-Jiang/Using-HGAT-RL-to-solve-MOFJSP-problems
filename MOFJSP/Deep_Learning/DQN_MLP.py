"""
DQN+MLP Solution Implementation
Uses the most traditional DQN version (no variants) combined with a multi-layer perceptron
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
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


# 1. Configuration parameters

class Config:
    # Training parameters
    lr = 5e-5
    lr_decay = 0.999
    min_lr = 1e-6
    gamma = 0.99
    batch_size = 1024
    buffer_capacity = 20000
    target_update_freq = 500

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.999

    hidden_dim = 1024
    num_hidden_layers = 3

    objective_weights = [0.4, 0.3, 0.3]

    max_steps = 1000
    epochs = 30000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_jobs = None
    max_machines = None
    global_max_proc_time = None
    global_max_due_date = None

    model_dir = "../model/dqn"
    model_name = "dqn_model.pth"
    model_path = os.path.join(model_dir, model_name)

    instance_dir = "../mo_fjsp_instances"

    use_state_norm = True
    reward_scaling = 2.0
    reward_clip = 1.0

    sampling_temperature = 1.0
    precollect_episodes = 1
    grad_norm = 1.0

    gradient_steps = 256
    log_interval = 200
    use_amp = True

    num_envs = 32
    steps_per_update = 64

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
        """Update allowed sizes based on current episode ratio"""
        if not self.cfg.curriculum_enabled:
            self.allowed_sizes = ['small', 'medium', 'large']
            return
        for start, end, sizes in self.cfg.curriculum_stages:
            if start <= episode_progress < end:
                self.allowed_sizes = sizes
                break

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

    return np.concatenate((feat_m_load, feat_m_time, feat_j_progress, feat_j_urgency, feat_j_proc))

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
            # Handle invalid action
            print(f"Warning: invalid action {action_idx} (j={j_id}, m={m_id})")
            return self._get_state(), -1.0, False, {}

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
        rem_ops = (inst.job_op_ranges[j_id, 1] - inst.job_op_ranges[j_id, 0]) - curr_op_idx
        u_ij = max(0, due_date - end_time) / (inst.max_due_date * rem_ops + 1e-5)

        # Base reward
        reward = - (w[0] * (proc_time / inst.max_proc_time) +
                   w[1] * (self.machine_load[m_id] / L_max_est) +
                   w[2] * u_ij)
        # Scale by total number of operations to make cumulative reward range similar across different instance scales
        scale_factor = max(1.0, inst.total_ops / 50.0)   # Assume baseline scale of 50 operations
        reward /= scale_factor

        reward /= self.cfg.reward_scaling
        reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)
        done = (self.finished_ops_count == self.total_ops)
        info = {}
        if done:
            c_max = max(self.machine_avail_time[:inst.num_machines])
            avg_load = np.mean(self.machine_load[:inst.num_machines])
            lb = np.sqrt(np.mean((self.machine_load[:inst.num_machines] - avg_load) ** 2))
            t_tardy = sum(max(0, self.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))
            global_penalty = (w[0] * c_max / 100.0 + w[1] * lb / 50.0 + w[2] * t_tardy / 50.0)
            global_penalty /= scale_factor   # same scaling
            global_penalty /= self.cfg.reward_scaling
            reward -= global_penalty
            reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)
            info['final_metrics'] = {'Cmax': c_max, 'LB': lb, 'Tardy': t_tardy}
        next_state = self._get_state()
        if done:
            next_state = np.zeros_like(next_state)
        return next_state, float(reward), done, info


# 7. DQN Network

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
        self.mask_buffer = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.next_mask_buffer = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.state_buffer[self.pos] = state
        self.action_buffer[self.pos] = action
        self.reward_buffer[self.pos] = reward
        self.next_state_buffer[self.pos] = next_state
        self.done_buffer[self.pos] = done
        self.mask_buffer[self.pos] = mask
        self.next_mask_buffer[self.pos] = next_mask
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
        self.action_dim = cfg.max_jobs * cfg.max_machines
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
        os.makedirs(cfg.model_dir, exist_ok=True)
        self._load_model()
        if cfg.use_amp and cfg.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _save_model(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, self.model_path)

    def _load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.cfg.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except ValueError:
                pass
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps_done = checkpoint.get('steps_done', 0)
            print(f"Loaded model: {self.model_path}, epsilon={self.epsilon:.4f}")
        else:
            print("Starting training from scratch.")

    def select_action_batch(self, states, masks, epsilon):
        """states, masks are already on GPU"""
        batch_size = states.shape[0]
        explore_mask = torch.rand(batch_size, device=self.cfg.device) < epsilon
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.cfg.device)

        with torch.no_grad():
            q_values = self.policy_net(states)
            q_values_masked = q_values.clone()
            q_values_masked[masks == 0] = -1e4
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
                        next_q[nm == 0] = -1e4
                        next_max = next_q.max(1)[0].unsqueeze(1)
                        target = r + (1 - d) * self.cfg.gamma * next_max
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
                    next_q[nm == 0] = -1e4
                    next_max = next_q.max(1)[0].unsqueeze(1)
                    target = r + (1 - d) * self.cfg.gamma * next_max
                loss = self.loss_fn(current_q, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_norm)
                self.optimizer.step()
            self.recent_q_values.append(q.mean().item())
            self.recent_losses.append(loss.item())
            updated = True

        if updated and self.steps_done % self.cfg.target_update_freq == 0:
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
        self.states.append(env.reset())
        self.masks.append(env.get_action_mask())
        self.dones[idx] = False
        self.episode_rewards[idx] = 0.0
        self.step_counts[idx] = 0
        self.total_ops_list.append(env.inst.total_ops)

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
        for i, act in enumerate(actions):
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

        new_masks = [self.envs[i].get_action_mask() for i in range(self.num_envs)]

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
    else:
        state_norm = None

    env_manager = EnvManager(cfg, instance_list, cfg.num_envs, agent, state_norm)

    episode_count = 0
    normalized_rewards = []
    reward_ema = None
    final_info = None
    step_counter = 0
    last_log_interval = 0

    try:
        while episode_count < cfg.epochs:
            # Update curriculum stage
            progress = episode_count / cfg.epochs
            env_manager.curriculum_sampler.update_stage(progress)

            experiences = env_manager.step_all()
            step_counter += 1
            agent.steps_done += 1
            agent.update_epsilon()

            for exp in experiences:
                state, action, reward, next_state, done, mask, next_mask, info = exp
                state_gpu = torch.from_numpy(state).to(cfg.device)
                next_state_gpu = torch.from_numpy(next_state).to(cfg.device)
                mask_gpu = torch.from_numpy(mask).to(cfg.device)
                next_mask_gpu = torch.from_numpy(next_mask).to(cfg.device)
                action_t = torch.tensor(action, dtype=torch.long, device=cfg.device)
                reward_t = torch.tensor(reward, dtype=torch.float32, device=cfg.device)
                done_t = torch.tensor(done, dtype=torch.float32, device=cfg.device)

                agent.memory.push(state_gpu, action_t, reward_t, next_state_gpu,
                                  done_t, mask_gpu, next_mask_gpu)

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

            # Logging
            current_interval = episode_count // cfg.log_interval
            if current_interval > last_log_interval:
                last_log_interval = current_interval
                display_episode = current_interval * cfg.log_interval

                avg_norm = np.mean(normalized_rewards[-100:]) if normalized_rewards else 0
                stats = agent.get_stats()
                lr = agent.optimizer.param_groups[0]['lr']
                metrics = ""
                if final_info and 'final_metrics' in final_info:
                    m = final_info['final_metrics']
                    metrics = f" | Cmax={m['Cmax']:.1f} LB={m['LB']:.2f} Tardy={m['Tardy']:.1f}"
                # Print current allowed sizes
                allowed_sizes = env_manager.curriculum_sampler.allowed_sizes
                print(
                    f"Episode {display_episode:6d}/{cfg.epochs} | Sizes: {allowed_sizes} | NormReward: {reward_ema:8.4f} | "
                    f"Avg100: {avg_norm:8.4f} | Epsilon: {agent.epsilon:.4f} | AvgQ: {stats['avg_q']:6.2f} | "
                    f"Loss: {stats['avg_loss']:.4f} | Buffer: {stats['buffer_usage']*100:3.0f}% | LR: {lr:.2e}{metrics}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    agent._save_model()
    print(f"\nTraining completed, model saved to {agent.model_path}")

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
        plt.savefig("dqn_curve.png", dpi=150)
        plt.show()
    else:
        print("No reward data to plot.")


if __name__ == "__main__":
    train()