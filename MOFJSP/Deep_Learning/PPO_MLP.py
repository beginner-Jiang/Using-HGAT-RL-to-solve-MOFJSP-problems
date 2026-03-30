"""
PPO+MLP求解实现程序
使用最传统的PPO版本非变种，结合多层感知机求解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import glob
import re
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
import traceback

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


class Config:
    lr = 5e-4
    lr_decay = 0.9995
    min_lr = 1e-6
    value_lr_scale = 1.0
    warmup_steps = 1000

    # 熵系数（线性衰减）
    entropy_start = 0.02
    entropy_end = 0.0005

    # PPO 参数
    gamma = 0.99
    gae_lambda = 0.98
    clip_eps = 0.2
    kl_target = 0.01
    beta_start = 1.0
    beta_lr = 0.02
    value_coef = 0.5
    entropy_coef = entropy_start

    # 网络与训练
    batch_size = 2048
    n_epochs = 5
    hidden_dim = 512
    num_hidden_layers = 2

    episodes_per_update = 100
    objective_weights = [0.4, 0.3, 0.3]

    # 模型保存
    model_dir = "../model/ppo"
    model_name = "ppo_model.pth"
    model_path = os.path.join(model_dir, model_name)

    # 数据集
    epochs = 30000
    max_jobs = None
    max_machines = None
    global_max_proc_time = None
    global_max_due_date = None
    instance_dir = "../mo_fjsp_instances"
    file_pattern = "mo_fjsp_*_train.txt"

    # 状态归一化
    use_state_norm = True
    reward_scaling = 20.0
    reward_clip = 20.0

    # 统计与梯度
    stats_window = 100
    value_clip = 0.2
    weight_decay = 1e-5
    grad_norm = 1.0

    precollect_episodes = 50

    # 课程学习
    enable_curriculum = True
    curriculum_stages = [
        (0.0, 0.3, ['small']),
        (0.3, 0.6, ['small', 'medium']),
        (0.6, 1.0, ['small', 'medium', 'large'])
    ]

    # 混合精度
    use_amp = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



# 文件读取与实例包装

@dataclass
class Operation:
    job_id: int
    op_id: int
    machine_times: Dict[int, float]
    avg_proc_time: float


def parse_size_from_filename(filename: str) -> str:
    match = re.search(r'(small|medium|large)', filename)
    return match.group(1) if match else "unknown"


def read_fjsp_instance(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到！")
        exit(1)

    if len(lines) < 2:
        raise ValueError("文件格式错误：行数不足")

    num_jobs, num_machines = map(int, lines[0].split())
    machine_ids = list(range(num_machines))

    job_lines = lines[1:1 + num_jobs]
    if len(job_lines) != num_jobs:
        raise ValueError(f"期望 {num_jobs} 个作业行，实际得到 {len(job_lines)} 行")

    due_date_line = lines[1 + num_jobs]
    due_dates = list(map(float, due_date_line.split()))
    if len(due_dates) != num_jobs:
        raise ValueError(f"期望 {num_jobs} 个交货期，实际得到 {len(due_dates)} 个")

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
                if 0 <= machine_id < num_machines:
                    machine_times[machine_id] = proc_time
                idx += 2
            avg_proc = sum(machine_times.values()) / len(machine_times) if machine_times else 0.0
            op = Operation(job_idx, op_idx, machine_times, avg_proc)
            job_ops.append(op)
        jobs.append(job_ops)

    return jobs, machine_ids, due_dates


def load_all_instances(instance_dir: str, pattern: str):
    file_paths = glob.glob(os.path.join(instance_dir, pattern))
    if not file_paths:
        raise FileNotFoundError(f"在目录 {instance_dir} 中未找到匹配 {pattern} 的文件")

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


class JobShopInstance:
    __slots__ = ('jobs', 'machine_ids', 'due_dates', 'file_name',
                 'num_jobs', 'num_machines', 'max_jobs', 'max_machines',
                 'total_ops', 'max_proc_time', 'max_due_date',
                 'total_ops_per_job', 'op_masks', 'op_proc_times', 'avg_proc_times',
                 'size')

    def __init__(self, jobs, machine_ids, due_dates, max_jobs, max_machines, file_name, size):
        self.jobs = jobs
        self.machine_ids = machine_ids
        self.due_dates = np.array(due_dates, dtype=np.float32)
        self.file_name = file_name
        self.size = size

        self.num_jobs = len(jobs)
        self.num_machines = len(machine_ids)
        self.max_jobs = max_jobs
        self.max_machines = max_machines

        self.total_ops = sum(len(job_ops) for job_ops in jobs)
        self.max_proc_time = 0
        max_ops_per_job = max(len(job_ops) for job_ops in jobs)

        self.op_masks = np.zeros((self.num_jobs, max_ops_per_job, max_machines), dtype=bool)
        self.op_proc_times = np.zeros((self.num_jobs, max_ops_per_job, max_machines), dtype=np.float32)
        self.avg_proc_times = np.zeros((self.num_jobs, max_ops_per_job), dtype=np.float32)

        for j, job_ops in enumerate(jobs):
            for op_idx, op in enumerate(job_ops):
                for m_id, pt in op.machine_times.items():
                    self.op_masks[j, op_idx, m_id] = True
                    self.op_proc_times[j, op_idx, m_id] = pt
                self.avg_proc_times[j, op_idx] = op.avg_proc_time
                self.max_proc_time = max(self.max_proc_time, max(op.machine_times.values()))

        self.max_due_date = max(due_dates) if due_dates else 1.0
        self.total_ops_per_job = np.array([len(job_ops) for job_ops in jobs], dtype=np.int32)


# 课程学习采样器

class CurriculumSampler:
    def __init__(self, instances: List[JobShopInstance], cfg: Config):
        self.instances = instances
        self.cfg = cfg
        self.instances_by_size = {'small': [], 'medium': [], 'large': []}
        for inst in instances:
            if inst.size in self.instances_by_size:
                self.instances_by_size[inst.size].append(inst)
            else:
                self.instances_by_size['large'].append(inst)
        self.allowed_sizes = ['small']

    def update_stage(self, episode_progress: float):
        if not self.cfg.enable_curriculum:
            self.allowed_sizes = ['small', 'medium', 'large']
            return
        for start, end, sizes in self.cfg.curriculum_stages:
            if start <= episode_progress < end:
                self.allowed_sizes = sizes
                break

    def sample(self) -> JobShopInstance:
        allowed_instances = []
        for sz in self.allowed_sizes:
            allowed_instances.extend(self.instances_by_size.get(sz, []))
        if not allowed_instances:
            allowed_instances = self.instances
        return random.choice(allowed_instances)


# 状态归一化（支持在线更新）

class RunningMeanStd:
    __slots__ = ('mean', 'var', 'count')
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# 调度环境（稀疏奖励，多目标归一化）

class MOFJSP_Env:
    __slots__ = ('cfg', 'inst', 'norm', 'action_dim',
                 'machine_avail_time', 'machine_load',
                 'job_next_op_idx', 'job_avail_time',
                 'finished_ops_count', 'total_ops_count',
                 'num_jobs_actual', 'num_machines_actual',
                 'max_proc_time', 'max_due_date')

    def __init__(self, cfg, instance: JobShopInstance, norm: RunningMeanStd = None):
        self.cfg = cfg
        self.inst = instance
        self.norm = norm
        self.action_dim = cfg.max_jobs * cfg.max_machines

        self.num_jobs_actual = instance.num_jobs
        self.num_machines_actual = instance.num_machines
        self.max_proc_time = instance.max_proc_time
        self.max_due_date = instance.max_due_date

        self.reset()

    def reset(self):
        self.machine_avail_time = np.zeros(self.cfg.max_machines, dtype=np.float32)
        self.machine_load = np.zeros(self.cfg.max_machines, dtype=np.float32)
        self.job_next_op_idx = np.zeros(self.cfg.max_jobs, dtype=np.int32)
        self.job_avail_time = np.zeros(self.cfg.max_jobs, dtype=np.float32)

        self.finished_ops_count = 0
        self.total_ops_count = self.inst.total_ops

        return self._get_state()

    def _get_state(self):
        inst = self.inst
        n_m = self.num_machines_actual
        n_j = self.num_jobs_actual

        if n_m > 0:
            max_time_est = max(self.machine_avail_time[:n_m].max(), 100.0)
            max_load_est = max(self.machine_load[:n_m].max(), 100.0)
            feat_m_load = self.machine_load.copy()
            feat_m_load[:n_m] /= max_load_est
            feat_m_time = self.machine_avail_time.copy()
            feat_m_time[:n_m] /= max_time_est
        else:
            feat_m_load = self.machine_load
            feat_m_time = self.machine_avail_time

        current_global_time = self.machine_avail_time[:n_m].max() if n_m > 0 else 0.0

        feat_j_progress = np.zeros(self.cfg.max_jobs, dtype=np.float32)
        feat_j_urgency = np.zeros(self.cfg.max_jobs, dtype=np.float32)
        feat_j_proc = np.zeros(self.cfg.max_jobs, dtype=np.float32)

        if n_j > 0:
            total_ops = inst.total_ops_per_job[:n_j]
            curr_idx = self.job_next_op_idx[:n_j]
            feat_j_progress[:n_j] = curr_idx / np.maximum(total_ops, 1.0)
            feat_j_urgency[:n_j] = current_global_time / np.maximum(inst.due_dates[:n_j], 1.0)
            mask = curr_idx < total_ops
            if np.any(mask):
                j_indices = np.where(mask)[0]
                op_indices = curr_idx[mask]
                feat_j_proc[j_indices] = inst.avg_proc_times[j_indices, op_indices] / inst.max_proc_time

        state_vec = np.concatenate([
            feat_m_load,
            feat_m_time,
            feat_j_progress,
            feat_j_urgency,
            feat_j_proc
        ]).astype(np.float32)

        if self.norm is not None:
            self.norm.update(state_vec.reshape(1, -1))
            state_vec = self.norm.normalize(state_vec)
        return state_vec

    def get_action_mask(self):
        mask = np.zeros(self.action_dim, dtype=bool)
        n_j = self.num_jobs_actual
        for j_id in range(n_j):
            curr_op_idx = self.job_next_op_idx[j_id]
            if curr_op_idx < self.inst.total_ops_per_job[j_id]:
                op_mask = self.inst.op_masks[j_id, curr_op_idx]
                start = j_id * self.cfg.max_machines
                end = start + self.cfg.max_machines
                mask[start:end] = op_mask
        return mask

    def step(self, action_idx):
        j_id = action_idx // self.cfg.max_machines
        m_id = action_idx % self.cfg.max_machines

        inst = self.inst
        curr_op_idx = self.job_next_op_idx[j_id]
        proc_time = inst.op_proc_times[j_id, curr_op_idx, m_id]

        start_time = max(self.job_avail_time[j_id], self.machine_avail_time[m_id])
        end_time = start_time + proc_time

        self.job_avail_time[j_id] = end_time
        self.machine_avail_time[m_id] = end_time
        self.machine_load[m_id] += proc_time
        self.job_next_op_idx[j_id] += 1
        self.finished_ops_count += 1

        reward = 0.0
        done = (self.finished_ops_count == self.total_ops_count)

        if done:
            w = self.cfg.objective_weights
            c_max = self.machine_avail_time[:inst.num_machines].max()
            avg_load = np.mean(self.machine_load[:inst.num_machines])
            lb = np.sqrt(np.mean((self.machine_load[:inst.num_machines] - avg_load) ** 2))
            t_tardy = sum(max(0, self.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))

            # 归一化每个目标：除以实例的最大可能值（或固定值），使量纲一致
            # 这里使用实例的 max_proc_time 和 max_due_date 作为基准
            norm_cmax = c_max / max(self.max_proc_time, 1.0)
            norm_lb = lb / max(self.max_proc_time, 1.0)      # 负载平衡也可用 max_proc_time 归一化
            norm_tardy = t_tardy / (self.max_due_date * inst.num_jobs + 1.0)

            final_penalty = w[0] * norm_cmax + w[1] * norm_lb + w[2] * norm_tardy
            reward = -final_penalty * self.cfg.reward_scaling
            reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)
            info = {'final_metrics': {'Cmax': c_max, 'LB': lb, 'Tardy': t_tardy,
                                      'norm_cmax': norm_cmax, 'norm_lb': norm_lb, 'norm_tardy': norm_tardy}}
        else:
            info = {}

        next_state = self._get_state() if not done else None
        return next_state, float(reward), done, info



# 神经网络

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            orthogonal_init(linear, gain=nn.init.calculate_gain('relu'))
            layers.extend([linear, nn.ReLU()])
            in_dim = hidden_dim
        output = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(output, gain=0.01)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, state, mask=None):
        logits = self.net(state)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        if mask is not None:
            fill_value = -1e4
            logits = logits.masked_fill(~mask, fill_value)
            row_has_valid = mask.any(dim=1)
            if not row_has_valid.all():
                invalid_rows = ~row_has_valid
                logits[invalid_rows] = 0.0
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            orthogonal_init(linear, gain=nn.init.calculate_gain('relu'))
            layers.extend([linear, nn.ReLU()])
            in_dim = hidden_dim
        output = nn.Linear(hidden_dim, 1)
        orthogonal_init(output, gain=1.0)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state).squeeze(-1)



# PPO 训练器

class PPOKLTrainer:
    def __init__(self, policy: PolicyNetwork, value: ValueNetwork, cfg: Config):
        self.policy = policy
        self.value = value
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_eps = cfg.clip_eps
        self.kl_target = cfg.kl_target
        self.beta = cfg.beta_start
        self.beta_lr = cfg.beta_lr
        self.value_coef = cfg.value_coef
        self.entropy_coef = cfg.entropy_coef
        self.n_epochs = cfg.n_epochs
        self.batch_size = cfg.batch_size
        self.value_clip = cfg.value_clip
        self.grad_norm = cfg.grad_norm
        self.min_lr = cfg.min_lr
        self.warmup_steps = cfg.warmup_steps
        self.target_lr = cfg.lr

        policy_params = list(policy.parameters())
        value_params = list(value.parameters())
        self.optimizer = optim.AdamW([
            {'params': policy_params, 'lr': cfg.min_lr, 'weight_decay': cfg.weight_decay},
            {'params': value_params, 'lr': cfg.min_lr * cfg.value_lr_scale, 'weight_decay': cfg.weight_decay}
        ])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)
        self.scaler = torch.cuda.amp.GradScaler() if (cfg.use_amp and DEVICE.type == 'cuda') else None

        self.current_episode = 0
        self.steps_done = 0
        os.makedirs(cfg.model_dir, exist_ok=True)
        self.model_path = cfg.model_path

        self.kl_ema = None
        self.kl_alpha = 0.9
        self.stats_window = cfg.stats_window
        self.recent_policy_loss = deque(maxlen=self.stats_window)
        self.recent_value_loss = deque(maxlen=self.stats_window)
        self.recent_entropy = deque(maxlen=self.stats_window)
        self.recent_kl = deque(maxlen=self.stats_window)
        self.recent_beta = deque(maxlen=self.stats_window)
        self.recent_advantage = deque(maxlen=self.stats_window)
        self.recent_value = deque(maxlen=self.stats_window)

        self._load_model()

    def _save_model(self):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'beta': self.beta,
            'steps_done': self.steps_done,
            'cfg': self.cfg
        }, self.model_path)

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=DEVICE)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.value.load_state_dict(checkpoint['value_state_dict'])
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except ValueError as e:
                    print(f"优化器状态加载失败: {e}")
                self.beta = checkpoint.get('beta', self.beta)
                self.steps_done = checkpoint.get('steps_done', 0)
                print(f"已加载模型: {self.model_path}, beta={self.beta:.4f}")
                return True
            except RuntimeError as e:
                print(f"模型结构不匹配，从头训练: {e}")
                return False
        else:
            print("未找到模型，从头训练。")
            return False

    def compute_gae_gpu(self, rewards, values, dones):
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def _update_lr_with_warmup(self):
        if self.steps_done < self.warmup_steps:
            scale = self.steps_done / max(1, self.warmup_steps)
            current_lr = self.min_lr + scale * (self.target_lr - self.min_lr)
            current_val_lr = current_lr * self.cfg.value_lr_scale
            for param_group in self.optimizer.param_groups:
                if param_group['weight_decay'] == self.cfg.weight_decay:
                    param_group['lr'] = current_lr
                else:
                    param_group['lr'] = current_val_lr
        else:
            for param_group in self.optimizer.param_groups:
                if param_group['weight_decay'] == self.cfg.weight_decay:
                    param_group['lr'] = self.target_lr
                else:
                    param_group['lr'] = self.target_lr * self.cfg.value_lr_scale

    def train_step(self, trajectories_list):
        all_states, all_actions, all_masks, all_old_logprobs, all_old_values = [], [], [], [], []
        all_advantages, all_returns = [], []

        for traj in trajectories_list:
            states = np.stack([t['state'] for t in traj])
            actions = np.array([t['action_id'] for t in traj])
            masks = np.stack([t['mask'] for t in traj])
            old_logprobs = np.array([t['logprob'] for t in traj])
            old_values = np.array([t['value'] for t in traj])
            rewards = [t['reward'] for t in traj]
            dones = [t['done'] for t in traj]

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
            old_values_t = torch.tensor(old_values, dtype=torch.float32, device=DEVICE)
            values_extended = torch.cat([old_values_t, torch.zeros(1, device=DEVICE)])
            adv_t, ret_t = self.compute_gae_gpu(rewards_t, values_extended, dones_t)

            all_states.append(torch.tensor(states, device=DEVICE))
            all_actions.append(torch.tensor(actions, device=DEVICE))
            all_masks.append(torch.tensor(masks, device=DEVICE))
            all_old_logprobs.append(torch.tensor(old_logprobs, device=DEVICE))
            all_old_values.append(old_values_t)
            all_advantages.append(adv_t)
            all_returns.append(ret_t)

        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        masks = torch.cat(all_masks, dim=0)
        old_logprobs = torch.cat(all_old_logprobs, dim=0)
        old_values = torch.cat(all_old_values, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)

        # 打印原始优势统计
        orig_adv_mean = advantages.mean().item()
        orig_adv_std = advantages.std().item()
        print(f"[Debug] Original advantages - mean: {orig_adv_mean:.6f}, std: {orig_adv_std:.6f}")

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("Warning: advantages contains NaN/Inf, skipping training step")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'kl': 0.0,
                'beta': self.beta
            }

        adv_std = advantages.std()
        if adv_std > 1e-4:
            advantages = (advantages - advantages.mean()) / (adv_std + EPS)
        else:
            advantages = advantages - advantages.mean()
            print("Warning: advantage std very small, skip scaling")

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        total_kl = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        any_batch_processed = False

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]
                batch_idx_t = torch.tensor(batch_idx, device=DEVICE)

                batch_states = states[batch_idx_t]
                batch_actions = actions[batch_idx_t]
                batch_masks = masks[batch_idx_t]
                batch_old_logprobs = old_logprobs[batch_idx_t]
                batch_advantages = advantages[batch_idx_t]
                batch_returns = returns[batch_idx_t]
                batch_old_values = old_values[batch_idx_t]

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        probs, logits = self.policy(batch_states, batch_masks)
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            print("Warning: logits contains NaN/Inf, skipping batch")
                            continue
                        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                        logprobs = dist.log_prob(batch_actions)
                        entropy = dist.entropy().mean()

                        values = self.value(batch_states)

                        ratio = torch.exp(logprobs - batch_old_logprobs)
                        ratio = torch.clamp(ratio, 1e-4, 1e4)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        kl_batch = (batch_old_logprobs - logprobs).mean().detach()

                        if self.value_clip is not None:
                            value_clipped = batch_old_values + torch.clamp(
                                values - batch_old_values, -self.value_clip, self.value_clip
                            )
                            loss_unclipped = F.smooth_l1_loss(values, batch_returns)
                            loss_clipped = F.smooth_l1_loss(value_clipped, batch_returns)
                            value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped)
                        else:
                            value_loss = F.smooth_l1_loss(values, batch_returns)

                        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    if torch.isnan(total_loss).any():
                        print("Warning: NaN detected in loss (AMP), skipping batch")
                        continue

                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    probs, logits = self.policy(batch_states, batch_masks)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("Warning: logits contains NaN/Inf, skipping batch")
                        continue
                    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                    logprobs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    values = self.value(batch_states)

                    ratio = torch.exp(logprobs - batch_old_logprobs)
                    ratio = torch.clamp(ratio, 1e-4, 1e4)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    kl_batch = (batch_old_logprobs - logprobs).mean().detach()

                    if self.value_clip is not None:
                        value_clipped = batch_old_values + torch.clamp(
                            values - batch_old_values, -self.value_clip, self.value_clip
                        )
                        loss_unclipped = F.smooth_l1_loss(values, batch_returns)
                        loss_clipped = F.smooth_l1_loss(value_clipped, batch_returns)
                        value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped)
                    else:
                        value_loss = F.smooth_l1_loss(values, batch_returns)

                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    if torch.isnan(total_loss).any():
                        print("Warning: NaN detected in loss, skipping batch")
                        continue

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_norm)
                    self.optimizer.step()

                total_kl += kl_batch.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
                any_batch_processed = True

        if num_batches == 0:
            print("Warning: No valid batches in this training step")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'kl': 0.0,
                'beta': self.beta
            }

        avg_kl = total_kl / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches

        if self.kl_ema is None:
            self.kl_ema = avg_kl
        else:
            self.kl_ema = self.kl_alpha * self.kl_ema + (1 - self.kl_alpha) * avg_kl

        kl_error = self.kl_ema - self.kl_target
        self.beta *= (1 + self.beta_lr * kl_error)
        self.beta = np.clip(self.beta, 0.01, 10.0)

        self.steps_done += dataset_size

        self._update_lr_with_warmup()
        if any_batch_processed and self.steps_done >= self.warmup_steps:
            self.scheduler.step()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], self.min_lr)

        self.recent_policy_loss.append(avg_policy_loss)
        self.recent_value_loss.append(avg_value_loss)
        self.recent_entropy.append(avg_entropy)
        self.recent_kl.append(avg_kl)
        self.recent_beta.append(self.beta)
        self.recent_advantage.append(advantages.mean().item())
        self.recent_value.append(returns.mean().item())

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl': avg_kl,
            'beta': self.beta
        }

    def get_stats(self):
        return {
            'policy_loss': np.mean(self.recent_policy_loss) if self.recent_policy_loss else 0.0,
            'value_loss': np.mean(self.recent_value_loss) if self.recent_value_loss else 0.0,
            'entropy': np.mean(self.recent_entropy) if self.recent_entropy else 0.0,
            'kl': np.mean(self.recent_kl) if self.recent_kl else 0.0,
            'beta': self.recent_beta[-1] if self.recent_beta else self.beta,
            'advantage': np.mean(self.recent_advantage) if self.recent_advantage else 0.0,
            'value': np.mean(self.recent_value) if self.recent_value else 0.0
        }


# 主训练函数

def train():
    cfg = Config()
    set_seed(42)

    print(f"正在从目录 {cfg.instance_dir} 加载实例...")
    try:
        instance_tuples, max_jobs, max_machines, global_max_proc, global_max_due = \
            load_all_instances(cfg.instance_dir, cfg.file_pattern)
    except FileNotFoundError as e:
        print(e)
        print("请检查实例路径和文件模式。")
        return

    cfg.max_jobs = max_jobs
    cfg.max_machines = max_machines
    cfg.global_max_proc_time = global_max_proc
    cfg.global_max_due_date = global_max_due

    print(f"共加载 {len(instance_tuples)} 个实例，最大作业数 {max_jobs}，最大机器数 {max_machines}")

    instance_list = []
    for jobs, mids, dues, fname, size in instance_tuples:
        inst = JobShopInstance(jobs, mids, dues, max_jobs, max_machines, fname, size)
        instance_list.append(inst)

    if cfg.enable_curriculum:
        sampler = CurriculumSampler(instance_list, cfg)
        print("课程学习已启用，将按阶段采样实例。")
    else:
        sampler = None
        print("课程学习未启用，将随机均匀采样所有实例。")

    state_dim = 2 * cfg.max_machines + 3 * cfg.max_jobs
    action_dim = cfg.max_jobs * cfg.max_machines

    policy = PolicyNetwork(state_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(DEVICE)
    value = ValueNetwork(state_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(DEVICE)

    try:
        policy = torch.jit.script(policy)
        value = torch.jit.script(value)
        print("网络已使用 torch.jit.script 加速。")
    except Exception as e:
        print(f"torch.jit.script 编译失败，使用普通模式：{e}")

    trainer = PPOKLTrainer(policy, value, cfg)

    state_norm = RunningMeanStd(shape=(state_dim,)) if cfg.use_state_norm else None
    if cfg.use_state_norm and state_norm is not None:
        print(f"\n预收集 {cfg.precollect_episodes} 个episode用于归一化初始化...")
        for _ in range(cfg.precollect_episodes):
            inst = random.choice(instance_list)
            env = MOFJSP_Env(cfg, inst, norm=None)
            state = env.reset()
            done = False
            states_buffer = []
            while not done:
                mask = env.get_action_mask()
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    break
                action_id = np.random.choice(valid_actions)
                next_state, _, done, _ = env.step(action_id)
                states_buffer.append(state)
                state = next_state if next_state is not None else None
            if states_buffer:
                state_norm.update(np.stack(states_buffer))
        print("归一化初始化完成，后续将在每个step在线更新统计量。")

    print(f"设备: {DEVICE}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"总训练 episode 数: {cfg.epochs}")
    print(f"每 {cfg.episodes_per_update} 个 episode 训练一次")
    print(f"批大小: {cfg.batch_size}, 隐藏层维度: {cfg.hidden_dim}, 层数: {cfg.num_hidden_layers}")

    normalized_rewards = []
    kl_list = []
    total_loss_list = []
    update_episodes = []          # 记录每次更新时的episode
    best_norm_reward = -float('inf')
    reward_ema = None
    cmax_list, lb_list, tardy_list = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    all_cmax, all_lb, all_tardy = [], [], []
    entropy_start, entropy_end = cfg.entropy_start, cfg.entropy_end
    trajectory_buffer = []
    episodes_collected = 0

    start_time = time.time()
    step_count = 0

    for ep in range(1, cfg.epochs + 1):
        progress = ep / cfg.epochs
        if sampler is not None:
            sampler.update_stage(progress)

        if sampler is not None:
            inst = sampler.sample()
        else:
            inst = random.choice(instance_list)

        env = MOFJSP_Env(cfg, inst, norm=state_norm if cfg.use_state_norm else None)
        state = env.reset()
        done = False
        trajectory = []
        ep_reward = 0.0
        final_info = None

        while not done:
            mask = env.get_action_mask()
            if not mask.any():
                print(f"警告：mask 全为 False，终止当前 episode")
                done = True
                break

            state_t = torch.from_numpy(state).to(DEVICE, non_blocking=True).unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(DEVICE, dtype=torch.bool, non_blocking=True).unsqueeze(0)

            with torch.no_grad():
                _, logits = policy(state_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                action = dist.sample().item()
                logprob = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                value_t = value(state_t).item()

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            step_count += 1

            step_data = {
                'state': state,
                'action_id': action,
                'reward': reward,
                'done': done,
                'logprob': logprob,
                'value': value_t,
                'mask': mask
            }
            trajectory.append(step_data)

            state = next_state if next_state is not None else None

            if done:
                final_info = info

        episodes_collected = ep
        # 归一化奖励：总奖励除以总工序数（便于比较不同规模）
        norm_reward = ep_reward / inst.total_ops
        normalized_rewards.append(norm_reward)
        if final_info:
            cmax_list.append(final_info['final_metrics']['Cmax'])
            lb_list.append(final_info['final_metrics']['LB'])
            tardy_list.append(final_info['final_metrics']['Tardy'])
            all_cmax.append(final_info['final_metrics']['Cmax'])
            all_lb.append(final_info['final_metrics']['LB'])
            all_tardy.append(final_info['final_metrics']['Tardy'])
        else:
            if all_cmax:
                all_cmax.append(all_cmax[-1])
                all_lb.append(all_lb[-1])
                all_tardy.append(all_tardy[-1])
            else:
                all_cmax.append(0)
                all_lb.append(0)
                all_tardy.append(0)

        trajectory_buffer.append(trajectory)

        reward_ema = norm_reward if reward_ema is None else 0.9 * reward_ema + 0.1 * norm_reward
        if norm_reward > best_norm_reward:
            best_norm_reward = norm_reward
            trainer._save_model()

        progress = min(ep / cfg.epochs, 1.0)
        trainer.entropy_coef = entropy_start * (1 - progress) + entropy_end * progress

        if len(trajectory_buffer) >= cfg.episodes_per_update:
            try:
                train_stats = trainer.train_step(trajectory_buffer)
                # 记录更新时的 episode 和相关统计
                update_episodes.append(ep)
                if train_stats:
                    kl_list.append(train_stats['kl'])
                    total_loss_list.append(train_stats['policy_loss'] + cfg.value_coef*train_stats['value_loss'] - trainer.entropy_coef*train_stats['entropy'])
            except Exception as e:
                print(f"训练出错: {e}")
                traceback.print_exc()
                train_stats = {}
            trajectory_buffer.clear()

        if ep % 100 == 0:
            avg_norm = np.mean(normalized_rewards[-100:]) if normalized_rewards else 0
            stats = trainer.get_stats()
            lr = trainer.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            steps_per_sec = step_count / elapsed
            metrics = f" | Cmax={final_info['final_metrics']['Cmax']:.1f} LB={final_info['final_metrics']['LB']:.2f} Tardy={final_info['final_metrics']['Tardy']:.1f}" if final_info else ""
            if cmax_list:
                metrics += f" | avgCmax={np.mean(cmax_list):.1f} avgLB={np.mean(lb_list):.2f} avgTardy={np.mean(tardy_list):.1f}"
            allowed_sizes = sampler.allowed_sizes if sampler else 'all'
            print(f"Episode {ep:6d}/{cfg.epochs} | Sizes: {allowed_sizes} | NormReward: {norm_reward:8.4f} | EMA: {reward_ema:8.4f} | Avg100: {avg_norm:8.4f} | "
                  f"Beta: {trainer.beta:.4f} | KL: {stats['kl']:.4f} | Entropy: {trainer.entropy_coef:.4f} | "
                  f"PolLoss: {stats['policy_loss']:.4f} | ValLoss: {stats['value_loss']:.4f} | "
                  f"Adv: {stats['advantage']:8.6f} | Val: {stats['value']:8.3f} | LR: {lr:.2e} | Steps/s: {steps_per_sec:.1f}{metrics}")

    trainer._save_model()
    print(f"\n训练完成，模型已保存至 {trainer.model_path}")

    # 绘图
    plt.figure(figsize=(15, 5))

    def moving_average(data, window_size=20):
        if len(data) < window_size:
            return data
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # 子图1：归一化奖励
    plt.subplot(1, 3, 1)
    plt.plot(normalized_rewards, alpha=0.3, color='blue', label='Raw')
    if len(normalized_rewards) >= 20:
        ma_rewards = moving_average(normalized_rewards, 20)
        plt.plot(range(19, len(normalized_rewards)), ma_rewards, color='blue', linewidth=2, label='Moving Avg (20)')
    plt.title("Normalized Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Norm Reward")
    plt.grid(True)
    plt.legend()
    if normalized_rewards:
        y_min, y_max = min(normalized_rewards), max(normalized_rewards)
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        plt.ylim(y_min - margin, y_max + margin)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 子图2：KL散度（使用 update_episodes 作为横坐标）
    plt.subplot(1, 3, 2)
    if len(kl_list) > 0:
        plt.plot(update_episodes, kl_list, alpha=0.3, color='orange', label='Raw')
        if len(kl_list) >= 20:
            ma_kl = moving_average(kl_list, 20)
            # 移动平均后长度减少，对应横坐标需截取
            plt.plot(update_episodes[19:], ma_kl, color='orange', linewidth=2, label='Moving Avg (20)')
    plt.title("KL Divergence Curve (per update)")
    plt.xlabel("Episode")
    plt.ylabel("KL")
    plt.grid(True)
    plt.legend()
    if kl_list:
        y_min, y_max = min(kl_list), max(kl_list)
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        plt.ylim(y_min - margin, y_max + margin)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 子图3：总损失（使用 update_episodes 作为横坐标）
    plt.subplot(1, 3, 3)
    if len(total_loss_list) > 0:
        plt.plot(update_episodes, total_loss_list, alpha=0.3, color='red', label='Raw')
        if len(total_loss_list) >= 20:
            ma_loss = moving_average(total_loss_list, 20)
            plt.plot(update_episodes[19:], ma_loss, color='red', linewidth=2, label='Moving Avg (20)')
    plt.title("Total Loss Curve (per update)")
    plt.xlabel("Episode")
    plt.ylabel("L(θ)")
    plt.grid(True)
    plt.legend()
    if total_loss_list:
        y_min, y_max = min(total_loss_list), max(total_loss_list)
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        plt.ylim(y_min - margin, y_max + margin)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig("ppo_training_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    train()