"""
基于同质析取图与PPO+KL的多目标柔性作业车间调度求解（多实例训练版）
- 彻底修复索引越界问题：collate_states 和 GAT 层多重安全防护
- 稳定奖励设计，避免剧烈波动
- 梯度累积与保守更新策略
- 增强多进程健壮性：超时退出、worker存活检测、队列异常处理
训练结束后可能存在无法退出的问题，可手动结束，不影响整体训练结果
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
import matplotlib.pyplot as plt
import time
import re
import gc
import multiprocessing as mp
from queue import Empty, Full
import signal

# 尝试导入numba，若不可用则回退到纯Python
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("警告: Numba未安装，将使用纯Python版本（可能较慢）")

# 混合精度
from torch.cuda.amp import autocast, GradScaler

# ----------------------------
# 1. 全局配置与常量
# ----------------------------
# 强制使用CPU，避免CUDA device-side assert
DEVICE = torch.device("cpu")
EPS = 1e-10
W1, W2, W3 = 0.4, 0.3, 0.3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# 2. 配置类（稳定训练配置）
# ----------------------------
class Config:
    n_episodes = 10000
    model_save_dir = "../model/ppo_gat"

    # ========== 稳定超参数 ==========
    lr = 1e-3                       # 学习率
    lr_decay = 0.999
    min_lr = 1e-6
    gamma = 0.99
    gae_lambda = 0.95
    kl_target = 0.01                 # 保守 KL 目标
    beta_start = 0.1                  # 初始 beta
    value_coef = 0.25                 # 价值系数
    entropy_coef = 0.01               # 熵系数
    entropy_decay = 0.999
    max_grad_norm = 0.5               # 梯度裁剪

    gat_hidden_dim = 64                # 网络容量
    gat_out_dim = 64
    policy_hidden_dim = 128
    gat_num_layers = 2
    gat_num_heads = 4

    value_clip = 0.1
    reward_scaling = 1.0               # 不缩放奖励
    reward_clip = 1.0                  # 奖励裁剪范围
    use_state_norm = True
    use_disjunctive_edges = False
    instance_dir = "../mo_fjsp_instances"
    file_pattern = "mo_fjsp_*_train.txt"
    sampling_temperature = 1.0
    stats_window = 100
    max_steps_per_episode = 2000

    train_step_size = 256              # 训练步数阈值
    ppo_epochs = 1
    use_amp = False

    num_workers = 4
    worker_queue_size = 2

    curriculum_enabled = True
    curriculum_stages = [
        (0.0, 0.3, ['small']),
        (0.3, 0.6, ['small', 'medium']),
        (0.6, 1.0, ['small', 'medium', 'large'])
    ]

cfg = Config()

# ----------------------------
# 3. 文件读取模块（不变）
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
        print(f"错误：文件 {file_path} 未找到！")
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

# ----------------------------
# 4. MOFJSP 实例包装类
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
# 5. 状态归一化工具（CPU版）
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
# 6. Numba加速函数（不变）
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
    # 纯Python回退
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
# 7. 环境类（稳定奖励设计）
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

        # 预计算静态信息
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

        # 节点索引映射
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

        # 预计算静态边
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

        # 预计算静态特征
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

        # 稳定奖励设计
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

        return {
            'node_features': torch.from_numpy(self.node_features).float(),
            'edge_src': torch.from_numpy(src).long(),
            'edge_tgt': torch.from_numpy(tgt).long(),
            'edge_features': torch.from_numpy(feat).float().unsqueeze(-1),
            'action_mask': self.get_action_mask()
        }

# ----------------------------
# 8. GAT 网络（带多重索引安全防护）
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

        # ---- 索引安全预处理 ----
        if edge_src.numel() > 0:
            # 确保类型正确
            edge_src = edge_src.long()
            edge_tgt = edge_tgt.long()
            # 裁剪到合法范围
            max_idx = num_nodes - 1
            edge_src = torch.clamp(edge_src, 0, max_idx)
            edge_tgt = torch.clamp(edge_tgt, 0, max_idx)
            # 再次检查，若有仍不合法的（理论上不会），则过滤
            valid = (edge_src >= 0) & (edge_src < num_nodes) & (edge_tgt >= 0) & (edge_tgt < num_nodes)
            if not valid.all():
                # 记录并过滤
                print(f"GATLayer: filtering {valid.logical_not().sum()} invalid edges after clamp.")
                edge_src = edge_src[valid]
                edge_tgt = edge_tgt[valid]
                edge_feat = edge_feat[valid]

        Wh = self.W(h).view(-1, self.num_heads, self.head_dim)

        # 如果没有有效边，直接返回线性变换
        if edge_feat.shape[0] == 0:
            return F.relu(Wh.view(-1, self.out_dim))

        # ---- 安全索引操作 ----
        try:
            We = self.W_edge(edge_feat).view(-1, self.num_heads, self.head_dim)
            # 再次确认索引不越界（防止多线程竞争导致的意外）
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
# 9. 策略与价值网络
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            nn.Dropout(0.1),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            nn.Dropout(0.1),
            orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        )

    def forward(self, state_emb, action_mask):
        logits = self.net(state_emb)
        logits = logits.float()
        logits = torch.clamp(logits, -5.0, 5.0)
        logits = logits / self.temperature
        logits = logits.masked_fill(action_mask == 0, -1e9)
        if (action_mask == 0).all():
            logits = torch.ones_like(logits)
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            valid = torch.nonzero(action_mask).squeeze(-1)
            if len(valid) > 0:
                probs = torch.zeros_like(probs)
                probs[valid] = 1.0 / len(valid)
            else:
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
# 10. 辅助函数（彻底修复索引越界）
# ----------------------------
def collate_states(states):
    """
    合并多个图的状态，对每个图的边进行严格过滤，确保合并后索引完全合法。
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
            # 严格检查并过滤边索引
            valid_mask = (src_i >= 0) & (src_i < num_nodes_i) & (tgt_i >= 0) & (tgt_i < num_nodes_i)
            if not valid_mask.all():
                # 有无效边，只保留有效的
                src_i = src_i[valid_mask]
                tgt_i = tgt_i[valid_mask]
                feat_i = feat_i[valid_mask]
            # 如果该图没有有效边，跳过
            if src_i.numel() > 0:
                # 加上偏移
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

    # 合并后最终检查
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
# 11. PPO训练器（带异常捕获）
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
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)
        self.model_dir = cfg.model_save_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_reward = -np.inf
        self.best_model_path = os.path.join(self.model_dir, "ppo_gat_best.pth")
        self.kl_ema = None
        self.kl_alpha = 0.9
        self.stats_window = cfg.stats_window
        self.recent = {k: deque(maxlen=self.stats_window) for k in ['policy_loss','value_loss','entropy','kl','beta','advantage','value']}
        self.gradient_accumulation_steps = 2
        self._load_model()

    def _save_model(self, reward=None):
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            torch.save({
                'gat': self.gat.state_dict(),
                'policy': self.policy.state_dict(),
                'value': self.value.state_dict(),
                'optim': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'beta': self.beta,
                'entropy_coef': self.entropy_coef,
                'state_norm_mean': self.state_norm.mean if self.state_norm else None,
                'state_norm_var': self.state_norm.var if self.state_norm else None,
                'state_norm_count': self.state_norm.count if self.state_norm else None,
            }, self.best_model_path)
            print(f"New best model saved with reward {reward:.4f}")

    def _load_model(self):
        if os.path.exists(self.best_model_path):
            try:
                ckpt = torch.load(self.best_model_path, map_location=DEVICE)
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
            print("从头开始训练。")

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

                    new_emb, _ = self.gat(merged_state, node_counts=node_counts)
                    new_emb = new_emb.float()
                    new_norm = self.state_norm.normalize(new_emb) if self.state_norm else new_emb / (new_emb.norm(dim=-1, keepdim=True) + 1e-8)

                    probs_new = self.policy(new_norm, mask_batch)

                    if torch.isnan(probs_new).any():
                        continue

                    dist = torch.distributions.Categorical(probs=probs_new)
                    logprobs_new = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    values_new = self.value(new_norm)

                    kl = (batch_old_probs * (torch.log(batch_old_probs + EPS) - torch.log(probs_new + EPS))).sum(-1).mean()
                    if torch.isnan(kl) or torch.isinf(kl):
                        continue

                    ratio = torch.exp(logprobs_new - batch_old_logprobs)
                    ratio = torch.clamp(ratio, 0.1, 10.0)

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 0.9, 1.1) * advantages
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
            self.beta = np.clip(self.beta, 0.001, 1.0)

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
# 12. 多进程工作函数
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
    policy = PolicyNetwork(cfg.gat_out_dim, max_jobs * max_machines, cfg.policy_hidden_dim, temperature=0.5).to('cpu')
    value = ValueNetwork(cfg.gat_out_dim, cfg.policy_hidden_dim).to('cpu')
    state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device='cpu') if cfg.use_state_norm else None

    # 等待主进程下发初始参数
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
                        global_emb, _ = gat(state)
                        global_emb = global_emb.float()
                        if state_norm:
                            emb_norm = state_norm.normalize(global_emb.unsqueeze(0)).squeeze(0)
                        else:
                            emb_norm = global_emb
                        mask = state['action_mask']
                        probs = policy(emb_norm, mask)

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

            # 检查是否有新参数或课程更新
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
        # 清理资源
        del gat, policy, value, state_norm
        gc.collect()

    # 发送剩余轨迹
    if local_trajectories:
        try:
            data_queue.put(local_trajectories, timeout=5)
        except:
            pass

    print(f"Worker {worker_id} finished after {episode_count} episodes.")

# ----------------------------
# 13. 主训练函数
# ----------------------------
def train(cfg):
    print(f"Loading instances from {cfg.instance_dir}...")
    instance_tuples, max_jobs, max_machines, max_proc_time, max_due_date = load_all_instances(cfg.instance_dir,
                                                                                              cfg.file_pattern)
    print(
        f"Loaded {len(instance_tuples)} instances. max_jobs={max_jobs}, max_machines={max_machines}, max_proc_time={max_proc_time}, max_due_date={max_due_date}")

    # 初始化网络
    gat = HomogeneousGAT(8, 1, cfg.gat_hidden_dim, cfg.gat_out_dim,
                         num_layers=cfg.gat_num_layers, num_heads=cfg.gat_num_heads).to(DEVICE)
    policy = PolicyNetwork(cfg.gat_out_dim, max_jobs * max_machines, cfg.policy_hidden_dim, temperature=0.5).to(DEVICE)
    value = ValueNetwork(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
    state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=DEVICE) if cfg.use_state_norm else None
    trainer = PPO_KL_Trainer(gat, policy, value, cfg, state_norm)

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

    # 下发初始参数
    for _ in range(cfg.num_workers):
        param_queue.put(init_params)

    current_allowed_sizes = cfg.curriculum_stages[0][2] if cfg.curriculum_stages else ['small', 'medium', 'large']

    total_episodes = 0
    total_steps = 0
    trajectories_buffer = []
    ep_rewards_buffer = []
    normalized_rewards = []
    kl_episodes = []
    loss_episodes = []
    best_norm_reward = -float('inf')
    pbar = tqdm(total=cfg.n_episodes, desc="Training", unit="ep")
    last_worker_alive_check = time.time()

    try:
        while total_episodes < cfg.n_episodes:
            # 定期检查worker存活状态
            if time.time() - last_worker_alive_check > 10:
                alive = any(p.is_alive() for p in workers)
                if not alive:
                    print("All workers have died. Stopping training.")
                    break
                last_worker_alive_check = time.time()

            try:
                worker_trajectories = data_queue.get(timeout=10)
            except Empty:
                # 超时后再次检查worker状态
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
                ep_rewards_buffer.append(ep_reward)

                # 课程学习进度更新
                if cfg.curriculum_enabled:
                    progress = total_episodes / cfg.n_episodes
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

                # 达到训练步数阈值
                if total_steps >= cfg.train_step_size:
                    batch_size = len(trajectories_buffer)
                    stats = trainer.train_step(trajectories_buffer)
                    trainer.entropy_coef *= cfg.entropy_decay

                    avg_reward = np.mean(ep_rewards_buffer) if ep_rewards_buffer else 0
                    if avg_reward > best_norm_reward:
                        best_norm_reward = avg_reward
                        trainer._save_model(avg_reward)

                    if stats:
                        # 将 KL 和损失重复 batch_size 次，使其与 episode 对齐
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

                    # 下发新参数给所有worker
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
                print(
                    f"[{time.strftime('%H:%M:%S')}] Ep {total_episodes:4d} | NormR {norm_reward:8.4f} | Steps {len(traj):4d}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping workers...")
    finally:
        stop_event.set()
        # 清空队列，让 worker 能够顺利退出
        # 先等待一小段时间，让 worker 响应 stop_event
        time.sleep(1)
        # 持续从 data_queue 中取数据，直到所有 worker 结束
        while any(p.is_alive() for p in workers):
            try:
                # 尝试清空队列，避免 worker 阻塞在 put
                while not data_queue.empty():
                    data_queue.get_nowait()
            except:
                pass
            # 再等待一会
            time.sleep(1)

        # 等待worker结束，最多等20秒
        for p in workers:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # 清理队列（可能已经损坏，尝试安全关闭）
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

        # 处理最后一批未训练的轨迹（使用本地缓冲区，不依赖队列）
        if trajectories_buffer:
            print("Training last batch...")
            batch_size = len(trajectories_buffer)
            stats = trainer.train_step(trajectories_buffer)
            if stats:
                # 确保长度匹配，防止后续绘图错误
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

        # 如果仍有长度不一致，补齐或截断
        if len(kl_episodes) < len(normalized_rewards):
            kl_episodes.extend([kl_episodes[-1] if kl_episodes else 0] * (len(normalized_rewards) - len(kl_episodes)))
        elif len(kl_episodes) > len(normalized_rewards):
            kl_episodes = kl_episodes[:len(normalized_rewards)]
        if len(loss_episodes) < len(normalized_rewards):
            loss_episodes.extend([loss_episodes[-1] if loss_episodes else 0] * (len(normalized_rewards) - len(loss_episodes)))
        elif len(loss_episodes) > len(normalized_rewards):
            loss_episodes = loss_episodes[:len(normalized_rewards)]

    pbar.close()
    print("Training Finished. Best Normalized Reward:", best_norm_reward)

    # 绘制曲线
    plt.figure(figsize=(15, 5))

    def moving_average(data, window_size=20):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # 奖励曲线
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
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # KL 曲线
    plt.subplot(1, 3, 2)
    plt.plot(kl_episodes, alpha=0.3, color='orange', label='Raw')
    if len(kl_episodes) >= 20:
        ma_kl = moving_average(kl_episodes, 20)
        plt.plot(range(19, len(kl_episodes)), ma_kl, color='orange', linewidth=2, label='Moving Avg (20)')
    plt.title("KL Divergence Curve")
    plt.xlabel("Episode")
    plt.ylabel("KL")
    plt.grid(True)
    plt.legend()
    if kl_episodes:
        y_min, y_max = min(kl_episodes), max(kl_episodes)
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        plt.ylim(y_min - margin, y_max + margin)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(loss_episodes, alpha=0.3, color='red', label='Raw')
    if len(loss_episodes) >= 20:
        ma_loss = moving_average(loss_episodes, 20)
        plt.plot(range(19, len(loss_episodes)), ma_loss, color='red', linewidth=2, label='Moving Avg (20)')
    plt.title("Total Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("L(θ)")
    plt.grid(True)
    plt.legend()
    if loss_episodes:
        y_min, y_max = min(loss_episodes), max(loss_episodes)
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        plt.ylim(y_min - margin, y_max + margin)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig("ppo_gat_curves.png")
    plt.show()

if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists(cfg.instance_dir):
        print(f"Warning: Instance directory {cfg.instance_dir} not found. Using current directory.")
        cfg.instance_dir = "."
    mp.set_start_method('spawn', force=True)
    train(cfg)