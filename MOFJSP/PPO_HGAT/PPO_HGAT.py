"""
PPO_HGAT 训练程序
完全实现PPO(使用KL散度,结合层次强化学习)+HGAT(异质析取图)
训练完毕后绘制三个目标值（Cmax, Load Balance, Tardy）的归一化曲线、KL散度及总损失的变化趋势曲线
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
import re

# 1. 全局配置

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

# 多目标权重
W1 = 0.4  # Cmax
W2 = 0.3  # Load Balance
W3 = 0.3  # Tardy

# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 实例目录配置
INSTANCE_DIR = "../mo_fjsp_instances"
FILE_PATTERN = "mo_fjsp_*_train.txt"
MODEL_DIR = "../model/ppo_hgat"
MODEL_NAME = "ppo_hgat_best.pth"

torch.backends.cudnn.benchmark = True

# 2. 配置类

class Config:
    n_episodes = 10000
    lr = 1e-3
    lr_decay = 0.5
    lr_decay_steps = 1000
    min_lr = 1e-5
    gamma = 0.99
    gae_lambda = 0.95
    kl_target = 0.1
    beta_start = 0.2
    beta_lr = 0.01
    value_coef = 0.5
    entropy_coef = 0.01
    entropy_decay = 0.9995
    max_grad_norm = 20.0
    gat_hidden_dim = 256
    gat_out_dim = 256
    policy_hidden_dim = 512
    value_clip = None
    reward_scaling = 5.0
    reward_clip = 2.0
    use_state_norm = False
    use_disjunctive_edges = False
    max_disj_edges_per_machine = 50
    sampling_temperature = 1.0
    max_steps_per_episode = 2000
    stats_window = 100
    max_edges_per_type = 30000
    num_envs = 10
    amp_enabled = False
    logits_clip = 50.0  # 新增：logits 裁剪范围

    # 课程学习参数
    curriculum_enabled = True
    curriculum_stages = [
        (0.0, 0.3, ['small']),
        (0.3, 0.6, ['small', 'medium']),
        (0.6, 1.0, ['small', 'medium', 'large'])
    ]

cfg = Config()


# 3. 数据读取与实例定义

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
        raise ValueError(f"实例文件 {file_path} 数据行不足")

    num_jobs, num_machines = map(int, lines[0].split())
    job_lines = lines[1:1 + num_jobs]
    due_dates_line = lines[-2]
    capabilities_line = lines[-1]
    due_dates = list(map(float, due_dates_line.split()))
    machine_capabilities = list(map(int, capabilities_line.split()))

    if len(due_dates) != num_jobs:
        print(f"警告：交货期数量 ({len(due_dates)}) 与作业数 ({num_jobs}) 不匹配")
    if len(machine_capabilities) != num_machines:
        print(f"警告：机器能力数量 ({len(machine_capabilities)}) 与机器数 ({num_machines}) 不匹配")

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
        raise FileNotFoundError(f"在目录 {instance_dir} 中未找到匹配 {pattern} 的文件")

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

        # 预计算边
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

    def get_pt(self, job, op, machine):
        return self.proc_time_matrix[self.op_index_map[(job, op)], machine].item()


# 4. 课程学习采样器

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
        if not self.cfg.curriculum_enabled:
            self.allowed_sizes = ['small', 'medium', 'large']
            return
        for start, end, sizes in self.cfg.curriculum_stages:
            if start <= episode_progress < end:
                self.allowed_sizes = sizes
                break

    def sample(self) -> MOFJSPInstance:
        allowed_instances = []
        for sz in self.allowed_sizes:
            allowed_instances.extend(self.instances_by_size.get(sz, []))
        if not allowed_instances:
            allowed_instances = self.instances
        return random.choice(allowed_instances)


# 5. 状态归一化工具

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


# 6. 异质图环境

class HeteroGraphEnv:
    def __init__(self, instance: MOFJSPInstance, global_max_jobs, global_max_ops, global_max_machines,
                 global_max_proc_time, global_max_due_date, cfg):
        self.inst = instance
        self.global_max_jobs = global_max_jobs
        self.global_max_ops = global_max_ops
        self.global_max_machines = global_max_machines
        self.global_max_proc_time = global_max_proc_time
        self.global_max_due_date = global_max_due_date
        self.cfg = cfg

        self.global_max_proc_time_tensor = torch.tensor(global_max_proc_time, device=DEVICE)
        self.global_max_due_date_tensor = torch.tensor(global_max_due_date, device=DEVICE)

        self.total_op_nodes = instance.total_ops
        self.total_machine_nodes = instance.n_machines

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
        for job in range(instance.n_jobs):
            for op in range(instance.ops_per_job[job]):
                node_idx = instance.op_index_map[(job, op)]
                step_idx = op / max(1, instance.ops_per_job[job] - 1)
                due_time = instance.due_dates[job] / (global_max_due_date + EPS)
                job_id_norm = job / max(1, global_max_jobs - 1)
                self.static_op_feats[node_idx, :] = torch.tensor([step_idx, due_time, job_id_norm], device=DEVICE)

        self.static_mac_feats = torch.zeros((self.global_max_machines, self.mac_feat_dim), dtype=torch.float32, device=DEVICE)
        for m in range(instance.n_machines):
            self.static_mac_feats[m, 2] = instance.capability_tensor[m]

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

        self.machine_available_time[machine.long()] = end
        self.machine_load[machine.long()] += p_time
        self.job_next_op[job.long()] = self.job_next_op[job.long()] + 1
        self.job_completion_time[job.long()] = torch.maximum(self.job_completion_time[job.long()], end)
        self.current_time = torch.maximum(self.current_time, end)
        self.op_finish_time[op_idx] = end
        self.op_scheduled[op_idx] = True

        # 即时奖励
        old_LB = torch.std(self.machine_load)
        temp_load = self.machine_load.clone()
        temp_load[machine.long()] += p_time
        new_LB = torch.std(temp_load)
        delta_LB = new_LB - old_LB

        norm_p = p_time / self.global_max_proc_time_tensor
        avg_load = torch.mean(self.machine_load) + 1e-8
        norm_LB = delta_LB / avg_load

        due = self.due_dates_tensor[job.long()]
        t_c = self.current_time
        n_i = self.ops_per_job_tensor[job.long()]
        u_ij = torch.clamp(due - t_c, min=0) / (self.global_max_due_date_tensor * n_i + EPS)

        reward = - (W1 * norm_p + W2 * norm_LB + W3 * u_ij)
        reward /= self.cfg.reward_scaling
        reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        done = self.op_scheduled.all().item()
        if done:
            # 最终全局惩罚
            c_max = self.current_time
            avg_load = torch.mean(self.machine_load)
            lb = torch.sqrt(torch.mean((self.machine_load - avg_load) ** 2))
            t_tardy = torch.sum(torch.clamp(self.job_completion_time - self.due_dates_tensor, min=0))

            global_penalty = (W1 * c_max / 100.0 + W2 * lb / 50.0 + W3 * t_tardy / 50.0)
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
        seq_feat_final = torch.ones((seq_src_final.shape[0], 1), dtype=torch.float32, device=DEVICE)

        alloc_valid = unsched_mask[self.alloc_edges_src]
        alloc_src_final = self.alloc_edges_src[alloc_valid]
        alloc_dst_final = self.alloc_edges_dst[alloc_valid]
        alloc_feat_final = self.alloc_edges_feat[alloc_valid]

        alloc_rev_valid = unsched_mask[self.alloc_rev_edges_dst]
        alloc_rev_src_final = self.alloc_rev_edges_src[alloc_rev_valid]
        alloc_rev_dst_final = self.alloc_rev_edges_dst[alloc_rev_valid]
        alloc_rev_feat_final = self.alloc_rev_edges_feat[alloc_rev_valid]

        disj_src_final = torch.empty(0, dtype=torch.long, device=DEVICE)
        disj_dst_final = torch.empty(0, dtype=torch.long, device=DEVICE)
        disj_feat_final = torch.empty((0, 1), dtype=torch.float32, device=DEVICE)

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
            'op_feats': self.op_feats_t,
            'mac_feats': self.mac_feats_t,
            'edges': edges_t,
            'edge_feats': edge_feats_t,
            'op_mask': op_mask,
        }


# 7. 批处理环境管理器

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


# 8. 异质图神经网络

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

        def apply_attn(src_emb, dst_emb, src_idx, dst_idx, feat_e, att_param, out_buffer):
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
            alpha = torch.exp(scores).to(dtype_out)

            denom = torch.zeros(out_buffer.shape[0], self.heads, device=device, dtype=dtype_out)
            denom.index_add_(0, dst_idx, alpha)
            denom = denom[dst_idx] + torch.tensor(EPS, dtype=dtype_out, device=device)
            norm_alpha = alpha / denom
            weighted = h_s * norm_alpha.unsqueeze(-1)
            out_buffer.index_add_(0, dst_idx, weighted)

        if edges['seq'][0].numel() > 0:
            apply_attn(z_op, z_op, edges['seq'][0], edges['seq'][1], edge_feats_proc['seq'], self.att_seq, out_op)
        if edges['op_mac'][0].numel() > 0:
            apply_attn(z_op, z_mac, edges['op_mac'][0], edges['op_mac'][1], edge_feats_proc['op_mac'], self.att_op_mac, out_mac)
        if edges['mac_op'][0].numel() > 0:
            apply_attn(z_mac, z_op, edges['mac_op'][0], edges['mac_op'][1], edge_feats_proc['mac_op'], self.att_mac_op, out_op)

        res_op = (out_op.flatten(1) + self.W_op(h_op))
        res_mac = (out_mac.flatten(1) + self.W_mac(h_mac))
        return F.elu(res_op), F.elu(res_mac)

class HeteroGAT(nn.Module):
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


# 9. 分层策略网络（改进数值稳定）

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
            if mask is not None:
                # 使用 -1e9 代替 -inf，避免 softmax 产生 NaN
                fill_value = -1e9
                logits = logits.masked_fill(mask == 0, fill_value)
            # 裁剪 logits 防止溢出
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            # 如果整行都是 fill_value，则返回均匀分布
            if (logits == fill_value).all():
                probs = torch.ones(N, device=logits.device) / N
            else:
                probs = F.softmax(logits, dim=0)
            probs = torch.clamp(probs, min=EPS, max=1.0)
            probs = probs / probs.sum()
        else:
            batch_size, N, _ = op_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, N, -1)
            cat = torch.cat([g_rep, op_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            if mask is not None:
                fill_value = -1e9
                logits = logits.masked_fill(mask == 0, fill_value)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            probs = torch.zeros_like(logits)
            for i in range(batch_size):
                row = logits[i]
                if (row == fill_value).all():
                    probs[i] = torch.ones(N, device=logits.device) / N
                else:
                    probs[i] = F.softmax(row, dim=0)
            probs = torch.clamp(probs, min=EPS, max=1.0)
            row_sums = probs.sum(dim=1, keepdim=True)
            probs = probs / row_sums
        return probs

class LowerPolicy(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim * 3, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=0.1)
        )

    def forward(self, g_emb, selected_op_emb, mac_embs, mask=None):
        g_emb = g_emb.float()
        selected_op_emb = selected_op_emb.float()
        mac_embs = mac_embs.float()
        if g_emb.dim() == 1:
            M = mac_embs.shape[0]
            g_rep = g_emb.unsqueeze(0).expand(M, -1)
            op_rep = selected_op_emb.unsqueeze(0).expand(M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=1)
            logits = self.net(cat).squeeze(-1)
            if mask is not None:
                fill_value = -1e9
                logits = logits.masked_fill(mask == 0, fill_value)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            if (logits == fill_value).all():
                probs = torch.ones(M, device=logits.device) / M
            else:
                probs = F.softmax(logits, dim=0)
            probs = torch.clamp(probs, min=EPS, max=1.0)
            probs = probs / probs.sum()
        else:
            batch_size, M, _ = mac_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, M, -1)
            op_rep = selected_op_emb.unsqueeze(1).expand(-1, M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
            if mask is not None:
                fill_value = -1e9
                logits = logits.masked_fill(mask == 0, fill_value)
            logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
            probs = torch.zeros_like(logits)
            for i in range(batch_size):
                row = logits[i]
                if (row == fill_value).all():
                    probs[i] = torch.ones(M, device=logits.device) / M
                else:
                    probs[i] = F.softmax(row, dim=0)
            probs = torch.clamp(probs, min=EPS, max=1.0)
            row_sums = probs.sum(dim=1, keepdim=True)
            probs = probs / row_sums
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


# 10. 批处理状态打包

def batch_states(states: List[Dict], n_op: int, n_mac: int):
    batch_size = len(states)
    op_feats_list = [s['op_feats'] for s in states]
    mac_feats_list = [s['mac_feats'] for s in states]
    op_mask_list = [s['op_mask'] for s in states]

    batch_op_feats = torch.cat(op_feats_list, dim=0)
    batch_mac_feats = torch.cat(mac_feats_list, dim=0)
    batch_op_mask = torch.cat(op_mask_list, dim=0)

    op_offsets = torch.arange(batch_size, device=DEVICE) * n_op
    mac_offsets = torch.arange(batch_size, device=DEVICE) * n_mac

    edge_types = ['seq', 'disj', 'op_mac', 'mac_op']
    batch_edges = {k: ([], []) for k in edge_types}
    batch_edge_feats = {k: [] for k in edge_types}

    for i, s in enumerate(states):
        op_offset = op_offsets[i]
        mac_offset = mac_offsets[i]
        for k in edge_types:
            if k == 'disj':
                continue
            src, dst = s['edges'][k]
            feat = s['edge_feats'][k]
            if src.numel() == 0:
                continue
            if k == 'seq':
                src_off = src + op_offset
                dst_off = dst + op_offset
            elif k == 'op_mac':
                src_off = src + op_offset
                dst_off = dst + mac_offset
            elif k == 'mac_op':
                src_off = src + mac_offset
                dst_off = dst + op_offset
            else:
                raise ValueError(f"Unknown edge type: {k}")
            batch_edges[k][0].append(src_off)
            batch_edges[k][1].append(dst_off)
            batch_edge_feats[k].append(feat)

    for k in edge_types:
        if k == 'disj':
            batch_edges[k] = (torch.empty(0, dtype=torch.long, device=DEVICE),
                              torch.empty(0, dtype=torch.long, device=DEVICE))
            batch_edge_feats[k] = torch.empty((0, 1), dtype=torch.float32, device=DEVICE)
            continue
        if not batch_edges[k][0]:
            batch_edges[k] = (torch.empty(0, dtype=torch.long, device=DEVICE),
                              torch.empty(0, dtype=torch.long, device=DEVICE))
            batch_edge_feats[k] = torch.empty((0, 1), dtype=torch.float32, device=DEVICE)
            continue

        src_all = torch.cat(batch_edges[k][0])
        dst_all = torch.cat(batch_edges[k][1])
        feat_all = torch.cat(batch_edge_feats[k], dim=0)

        num_edges = src_all.shape[0]
        if num_edges > cfg.max_edges_per_type:
            perm = torch.randperm(num_edges, device=DEVICE)[:cfg.max_edges_per_type]
            src_all = src_all[perm]
            dst_all = dst_all[perm]
            feat_all = feat_all[perm]

        batch_edges[k] = (src_all, dst_all)
        batch_edge_feats[k] = feat_all

    batch_state_dict = {
        'op_feats': batch_op_feats,
        'mac_feats': batch_mac_feats,
        'edges': batch_edges,
        'edge_feats': batch_edge_feats,
    }

    state_slices = torch.zeros((batch_size, 4), dtype=torch.long, device=DEVICE)
    for i in range(batch_size):
        state_slices[i, 0] = i * n_op
        state_slices[i, 1] = (i+1) * n_op
        state_slices[i, 2] = i * n_mac
        state_slices[i, 3] = (i+1) * n_mac

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

    def save(self, path):
        torch.save({
            'gat': self.gat.state_dict(),
            'upper': self.upper.state_dict(),
            'lower': self.lower.state_dict(),
            'critic_u': self.critic_u.state_dict(),
            'critic_l': self.critic_l.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'beta': self.beta,
            'entropy_coef': self.entropy_coef,
            'episode_counter': self.episode_counter
        }, path)
        print(f"模型已保存至 {path}")

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

            self.gat = self.gat.to(DEVICE).float()
            self.upper = self.upper.to(DEVICE).float()
            self.lower = self.lower.to(DEVICE).float()
            self.critic_u = self.critic_u.to(DEVICE).float()
            self.critic_l = self.critic_l.to(DEVICE).float()

            self.params = list(self.gat.parameters()) + list(self.upper.parameters()) + \
                          list(self.lower.parameters()) + list(self.critic_u.parameters()) + list(self.critic_l.parameters())
            self.opt = torch.optim.Adam(self.params, lr=self.cfg.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay)

            self.beta = checkpoint.get('beta', self.beta)
            self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            self.episode_counter = checkpoint.get('episode_counter', 0)

            print(f"模型已从 {path} 加载并转换为 float32，beta={self.beta:.4f}, entropy_coef={self.entropy_coef:.4f}")
            return True
        else:
            print(f"未找到模型文件 {path}，从头开始训练。")
            return False

    def get_action_batch(self, states: List[Dict], insts: List[MOFJSPInstance], deterministic=False):
        batch_size = len(states)
        n_op_global = states[0]['op_feats'].shape[0]
        n_mac_global = states[0]['mac_feats'].shape[0]

        # 提前检测 op_mask 全零的状态（已完成环境）
        invalid_indices = []
        for i, s in enumerate(states):
            if s['op_mask'].sum().item() == 0:
                invalid_indices.append(i)

        if invalid_indices:
            actions = [None] * batch_size
            probs_list = [None] * batch_size
            indices_list = [None] * batch_size
            for i in invalid_indices:
                # 分配默认动作 (0,0,0) 并跳过网络计算
                actions[i] = (0, 0, 0)
                probs_list[i] = (1.0, 1.0)
                indices_list[i] = (0, 0)
                print(f"警告：状态 {i} 的 op_mask 全零，分配默认动作 (0,0,0)")
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

        # 正常流程
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

            # 确保每行概率有效
            for i in range(batch_size):
                if torch.isnan(u_probs_batch[i]).any() or u_probs_batch[i].sum().item() < EPS:
                    print(f"严重错误：Upper 概率分布第 {i} 行无效，请检查网络输出")
                    print(f"概率分布: {u_probs_batch[i]}")
                    raise RuntimeError("概率分布无效，训练中止")

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
                if op_idx_local < inst.total_ops:
                    batch_mac_masks_op[i, :inst.n_machines] = inst.op_mac_mask[op_idx_local]

            l_probs_batch = self.lower(g_embs, selected_op_embs, batch_h_mac, batch_mac_masks_op)

            for i in range(batch_size):
                if torch.isnan(l_probs_batch[i]).any() or l_probs_batch[i].sum().item() < EPS:
                    print(f"严重错误：Lower 概率分布第 {i} 行无效，请检查网络输出")
                    print(f"概率分布: {l_probs_batch[i]}")
                    raise RuntimeError("概率分布无效，训练中止")

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

        chunk_size = 256

        # 优势计算
        with torch.no_grad():
            batch_state_dict, state_slices, _ = batch_states(states, n_op_global, n_mac_global)
            h_op_all, h_mac_all = self.gat(batch_state_dict)

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

            # 优势归一化
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

            del h_op_all, h_mac_all, batch_state_dict, state_slices, g_embs, values, next_values, deltas, rev_deltas, rev_advantages
            torch.cuda.empty_cache()

        total_kl = 0.0
        total_policy_loss_u = 0.0
        total_policy_loss_l = 0.0
        total_value_loss = 0.0
        valid_chunks = 0

        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_len = chunk_end - chunk_start

            chunk_states = states[chunk_start:chunk_end]
            chunk_op_indices = op_indices[chunk_start:chunk_end]
            chunk_mac_indices = mac_indices[chunk_start:chunk_end]
            chunk_advantages = advantages[chunk_start:chunk_end]
            chunk_returns = returns[chunk_start:chunk_end]
            chunk_old_u_probs = old_u_probs[chunk_start:chunk_end]
            chunk_old_l_probs = old_l_probs[chunk_start:chunk_end]

            chunk_batch_state_dict, chunk_state_slices, _ = batch_states(chunk_states, n_op_global, n_mac_global)

            h_op_all, h_mac_all = self.gat(chunk_batch_state_dict)

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

            curr_u_probs_batch = self.upper(g_embs_chunk, batch_h_op, batch_op_masks)
            curr_u_probs = curr_u_probs_batch[torch.arange(chunk_len), chunk_op_indices]
            curr_u_probs = torch.clamp(curr_u_probs, EPS, 1.0 - EPS)
            chunk_old_u_probs_clamped = torch.clamp(chunk_old_u_probs, EPS, 1.0 - EPS)

            batch_mac_masks_op = torch.zeros((chunk_len, n_mac_global), device=DEVICE)
            for j in range(chunk_len):
                idx = chunk_start + j
                if op_indices[idx] < inst.total_ops:
                    batch_mac_masks_op[j, :inst.n_machines] = inst.op_mac_mask[op_indices[idx]]

            selected_op_embs = batch_h_op[torch.arange(chunk_len), chunk_op_indices]
            curr_l_probs_batch = self.lower(g_embs_chunk, selected_op_embs, batch_h_mac, batch_mac_masks_op)
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

            # 标准KL散度 KL(old||new)，确保数值稳定
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

            entropy_u = -(curr_u_probs * torch.log(curr_u_probs + EPS)).mean()
            entropy_l = -(curr_l_probs * torch.log(curr_l_probs + EPS)).mean()
            entropy = entropy_u + entropy_l

            total_loss = policy_loss_u + policy_loss_l + self.value_coef * value_loss + self.beta * kl - self.entropy_coef * entropy

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"警告：损失为 NaN/Inf 在块 {chunk_start}-{chunk_end}，跳过此块。")
                del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
                del curr_u_probs_batch, curr_l_probs_batch, selected_op_embs
                del ratio_u, ratio_l, surr1_u, surr2_u, surr1_l, surr2_l
                del policy_loss_u, policy_loss_l, kl, v_pred, value_loss, total_loss
                torch.cuda.empty_cache()
                continue

            self.opt.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"警告：梯度范数为 NaN/Inf，跳过此块更新。")
                self.opt.zero_grad()
            else:
                self.opt.step()

            total_kl += kl.item() * chunk_len
            total_policy_loss_u += policy_loss_u.item() * chunk_len
            total_policy_loss_l += policy_loss_l.item() * chunk_len
            total_value_loss += value_loss.item() * chunk_len
            valid_chunks += 1

            del h_op_all, h_mac_all, g_embs_chunk, batch_h_op, batch_h_mac, batch_op_masks
            del curr_u_probs_batch, curr_l_probs_batch, selected_op_embs
            del ratio_u, ratio_l, surr1_u, surr2_u, surr1_l, surr2_l
            del policy_loss_u, policy_loss_l, kl, v_pred, value_loss, total_loss
            torch.cuda.empty_cache()

            self.update_counter += 1
            if self.update_counter % 10 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        if valid_chunks == 0:
            print("警告：所有更新块均被跳过，无有效更新。")
            return None

        avg_kl = total_kl / T
        avg_policy_loss_u = total_policy_loss_u / T
        avg_policy_loss_l = total_policy_loss_l / T
        avg_value_loss = total_value_loss / T

        if self.kl_ema is None:
            self.kl_ema = avg_kl
        else:
            self.kl_ema = self.kl_alpha * self.kl_ema + (1 - self.kl_alpha) * avg_kl
        kl_error = self.kl_ema - self.cfg.kl_target
        self.beta *= (1 + self.beta_lr * kl_error)
        self.beta = np.clip(self.beta, 0.01, 10.0)

        self.entropy_coef *= self.cfg.entropy_decay

        self.recent_policy_loss_u.append(avg_policy_loss_u)
        self.recent_policy_loss_l.append(avg_policy_loss_l)
        self.recent_value_loss.append(avg_value_loss)
        self.recent_kl.append(avg_kl)
        self.recent_beta.append(self.beta)
        self.recent_advantage.append(advantages.mean().item())
        self.recent_return.append(returns.mean().item())

        stats = {
            'policy_loss_u': avg_policy_loss_u,
            'policy_loss_l': avg_policy_loss_l,
            'value_loss': avg_value_loss,
            'kl': avg_kl,
            'beta': self.beta,
            'advantage': advantages.mean().item(),
            'return': returns.mean().item(),
            'total_loss': (
                        avg_policy_loss_u + avg_policy_loss_l + self.value_coef * avg_value_loss + self.beta * avg_kl)
        }
        return stats


# 12. 绘图辅助函数（新增）

def plot_three_goals(cmax_list, lb_list, tardy_list, save_path):
    """绘制三个目标值随 episode 变化的子图（原始值）"""
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
    print(f"三个目标曲线（原始值）已保存至 {save_path}")

def plot_three_goals_normalized(cmax_list, lb_list, tardy_list, save_path):
    """绘制三个目标值随 episode 变化的子图（归一化至[0,1]）"""
    # 计算每个目标的 Min-Max 归一化因子
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
    print(f"三个目标曲线（归一化）已保存至 {save_path}")

def plot_kl_divergence(kl_list, save_path):
    """单独绘制 KL 散度曲线"""
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
    if kl_list:
        y_min, y_max = min(kl_list), max(kl_list)
        margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.01
        plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"KL散度曲线已保存至 {save_path}")

def plot_loss(loss_list, save_path):
    """单独绘制总损失曲线"""
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
    if loss_list:
        y_min, y_max = min(loss_list), max(loss_list)
        margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.01
        plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"总损失曲线已保存至 {save_path}")


# 13. 主训练函数

def train():
    cfg.use_disjunctive_edges = False
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    print(f"正在从目录 {INSTANCE_DIR} 加载所有实例...")
    instance_tuples, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = load_all_instances(INSTANCE_DIR, FILE_PATTERN)
    print(f"共加载 {len(instance_tuples)} 个实例")
    print(f"全局最大作业数: {max_jobs}, 最大机器数: {max_machines}, 最大总工序数: {max_total_ops}, 最大加工时间: {max_proc_time}, 最大交货期: {max_due_date}, 最大机器能力: {max_capability}")

    # 构建实例对象
    instance_list = []
    for jobs, caps, dues, fname, size in instance_tuples:
        inst = MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability)
        instance_list.append(inst)

    # 课程学习采样器
    curriculum_sampler = CurriculumSampler(instance_list, cfg)

    global_max_ops = max(inst.total_ops for inst in instance_list)
    global_max_machines = max(inst.n_machines for inst in instance_list)
    print(f"全局最大工序数: {global_max_ops}, 全局最大机器数: {global_max_machines}")

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
    agent.load(best_model_path)

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

    total_episodes = 0
    normalized_rewards = []
    kl_episodes = []
    loss_episodes = []
    best_norm_reward = -float('inf')
    reward_ema = None

    # 新增三个目标列表
    cmax_list = []
    lb_list = []
    tardy_list = []

    pbar = tqdm(total=cfg.n_episodes, desc="Training", unit="ep")

    log_interval = 100
    empty_cache_interval = 50

    while total_episodes < cfg.n_episodes:
        progress = total_episodes / cfg.n_episodes
        curriculum_sampler.update_stage(progress)

        # 检查 states 中是否有 None
        for i in range(num_envs):
            if states[i] is None:
                print(f"检测到状态 {i} 为 None，强制重置该环境")
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
            print(f"采样或执行步时发生异常：{e}，重置所有环境并继续")
            for j in range(num_envs):
                new_inst = curriculum_sampler.sample()
                envs[j] = HeteroGraphEnv(new_inst, max_jobs, global_max_ops, global_max_machines, max_proc_time, max_due_date, cfg)
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

                if total_episodes % cfg.lr_decay_steps == 0 and total_episodes > 0:
                    agent.scheduler.step()
                    for param_group in agent.opt.param_groups:
                        param_group['lr'] = max(param_group['lr'], cfg.min_lr)

                if total_episodes % log_interval == 0 and stats is not None:
                    lr = agent.opt.param_groups[0]['lr']
                    print(f"\nEpisode {total_episodes}: LR={lr:.2e}, Beta={stats['beta']:.3f}, "
                          f"Adv={stats['advantage']:.4f}, KL={stats['kl']:.4f}, "
                          f"Entropy={agent.entropy_coef:.4f}, AllowedSizes={curriculum_sampler.allowed_sizes}")

                if norm_reward > best_norm_reward:
                    best_norm_reward = norm_reward
                    agent.save(best_model_path)

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

        if total_episodes % empty_cache_interval == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    pbar.close()
    print("训练完成。")
    print(f"最佳归一化奖励: {best_norm_reward:.4f}")

    # 保存曲线图
    os.makedirs("../Figure_And_File/PPO_HGAT/Train", exist_ok=True)
    plot_three_goals(cmax_list, lb_list, tardy_list, "../Figure_And_File/PPO_HGAT/Train/three_goals.png")
    plot_three_goals_normalized(cmax_list, lb_list, tardy_list, "../Figure_And_File/PPO_HGAT/Train/three_goals_normalized.png")
    plot_kl_divergence(kl_episodes, "../Figure_And_File/PPO_HGAT/Train/kl_divergence.png")
    plot_loss(loss_episodes, "../Figure_And_File/PPO_HGAT/Train/total_loss.png")

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
        print(f"已生成示例实例文件: {example_file}")

    train()