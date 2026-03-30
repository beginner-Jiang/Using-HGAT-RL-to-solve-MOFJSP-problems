"""
参数敏感度检测脚本（优化版）
基于已训练的最佳模型，对以下参数进行扫描：
- 多目标权重 w (w1, w2, w3)
- KL惩罚系数 beta
- 折扣因子 gamma
- 注意力头数 K

对于每个参数的不同取值，重新训练模型（训练轮数可设），在测试集上评估平均加权目标值及各子目标值，
并记录收敛所需的训练轮数。最后绘制曲线图并输出表格。
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
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import re

# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10

W1_DEFAULT = 0.4
W2_DEFAULT = 0.3
W3_DEFAULT = 0.3

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

INSTANCE_DIR = "../mo_fjsp_instances"
FILE_PATTERN = "mo_fjsp_*_train.txt"
MODEL_DIR = "../model/ppo_hgat"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hgat_best.pth")

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

NUM_EPOCHS = 10

REWARD_SCALING = 5.0
REWARD_CLIP = 2.0

# GPU 版归一化工具
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

# 数据读取与实例定义
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

# 异质析取图环境
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

        old_LB = torch.std(self.machine_load)
        temp_load = self.machine_load.clone()
        temp_load[machine.long()] += p_time
        new_LB = torch.std(temp_load)
        delta_LB = new_LB - old_LB

        norm_p = p_time / self.global_max_proc_time_tensor
        due = self.due_dates_tensor[job.long()]
        t_c = self.current_time
        n_i = self.ops_per_job_tensor[job.long()]
        u_ij = torch.clamp(due - t_c, min=0) / (self.global_max_due_date_tensor * n_i + EPS)

        reward = - (self.cfg.w1 * norm_p + self.cfg.w2 * delta_LB / self.global_max_proc_time_tensor + self.cfg.w3 * u_ij)
        reward /= self.cfg.reward_scaling
        reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        done = self.op_scheduled.all().item()
        if done:
            c_max = self.current_time
            avg_load = torch.mean(self.machine_load)
            lb = torch.sqrt(torch.mean((self.machine_load - avg_load) ** 2))
            t_tardy = torch.sum(torch.clamp(self.job_completion_time - self.due_dates_tensor, min=0))

            global_penalty = (self.cfg.w1 * c_max / 100.0 + self.cfg.w2 * lb / 50.0 + self.cfg.w3 * t_tardy / 50.0)
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

# 异质图神经网络
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

        self.leaky = nn.LeakyReLU(0.2)
        self.max_edges = 300000

    def forward(self, h_op, h_mac, edges, edge_feats):
        dtype = h_op.dtype
        device = h_op.device

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

        def apply_attn(src_emb, dst_emb, src_idx, dst_idx, feat_e, att_param, out_buffer):
            if src_idx.numel() == 0:
                return
            if src_idx.numel() > self.max_edges:
                return
            h_s = src_emb[src_idx]
            h_d = dst_emb[dst_idx]
            att_param = att_param.to(dtype)
            f_e = feat_e.unsqueeze(1).repeat(1, self.heads, 1)
            concat = torch.cat([h_s, h_d, f_e], dim=-1)
            scores = (concat * att_param).sum(dim=-1)
            scores = self.leaky(scores)
            alpha = torch.exp(scores)

            denom = torch.zeros(out_buffer.shape[0], self.heads, device=device, dtype=dtype)
            denom.index_add_(0, dst_idx, alpha)
            denom = denom[dst_idx] + torch.tensor(EPS, dtype=dtype, device=device)
            norm_alpha = alpha / denom
            weighted = h_s * norm_alpha.unsqueeze(-1)
            weighted = weighted.to(dtype)
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
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e10)
            probs = F.softmax(logits, dim=0)
        else:
            batch_size, N, _ = op_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, N, -1)
            cat = torch.cat([g_rep, op_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
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

    def forward(self, g_emb, selected_op_emb, mac_embs, mask=None):
        if g_emb.dim() == 1:
            M = mac_embs.shape[0]
            g_rep = g_emb.unsqueeze(0).expand(M, -1)
            op_rep = selected_op_emb.unsqueeze(0).expand(M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=1)
            logits = self.net(cat).squeeze(-1)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e10)
            probs = F.softmax(logits, dim=0)
        else:
            batch_size, M, _ = mac_embs.shape
            g_rep = g_emb.unsqueeze(1).expand(-1, M, -1)
            op_rep = selected_op_emb.unsqueeze(1).expand(-1, M, -1)
            cat = torch.cat([g_rep, op_rep, mac_embs], dim=2)
            logits = self.net(cat).squeeze(-1)
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
            print(f"模型已从 {path} 加载，gamma={self.gamma:.3f}, beta={self.beta:.3f}")
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
        l_probs = self.lower(g_emb, selected_op_emb, h_mac, mac_mask)

        if deterministic:
            mac_idx = torch.argmax(l_probs)
        else:
            dist_l = torch.distributions.Categorical(l_probs)
            mac_idx = dist_l.sample()

        machine = mac_idx

        return (job, op, machine), (u_probs[op_idx], l_probs[mac_idx]), (op_idx, mac_idx)

    def update(self, trajectory):
        T = len(trajectory)
        if T == 0:
            return

        states = [t['state'] for t in trajectory]
        rewards = torch.tensor([t['reward'] for t in trajectory], dtype=torch.float32, device=DEVICE)
        old_u_probs = torch.stack([t['old_probs'][0] for t in trajectory]).detach()
        old_l_probs = torch.stack([t['old_probs'][1] for t in trajectory]).detach()
        op_indices = torch.tensor([t['indices'][0] for t in trajectory], device=DEVICE)
        mac_indices = torch.tensor([t['indices'][1] for t in trajectory], device=DEVICE)

        # 计算价值估计和GAE
        values_u = []
        with torch.no_grad():
            for s in states:
                h_op, _ = self.gat(s)
                g_emb = h_op.mean(dim=0)
                if self.state_norm is not None:
                    g_emb = self.state_norm.normalize(g_emb)
                values_u.append(self.critic_u(g_emb).item())
        values_u = torch.tensor(values_u, device=DEVICE)

        advantages = []
        gae = 0
        for i in reversed(range(T)):
            next_val = values_u[i+1] if i < T-1 else 0.0
            delta = rewards[i] + self.gamma * next_val - values_u[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, device=DEVICE)
        returns = values_u + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        # 逐时间步更新
        self.opt.zero_grad()
        total_loss = 0.0
        for i in range(T):
            s = states[i]
            h_op, h_mac = self.gat(s)
            g_emb = h_op.mean(dim=0)
            if self.state_norm is not None:
                g_emb = self.state_norm.normalize(g_emb)

            op_mask = s['op_mask']
            curr_u_probs = self.upper(g_emb, h_op, op_mask)
            curr_u_prob = curr_u_probs[op_indices[i]]

            mac_mask = torch.zeros(h_mac.shape[0], device=DEVICE)
            if op_indices[i] < self.inst.total_ops:
                mac_mask[:self.inst.n_machines] = self.inst.op_mac_mask[op_indices[i]]
            selected_op_emb = h_op[op_indices[i]]
            curr_l_probs = self.lower(g_emb, selected_op_emb, h_mac, mac_mask)
            curr_l_prob = curr_l_probs[mac_indices[i]]

            ratio_u = curr_u_prob / (old_u_probs[i] + EPS)
            ratio_l = curr_l_prob / (old_l_probs[i] + EPS)
            adv = advantages[i]

            surr1_u = ratio_u * adv
            surr2_u = torch.clamp(ratio_u, 1-self.clip, 1+self.clip) * adv
            loss_u = -torch.min(surr1_u, surr2_u)

            surr1_l = ratio_l * adv
            surr2_l = torch.clamp(ratio_l, 1-self.clip, 1+self.clip) * adv
            loss_l = -torch.min(surr1_l, surr2_l)

            kl_u = torch.log(old_u_probs[i] / (curr_u_prob + EPS) + EPS)
            kl_l = torch.log(old_l_probs[i] / (curr_l_prob + EPS) + EPS)

            v_pred = self.critic_u(g_emb).squeeze()
            loss_v = F.mse_loss(v_pred, returns[i])

            loss = loss_u + loss_l + 0.5 * loss_v + self.beta * (kl_u + kl_l)
            total_loss += loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 0.5)
        self.opt.step()

# 评估函数
def evaluate(agent, test_instances, global_max_jobs, global_max_ops, global_max_machines, global_max_proc_time, global_max_due_date,
             cfg, deterministic=True):
    """
    评估 agent 在测试实例上的性能。
    test_instances: MOFJSPInstance 对象列表
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
        # 直接迭代 MOFJSPInstance 对象，无需解包
        for inst in tqdm(test_instances, desc="Evaluating", leave=False):
            agent.inst = inst  # 设置当前实例供 get_action 使用
            env = HeteroGraphEnv(inst, global_max_jobs, global_max_ops, global_max_machines,
                                 global_max_proc_time, global_max_due_date, cfg)
            state = env.reset()
            done = False
            while not done:
                action, _, _ = agent.get_action(state, deterministic=deterministic)
                next_state, _, done = env.step(action)
                state = next_state
            Cmax = max(env.job_completion_time).item()
            LB = torch.std(env.machine_load).item() if env.machine_load.numel() > 0 else 0.0
            Tardy = torch.sum(torch.clamp(env.job_completion_time - env.due_dates_tensor, min=0)).item()
            F_val = cfg.w1 * Cmax + cfg.w2 * LB + cfg.w3 * Tardy
            F_list.append(F_val)
            Cmax_list.append(Cmax)
            LB_list.append(LB)
            Tardy_list.append(Tardy)

    return np.mean(F_list), np.mean(Cmax_list), np.mean(LB_list), np.mean(Tardy_list)

# 参数敏感度检测主函数
def parameter_sensitivity_scan():
    # 加载所有实例并划分
    print("加载实例...")
    all_instances, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = load_all_instances(INSTANCE_DIR, FILE_PATTERN)
    random.shuffle(all_instances)
    total = len(all_instances)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)
    train_tuples = all_instances[:train_end]
    val_tuples = all_instances[train_end:val_end]
    test_tuples = all_instances[val_end:]
    print(f"训练集: {len(train_tuples)}, 验证集: {len(val_tuples)}, 测试集: {len(test_tuples)}")
    print(f"全局最大作业数: {max_jobs}, 最大机器数: {max_machines}, 最大总工序数: {max_total_ops}, 最大加工时间: {max_proc_time}, 最大交货期: {max_due_date}, 最大能力值: {max_capability}")

    # 包装实例
    train_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability) for jobs, caps, dues, fname, size in train_tuples]
    val_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability) for jobs, caps, dues, fname, size in val_tuples]
    test_instances = [MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability) for jobs, caps, dues, fname, size in test_tuples]

    dim_op = 3
    dim_mac = 3
    hidden = 64
    out = 64
    lr = 3e-4

    # 定义配置类
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
        'w': [
            {'w1': 0.2, 'w2': 0.4, 'w3': 0.4},
            {'w1': 0.33, 'w2': 0.33, 'w3': 0.34},
            {'w1': 0.4, 'w2': 0.3, 'w3': 0.3},
            {'w1': 0.5, 'w2': 0.25, 'w3': 0.25},
            {'w1': 0.6, 'w2': 0.2, 'w3': 0.2}
        ],
        'beta': [0.001, 0.005, 0.01, 0.02, 0.05],
        'gamma': [0.90, 0.95, 0.99, 0.995, 0.999],
        'K': [1, 2, 4, 8]
    }

    results = {
        'w': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'beta': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'gamma': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []},
        'K': {'values': [], 'F': [], 'Cmax': [], 'LB': [], 'Tardy': [], 'epochs': []}
    }

    # 计算总扫描次数用于进度条
    total_scans = len(param_configs['w']) + len(param_configs['beta']) + len(param_configs['gamma']) + len(param_configs['K'])
    pbar_scan = tqdm(total=total_scans, desc="参数扫描", position=0)

    # 扫描权重 w
    print("\n========== 扫描多目标权重 w ==========")
    for w_dict in param_configs['w']:
        w1, w2, w3 = w_dict['w1'], w_dict['w2'], w_dict['w3']
        print(f"训练 w=({w1:.2f},{w2:.2f},{w3:.2f})")
        cfg = Config(w1, w2, w3, gamma=0.99, beta=0.01, K=4)
        gat = HeteroGAT(dim_op, dim_mac, hidden, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                     gamma=cfg.gamma, beta=cfg.beta, lr=lr, state_norm=state_norm)

        for ep in range(1, NUM_EPOCHS+1):
            # 随机选择一个训练实例
            inst = random.choice(train_instances)
            agent.inst = inst
            env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg)
            state = env.reset()
            done = False
            traj = []
            while not done:
                action_tuple, probs, indices = agent.get_action(state, deterministic=False)
                next_state, r, done = env.step(action_tuple)
                traj.append({
                    'state': state,
                    'action': action_tuple,
                    'reward': r.item(),
                    'old_probs': probs,
                    'indices': indices
                })
                state = next_state
            if len(traj) > 0:
                agent.update(traj)
        # 评估
        F, C, L, T_val = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['w']['values'].append(f"({w1},{w2},{w3})")
        results['w']['F'].append(F)
        results['w']['Cmax'].append(C)
        results['w']['LB'].append(L)
        results['w']['Tardy'].append(T_val)
        results['w']['epochs'].append(NUM_EPOCHS)
        pbar_scan.update(1)

    # 扫描 beta
    print("\n========== 扫描 KL惩罚系数 beta ==========")
    for beta in param_configs['beta']:
        print(f"训练 beta={beta}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=0.99, beta=beta, K=4)
        gat = HeteroGAT(dim_op, dim_mac, hidden, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                     gamma=cfg.gamma, beta=cfg.beta, lr=lr, state_norm=state_norm)
        for ep in range(1, NUM_EPOCHS+1):
            inst = random.choice(train_instances)
            agent.inst = inst
            env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg)
            state = env.reset()
            done = False
            traj = []
            while not done:
                action_tuple, probs, indices = agent.get_action(state, deterministic=False)
                next_state, r, done = env.step(action_tuple)
                traj.append({
                    'state': state,
                    'action': action_tuple,
                    'reward': r.item(),
                    'old_probs': probs,
                    'indices': indices
                })
                state = next_state
            if len(traj) > 0:
                agent.update(traj)
        F, C, L, T_val = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['beta']['values'].append(beta)
        results['beta']['F'].append(F)
        results['beta']['Cmax'].append(C)
        results['beta']['LB'].append(L)
        results['beta']['Tardy'].append(T_val)
        results['beta']['epochs'].append(NUM_EPOCHS)
        pbar_scan.update(1)

    # 扫描 gamma
    print("\n========== 扫描折扣因子 gamma ==========")
    for gamma in param_configs['gamma']:
        print(f"训练 gamma={gamma}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=gamma, beta=0.01, K=4)
        gat = HeteroGAT(dim_op, dim_mac, hidden, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                     gamma=cfg.gamma, beta=cfg.beta, lr=lr, state_norm=state_norm)
        for ep in range(1, NUM_EPOCHS+1):
            inst = random.choice(train_instances)
            agent.inst = inst
            env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg)
            state = env.reset()
            done = False
            traj = []
            while not done:
                action_tuple, probs, indices = agent.get_action(state, deterministic=False)
                next_state, r, done = env.step(action_tuple)
                traj.append({
                    'state': state,
                    'action': action_tuple,
                    'reward': r.item(),
                    'old_probs': probs,
                    'indices': indices
                })
                state = next_state
            if len(traj) > 0:
                agent.update(traj)
        F, C, L, T_val = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['gamma']['values'].append(gamma)
        results['gamma']['F'].append(F)
        results['gamma']['Cmax'].append(C)
        results['gamma']['LB'].append(L)
        results['gamma']['Tardy'].append(T_val)
        results['gamma']['epochs'].append(NUM_EPOCHS)
        pbar_scan.update(1)

    # 扫描 K
    print("\n========== 扫描注意力头数 K ==========")
    for K in param_configs['K']:
        if out % K != 0:
            print(f"警告: out_dim={out} 不能被 K={K} 整除，跳过")
            pbar_scan.update(1)
            continue
        print(f"训练 K={K}")
        cfg = Config(W1_DEFAULT, W2_DEFAULT, W3_DEFAULT, gamma=0.99, beta=0.01, K=K)
        gat = HeteroGAT(dim_op, dim_mac, hidden, out, num_heads=cfg.K).to(DEVICE)
        upper = UpperPolicy(out, hidden).to(DEVICE)
        lower = LowerPolicy(out, hidden).to(DEVICE)
        critic_u = Critic(out, hidden).to(DEVICE)
        critic_l = Critic(out, hidden).to(DEVICE)
        state_norm = TorchRunningMeanStd(shape=(out,), device=DEVICE)
        agent = HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                     gamma=cfg.gamma, beta=cfg.beta, lr=lr, state_norm=state_norm)
        for ep in range(1, NUM_EPOCHS+1):
            inst = random.choice(train_instances)
            agent.inst = inst
            env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg)
            state = env.reset()
            done = False
            traj = []
            while not done:
                action_tuple, probs, indices = agent.get_action(state, deterministic=False)
                next_state, r, done = env.step(action_tuple)
                traj.append({
                    'state': state,
                    'action': action_tuple,
                    'reward': r.item(),
                    'old_probs': probs,
                    'indices': indices
                })
                state = next_state
            if len(traj) > 0:
                agent.update(traj)
        F, C, L, T_val = evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date, cfg, deterministic=True)
        results['K']['values'].append(K)
        results['K']['F'].append(F)
        results['K']['Cmax'].append(C)
        results['K']['LB'].append(L)
        results['K']['Tardy'].append(T_val)
        results['K']['epochs'].append(NUM_EPOCHS)
        pbar_scan.update(1)

    pbar_scan.close()

    # 绘制曲线
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    param_names = ['w', 'beta', 'gamma', 'K']
    titles = ['Multi-objective weights w', 'KL penalty coefficient β', 'Discount factor γ', 'Number of attention heads K']
    for idx, (param, ax) in enumerate(zip(param_names, axes.flatten())):
        values = results[param]['values']
        C_vals = results[param]['Cmax']
        L_vals = results[param]['LB']
        T_vals = results[param]['Tardy']
        ax.plot(values, C_vals, marker='o', label='Cmax', color='blue')
        ax.plot(values, L_vals, marker='s', label='LB', color='orange')
        ax.plot(values, T_vals, marker='^', label='Tardy', color='green')
        ax.set_xlabel(titles[idx])
        ax.set_ylabel('Objective value')
        ax.legend()
        ax.grid(True)
        if param == 'w':
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values, rotation=45)
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=150)
    plt.show()

    print("\n\n========== 参数敏感度检测结果汇总 ==========")
    for param in param_names:
        print(f"\n--- {param} ---")
        df = pd.DataFrame({
            'Value': results[param]['values'],
            'Avg F': results[param]['F'],
            'Avg Cmax': results[param]['Cmax'],
            'Avg LB': results[param]['LB'],
            'Avg Tardy': results[param]['Tardy'],
            'Epochs': results[param]['epochs']
        })
        print(df.to_string(index=False))
        df.to_csv(f'sensitivity_{param}.csv', index=False)

    print("\n图形已保存为 parameter_sensitivity.png，表格已保存为 CSV 文件。")

if __name__ == "__main__":
    parameter_sensitivity_scan()