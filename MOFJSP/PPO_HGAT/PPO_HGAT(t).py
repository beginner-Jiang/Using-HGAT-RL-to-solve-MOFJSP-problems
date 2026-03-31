"""
PPO_HGAT Test Program
Loads the trained best model, runs scheduling on test set instances, and computes test metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import glob
import time
import warnings
import argparse
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10

# Objective function weights (consistent with training)
W1 = 0.4
W2 = 0.3
W3 = 0.3

# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Default paths
DEFAULT_INSTANCE_DIR = "../mo_fjsp_instances"
DEFAULT_TEST_PATTERN = "mo_fjsp_*_test.txt"
DEFAULT_MODEL_DIR = "../model/ppo_hgat"
DEFAULT_MODEL_NAME = "ppo_hgat_best.pth"
DEFAULT_REWARD_SCALING = 50.0
DEFAULT_REWARD_CLIP = 10.0


# Data reading and instance definition (consistent with training)
def parse_size_from_filename(filename: str) -> str:
    match = re.search(r'(small|medium|large)', filename)
    return match.group(1) if match else "unknown"


@dataclass
class Operation:
    job_id: int
    op_id: int
    machine_times: Dict[int, float]


def read_fjsp_instance(file_path: str):
    """Reads an FJSP instance file, returns job list, machine capabilities list, due dates list, filename, and size"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
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
    """Loads all matching instances and returns the instance list and global maximums (including machine capability)"""
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
        if caps:
            max_capability = max(max_capability, max(caps))

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

        for j_idx, job in enumerate(jobs):
            for op in job:
                key = (j_idx, op.op_id)
                self.available_machines[key] = []
                for m, t in op.machine_times.items():
                    self.processing_times[(j_idx, op.op_id, m)] = t
                    self.available_machines[key].append(m)

        all_times = list(self.processing_times.values())
        self.max_proc_time = max(all_times) if all_times else 1.0
        self.mean_proc_time = np.mean(all_times) if all_times else 1.0

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

        # Capability feature normalization
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

    def get_pt(self, job, op, machine):
        return self.proc_time_matrix[self.op_index_map[(job, op)], machine].item()


# Environment class (consistent with training)
class HeteroGraphEnv:
    def __init__(self, instance: MOFJSPInstance, global_max_jobs, global_max_ops, global_max_machines,
                 global_max_proc_time, global_max_due_date, global_max_capability, cfg):
        self.inst = instance
        self.global_max_jobs = global_max_jobs
        self.global_max_ops = global_max_ops
        self.global_max_machines = global_max_machines
        self.global_max_proc_time = global_max_proc_time
        self.global_max_due_date = global_max_due_date
        self.global_max_capability = global_max_capability
        self.cfg = cfg

        self.global_max_proc_time_tensor = torch.tensor(global_max_proc_time, device=DEVICE)
        self.global_max_due_date_tensor = torch.tensor(global_max_due_date, device=DEVICE)
        self.global_max_capability_tensor = torch.tensor(global_max_capability, device=DEVICE)

        self.total_op_nodes = instance.total_ops
        self.total_machine_nodes = instance.n_machines

        self.op_feat_dim = 3
        self.mac_feat_dim = 3  # Consistent with training: load, available time, capability

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

        # Static machine features: capability (third dimension)
        self.static_mac_feats = torch.zeros((self.global_max_machines, self.mac_feat_dim), dtype=torch.float32, device=DEVICE)
        for m in range(instance.n_machines):
            self.static_mac_feats[m, 2] = instance.capability_tensor[m]

        self.reset()

    def reset(self):
        self.scheduled_ops = []  # Optional, for Gantt charts
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

        # Record schedule (optional)
        self.scheduled_ops.append((job.item(), op.item(), machine.item(), start.item(), end.item()))

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

        reward = - (W1 * norm_p + W2 * delta_LB / self.global_max_proc_time_tensor + W3 * u_ij)
        reward /= self.cfg.reward_scaling
        reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        done = self.op_scheduled.all().item()
        if done:
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
        # Third dimension (capability) remains static

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


# Network definitions (consistent with training)
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


# PPO Agent
class HierarchicalPPOAgent:
    def __init__(self, inst, gat, upper, lower, critic_u, critic_l, state_norm=None, gamma=0.99, lam=0.95, clip=0.2, beta=0.01, lr=3e-4):
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
            # Force conversion to float32
            self.gat = self.gat.float()
            self.upper = self.upper.float()
            self.lower = self.lower.float()
            self.critic_u = self.critic_u.float()
            self.critic_l = self.critic_l.float()
            # Recreate optimizer (optional, not needed for inference)
            self.params = list(self.gat.parameters()) + list(self.upper.parameters()) + \
                          list(self.lower.parameters()) + list(self.critic_u.parameters()) + list(self.critic_l.parameters())
            self.opt = torch.optim.Adam(self.params, lr=self.lr)
            self.gamma = checkpoint.get('gamma', 0.99)
            self.beta = checkpoint.get('beta', 0.01)
            print(f"Model loaded from {path} and converted to float32, gamma={self.gamma:.3f}, beta={self.beta:.3f}")
            return True
        else:
            print(f"Model file {path} not found")
            return False

    def get_action(self, state, deterministic=True):
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

            # Machine mask
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

        return (job, op, machine)


# Test helper functions
def compute_final_objectives(env):
    Cmax = max(env.job_completion_time).item()
    LB = torch.std(env.machine_load).item() if env.machine_load.numel() > 0 else 0.0
    tardy_sum = torch.sum(torch.clamp(env.job_completion_time - env.due_dates_tensor, min=0)).item()
    F = W1 * Cmax + W2 * LB + W3 * tardy_sum
    return Cmax, LB, tardy_sum, F


def run_episode(agent, env, deterministic=True):
    state = env.reset()
    done = False
    start_time = time.time()
    while not done:
        action = agent.get_action(state, deterministic=deterministic)
        next_state, _, done = env.step(action)
        state = next_state
    end_time = time.time()
    elapsed = end_time - start_time
    Cmax, LB, tardy, F = compute_final_objectives(env)
    return {'Cmax': Cmax, 'LB': LB, 'tardy': tardy, 'F': F, 'time': elapsed}


# Main test function (supports command line arguments)
def main():
    parser = argparse.ArgumentParser(description="PPO_HGAT Model Test Script")
    parser.add_argument('--instance_dir', type=str, default=DEFAULT_INSTANCE_DIR,
                        help=f'Test instance directory (default: {DEFAULT_INSTANCE_DIR})')
    parser.add_argument('--test_pattern', type=str, default=DEFAULT_TEST_PATTERN,
                        help=f'Test file matching pattern (default: "{DEFAULT_TEST_PATTERN}")')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR,
                        help=f'Model directory (default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help=f'Model file name (default: {DEFAULT_MODEL_NAME})')
    parser.add_argument('--reward_scaling', type=float, default=DEFAULT_REWARD_SCALING,
                        help=f'Reward scaling factor (default: {DEFAULT_REWARD_SCALING})')
    parser.add_argument('--reward_clip', type=float, default=DEFAULT_REWARD_CLIP,
                        help=f'Reward clipping threshold (default: {DEFAULT_REWARD_CLIP})')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU usage')
    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = DEVICE

    print("=" * 50)
    print("PPO_HGAT Model Test")
    print("=" * 50)
    print(f"Using device: {device}")
    print(f"Instance directory: {args.instance_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Model file: {args.model_name}")

    # Check if paths exist
    if not os.path.isdir(args.instance_dir):
        print(f"Error: Instance directory does not exist - {args.instance_dir}")
        sys.exit(1)
    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory does not exist - {args.model_dir}")
        sys.exit(1)
    model_path = os.path.join(args.model_dir, args.model_name)
    if not os.path.isfile(model_path):
        print(f"Error: Model file does not exist - {model_path}")
        sys.exit(1)

    # Load test instances and get global maximums (including machine capability)
    try:
        all_instances, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = \
            load_all_instances(args.instance_dir, args.test_pattern)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"\nGlobal max jobs: {max_jobs}")
    print(f"Global max machines: {max_machines}")
    print(f"Global max total operations: {max_total_ops}")
    print(f"Global max processing time: {max_proc_time}")
    print(f"Global max due date: {max_due_date}")
    print(f"Global max machine capability: {max_capability}")

    # Build instance objects
    test_instances = []
    for jobs, caps, dues, fname, size in all_instances:
        inst = MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, max_capability)
        test_instances.append(inst)

    print(f"\nLoaded {len(test_instances)} test instances:")
    for inst in test_instances:
        print(f"  {inst.file_name}")

    # Match training model
    dim_op = 3
    dim_mac = 3
    gat_hidden = 256
    gat_out = 256
    policy_hidden = 512

    gat = HeteroGAT(dim_op, dim_mac, gat_hidden, gat_out).to(device)
    upper = UpperPolicy(gat_out, policy_hidden).to(device)
    lower = LowerPolicy(gat_out, policy_hidden).to(device)
    critic_u = Critic(gat_out, policy_hidden).to(device)
    critic_l = Critic(gat_out, policy_hidden).to(device)

    # Load normalization statistics if they exist
    norm_path = os.path.join(args.model_dir, "state_norm.pt")
    if os.path.exists(norm_path):
        state_norm = TorchRunningMeanStd(shape=(gat_out,), device=device)
        norm_state = torch.load(norm_path, map_location=device)
        state_norm.mean = norm_state['mean'].to(device)
        state_norm.var = norm_state['var'].to(device)
        state_norm.count = norm_state['count']
        print("State normalization parameters loaded.")
    else:
        state_norm = None
        print("State normalization file not found, skipping normalization.")

    # Create Agent (for inference only)
    dummy_inst = test_instances[0]  # Temporary instance, will be replaced after loading
    agent = HierarchicalPPOAgent(dummy_inst, gat, upper, lower, critic_u, critic_l, state_norm=state_norm)

    if not agent.load(model_path):
        sys.exit(1)

    # Create configuration object (for environment)
    class DummyConfig:
        def __init__(self, scaling, clip):
            self.reward_scaling = scaling
            self.reward_clip = clip
            self.use_disjunctive_edges = False

    cfg = DummyConfig(args.reward_scaling, args.reward_clip)

    # Run tests
    results = []
    for idx, inst in enumerate(test_instances):
        print(f"\nTesting instance {idx+1}/{len(test_instances)}: {inst.file_name}")
        env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                             max_proc_time, max_due_date, max_capability, cfg)
        agent.inst = inst  # Update current instance
        res = run_episode(agent, env, deterministic=True)
        results.append({
            'file': inst.file_name,
            'Cmax': res['Cmax'],
            'LB': res['LB'],
            'tardy': res['tardy'],
            'F': res['F'],
            'time': res['time']
        })
        print(f"  Cmax={res['Cmax']:.2f}, LB={res['LB']:.4f}, Tardy={res['tardy']:.2f}, F={res['F']:.4f}, Time={res['time']:.4f}s")

    # Statistical results
    F_values = [r['F'] for r in results]
    F_min = min(F_values)
    F_max = max(F_values)
    F_mean = np.mean(F_values)
    F_std = np.std(F_values, ddof=1)
    time_mean = np.mean([r['time'] for r in results])

    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    print(f"Number of test instances: {len(results)}")
    print(f"Objective function value range: [{F_min:.4f}, {F_max:.4f}]")
    print(f"Average objective function value F_mean: {F_mean:.4f}")
    print(f"Objective function value standard deviation σ_F: {F_std:.4f}")
    print(f"Average solving time t_mean: {time_mean:.4f} seconds")

    print("\nDetailed results (per instance):")
    print(f"{'File':<25} {'Cmax':<10} {'LB':<10} {'Tardy':<10} {'F':<10} {'Time(s)':<10}")
    for r in results:
        fname = r['file'][:23] + '...' if len(r['file']) > 23 else r['file']
        print(f"{fname:<25} {r['Cmax']:<10.2f} {r['LB']:<10.4f} {r['tardy']:<10.2f} {r['F']:<10.4f} {r['time']:<10.4f}")


if __name__ == "__main__":
    main()