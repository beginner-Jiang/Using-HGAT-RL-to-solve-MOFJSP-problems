"""
HGNN: Flexible Job-shop Scheduling via Heterogeneous Graph Neural Network and Deep Reinforcement Learning (baseline implementation)

Reference: W. Song, X. Chen, Q. Li and Z. Cao, "Flexible Job-shop Scheduling via Graph Neural
Network and Deep Reinforcement Learning," IEEE Transactions on Industrial Informatics,
vol. 19, no. 2, pp. 1600-1610, 2023.

Fair-comparison notes:
  * the instance reader, the scheduling environment (``HeteroGraphEnv``) and the multi-objective
    reward are reused verbatim from the proposed HGHRL method (``HGHRL.HGHRL``), so all
    methods share the same state transition, the same evaluation protocol and the same objective
    weights;
  * the hierarchical policy of the proposed method is replaced by a **single-level policy** that
    samples an (operation, machine) pair directly from the feasible joint action space (the
    single-level control experiment requested by the reviewers); the two layer-specific critics of
    the proposed method are not used either;
  * only the encoder and the decision structure change, every other training hyper-parameter is
    kept identical, so the comparison stays controlled.
"""

import os
import sys
import math
import time
import random
import warnings
import argparse
from typing import List, Dict, Tuple

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
# Without a display session (background run, stdout redirected to a file, headless server)
# the Qt backend cannot be initialised and the process would crash; fall back to the file-only
# Agg backend and keep the default backend for interactive runs.
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Intra-op thread count: every training run is launched with a slice of the CPU
# budget, and the torch default (one thread per physical core) oversubscribes the
# machine while several runs share it.
THREAD_BUDGET = int(os.environ.get('TORCH_NUM_THREADS', '0') or 0)
if THREAD_BUDGET > 0:
    torch.set_num_threads(THREAD_BUDGET)

# Add the project root to the module search path so that this file can be imported by the
# experiment scripts and also executed directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from HGHRL.HGHRL import (
    read_fjsp_instance,
    load_all_instances,
    MOFJSPInstance,
    HeteroGraphEnv,
    CurriculumSampler,
    orthogonal_init,
    batch_states,
    cfg as HGATConfig,
)

# This baseline keeps the node features of its proposing paper (three operation columns
# and three machine columns).  The shared environment defaults to the proposed method's
# Appendix-B table, so the wrapper pins the original baseline specification.
_BaseHeteroGraphEnv = HeteroGraphEnv


def HeteroGraphEnv(*args, **kwargs):
    kwargs.setdefault("feature_mode", "legacy")
    return _BaseHeteroGraphEnv(*args, **kwargs)

# 1. Global configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

# Multi-objective weights (kept identical to the proposed method)
W1 = 0.4  # Cmax
W2 = 0.3  # Load Balance
W3 = 0.3  # Tardy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Training instance directory (project root / mo_fjsp_instances)
INSTANCE_DIR = os.path.join(_PROJECT_ROOT, "DataSet", "mo_fjsp_instances")
FILE_PATTERN = "mo_fjsp_*_train.txt"


# 2. Configuration class

class Config:
    n_episodes = 10000
    # Optimiser settings of the proposed method: a controlled comparison requires the learned
    # baselines to train with the same learning rate, schedule and gradient clip.  The values
    lr = 5e-4
    lr_decay = 0.5
    lr_decay_steps = 2000
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    # Tightened from 0.2.  The ratio clip is the mechanism that bounds how far one update may
    # move the policy, and the advantage of this update is estimated from a single episode (one
    clip_ratio = 0.1
    value_coef = 0.5
    entropy_coef = 0.01
    entropy_decay = 0.9995
    # KL penalty of the proposed method: the coefficient is adapted to the measured divergence so
    # that one update never moves the policy far away from the rollouts it was collected with.
    kl_target = 0.1
    beta_start = 0.2
    beta_lr = 0.05
    # Restored to the family norm of ten.  The value was loosened to fifty and the GAT baseline
    # of the same family diverged under it: its total loss rose to 1.1e6 because its value
    max_grad_norm = 10.0
    update_epochs = 1
    # Number of instances that are collected in parallel.  The encoder, the joint policy and the
    # value network then process several states per forward pass, which amortises the kernel
    n_envs = 12
    chunk_size = 32
    logits_clip = 50.0
    max_steps_per_episode = 2000

    # Rolling checkpoint that an interrupted run continues from: the weights
    # and the state of the optimiser are written every checkpoint_interval
    # episodes.
    resume_name = 'hgnn_last.pth'
    checkpoint_interval = 25
    log_interval = 100

    # Convergence control.  One episode is a noisy sample of the policy (its instance scale and
    # its number of operations change from episode to episode), so the weights are selected on the
    plateau_window = 100
    # The reported weights are selected on a fixed validation set instead of on the smoothed
    # training return: the curriculum moves the instance distribution over the run, so the
    val_interval = 250
    # Four instances per size halve the noise of the six instance set that the previous
    # generation selected its weights on: the best checkpoint and the early stop of a run
    val_instances_per_size = 8
    val_seed = 0
    # A stale check is one validation of 250 episodes, so a patience of six holds a run
    # 1500 episodes past its peak (the generation before it used eight, i.e. 2000).
    val_patience = 6
    val_tol = 0.01
    # No early stop before this budget, and the window has to open before the peak of a method:
    # the best checkpoint of the previous generation was written at episode 1250 for the DAN
    min_episodes = 500
    # The curriculum advances on a fixed episode scale instead of on the --episodes budget, so
    # that the schedule does not change when the budget of the run is changed.
    curriculum_episodes = 4000
    # Floor of the entropy coefficient.  The decay alone drives the coefficient to 1e-4 over a
    # run of ten thousand episodes, i.e. the policy becomes deterministic before the curriculum
    entropy_min = 0.02

    # Encoder
    dim_op = 3   # original baseline features: position, due date, job id
    dim_mac = 3
    hidden_dim = 256
    # Depth raised from two.  The head of this baseline is a pointer score of the operation and
    # machine embeddings, its action set is the ready operations of the shared environment, and
    enc_layers = 2
    policy_hidden_dim = 512

    # Model output
    model_name = "hgnn_best.pth"
    model_save_dir = os.path.join(_PROJECT_ROOT, "model", "hgnn")
    figure_dir = os.path.join(_PROJECT_ROOT, "Figure_And_File", "HGNN")


cfg = Config()


# 3. Joint action utilities
# The single-level policy acts on (operation, machine) pairs; the helpers below build the
# joint action mask and convert between flat action indices and environment actions.

def build_joint_mask(inst, state, n_ops, n_machines):
    """
    Build the (operation, machine) joint action mask: an entry is 1 only if the operation is
    currently schedulable and the machine is able to process it.
    """
    joint_mask = torch.zeros((n_ops, n_machines), dtype=torch.float32, device=DEVICE)
    joint_mask[:inst.total_ops, :inst.n_machines] = inst.op_mac_mask
    joint_mask = joint_mask * (state['op_mask'] > 0.5).float().unsqueeze(1)
    return joint_mask


def build_joint_mask_batch(insts, states, n_ops, n_machines):
    """Feasible (operation, machine) mask of every state of a batch.

    ``insts[i]`` belongs to ``states[i]``; the update passes the same instance for every state of
    a chunk.  The result is the batched form of :func:`build_joint_mask`, i.e. a
    ``[batch, n_ops, n_machines]`` tensor that combines the capability matrix of the instance with
    the operation each job releases next.
    """
    op_mask = torch.stack([state['op_mask'] for state in states], dim=0).float()
    bases = []
    cached_inst, cached_base = None, None
    for inst in insts:
        if inst is not cached_inst:
            cached_base = torch.zeros((n_ops, n_machines), dtype=torch.float32, device=DEVICE)
            cached_base[:inst.total_ops, :inst.n_machines] = inst.op_mac_mask
            cached_inst = inst
        bases.append(cached_base)
    base = torch.stack(bases, dim=0)
    return base * (op_mask > 0.5).unsqueeze(-1).float()


def joint_index_to_env_action(index, n_machines, inst):
    """Convert a flat joint action index into the (job, op, machine) tuple used by the environment."""
    op_index = torch.div(index, n_machines, rounding_mode='floor')
    machine = index - op_index * n_machines
    op_index = torch.clamp(op_index, 0, inst.total_ops - 1)
    job = inst.op_idx_to_job[op_index]
    op = inst.op_idx_to_op[op_index]
    return (job, op, machine)




# 4. HGNN encoder
# Heterogeneous mean-aggregation encoder: every edge type owns a separate linear transform, the

class HGNNLayer(nn.Module):
    """One heterogeneous mean-aggregation layer."""

    def __init__(self, in_dim_op, in_dim_mac, out_dim, edge_dim=1):
        super().__init__()
        gain = nn.init.calculate_gain('relu')
        self.op_self = orthogonal_init(nn.Linear(in_dim_op, out_dim), gain=gain)
        self.mac_self = orthogonal_init(nn.Linear(in_dim_mac, out_dim), gain=gain)
        self.op_from_op = orthogonal_init(nn.Linear(in_dim_op + edge_dim, out_dim), gain=gain)
        self.op_from_mac = orthogonal_init(nn.Linear(in_dim_mac + edge_dim, out_dim), gain=gain)
        self.mac_from_op = orthogonal_init(nn.Linear(in_dim_op + edge_dim, out_dim), gain=gain)
        # Competing operations of the manuscript.  The shared environment reports the
        # disjunctive pairs next to the conjunctive edges and this encoder carried no
        self.op_from_disj = orthogonal_init(nn.Linear(in_dim_op + edge_dim, out_dim), gain=gain)

    @staticmethod
    def _mean_aggregate(size, index, message):
        """Average the messages per destination node given by ``index``."""
        if index.numel() == 0:
            return torch.zeros((size, message.shape[-1]), dtype=message.dtype, device=message.device)
        output = torch.zeros((size, message.shape[-1]), dtype=message.dtype, device=message.device)
        output.index_add_(0, index, message)
        counts = torch.zeros(size, dtype=message.dtype, device=message.device)
        counts.index_add_(0, index, torch.ones_like(index, dtype=message.dtype))
        return output / counts.clamp(min=1.0).unsqueeze(-1)

    def _messages(self, src_idx, dst_idx, src_emb, edge_feat, linear, size):
        if src_idx.numel() == 0:
            return torch.zeros((size, linear.out_features), dtype=src_emb.dtype, device=src_emb.device)
        feat = edge_feat.to(src_emb.dtype).reshape(-1, 1)
        message = linear(torch.cat([src_emb[src_idx], feat], dim=-1))
        return self._mean_aggregate(size, dst_idx, message)

    def forward(self, h_op, h_mac, edges, edge_feats):
        h_op = h_op.float()
        h_mac = h_mac.float()
        n_ops, n_macs = h_op.shape[0], h_mac.shape[0]

        agg_op = self._messages(edges['seq'][0], edges['seq'][1], h_op, edge_feats['seq'],
                                self.op_from_op, n_ops)
        agg_op = agg_op + self._messages(edges['mac_op'][0], edges['mac_op'][1], h_mac,
                                         edge_feats['mac_op'], self.op_from_mac, n_ops)
        # The disjunctive channel carries the competition between the operations that share a
        # machine, which is the information a load decision needs; it is part of the state the
        # manuscript describes and was unused before this wave.
        agg_op = agg_op + self._messages(edges['disj'][0], edges['disj'][1], h_op,
                                         edge_feats['disj'], self.op_from_disj, n_ops)
        agg_mac = self._messages(edges['op_mac'][0], edges['op_mac'][1], h_op, edge_feats['op_mac'],
                                 self.mac_from_op, n_macs)

        new_op = F.elu(self.op_self(h_op) + agg_op)
        new_mac = F.elu(self.mac_self(h_mac) + agg_mac)
        return new_op, new_mac


class HGNNEncoder(nn.Module):
    """Multi-layer heterogeneous mean-aggregation encoder."""

    def __init__(self, dim_op, dim_mac, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim_op, in_dim_mac = dim_op, dim_mac
        for _ in range(num_layers):
            self.layers.append(HGNNLayer(in_dim_op, in_dim_mac, hidden_dim))
            in_dim_op = in_dim_mac = hidden_dim
        self.out_dim = hidden_dim

    def forward(self, state_dict):
        h_op = state_dict['op_feats']
        h_mac = state_dict['mac_feats']
        edges = state_dict['edges']
        edge_feats = state_dict['edge_feats']
        for layer in self.layers:
            h_op, h_mac = layer(h_op, h_mac, edges, edge_feats)
        return h_op, h_mac


# 5. Single-level joint policy and value network

class JointPolicy(nn.Module):
    """
    Single-level policy network: emits a probability distribution directly over the
    (operation, machine) joint action space.

    The score is bilinear, <q(h_op_i), k(h_mac_m)>, plus an operation-dependent and a
    machine-dependent bias term. Both bias terms consume the embedding of the node and the
    global graph embedding, so the global context still takes part in the decision.
    """

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
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
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, g_emb, h_op, h_mac, joint_mask):
        n_ops, n_machines = joint_mask.shape
        g_emb = g_emb.float()

        query = self.op_query(h_op).float() * self.scale
        key = self.mac_key(h_mac).float()
        logits = query @ key.t()

        g_op = g_emb.unsqueeze(0).expand(n_ops, -1)
        g_mac = g_emb.unsqueeze(0).expand(n_machines, -1)
        logits = logits + self.op_bias(torch.cat([g_op, h_op], dim=-1))
        logits = logits + self.mac_bias(torch.cat([g_mac, h_mac], dim=-1)).t()

        logits = logits.reshape(-1)
        mask = joint_mask.reshape(-1)
        feasible = mask > 0.5
        fill_value = -1e9
        # Shift the feasible logits by their maximum before clipping: softmax is shift
        # invariant, so the relative differences between the actions survive, whereas the
        if feasible.any():
            logits = logits - logits[feasible].max()
        logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
        logits = logits.masked_fill(~feasible, fill_value)

        if (logits == fill_value).all():
            probs = torch.full_like(logits, 1.0 / logits.numel())
        else:
            probs = F.softmax(logits, dim=0)
            # The probability of an infeasible action is exactly 0; renormalise on the feasible ones
            probs = probs * feasible.float()
        return probs / probs.sum().clamp(min=EPS)

    def forward_batch(self, g_emb, h_op, h_mac, joint_mask):
        """
        Batched :meth:`forward`: one probability vector per state of the batch.

        ``h_op`` and ``h_mac`` hold one block per state, and the flattened action index of a state
        is ``operation * n_machines + machine``, exactly as in the single-state version, so a
        sampled index keeps the same meaning.  The heavy calls (both bias networks and the
        bilinear score) run once per batch instead of once per state.
        """
        batch, n_ops, _ = h_op.shape
        n_machines = h_mac.shape[1]
        g_emb = g_emb.float()

        query = self.op_query(h_op).float() * self.scale
        key = self.mac_key(h_mac).float()
        logits = torch.bmm(query, key.transpose(1, 2))

        g_op = g_emb.unsqueeze(1).expand(-1, n_ops, -1)
        g_mac = g_emb.unsqueeze(1).expand(-1, n_machines, -1)
        logits = logits + self.op_bias(torch.cat([g_op, h_op], dim=-1))
        logits = logits + self.mac_bias(torch.cat([g_mac, h_mac], dim=-1)).transpose(1, 2)

        logits = logits.reshape(batch, -1)
        feasible = joint_mask.reshape(batch, -1) > 0.5
        fill_value = -1e9
        # Shift and clipping follow the single-state branch: see the comment there.
        has_feasible = feasible.any(dim=1, keepdim=True)
        row_max = logits.masked_fill(~feasible, fill_value).max(dim=1, keepdim=True).values
        # A state without a feasible action keeps its unshifted logits and receives the uniform
        # distribution below, which is what the single-state branch returns as well.
        logits = logits - torch.where(has_feasible, row_max, torch.zeros_like(row_max))
        logits = torch.clamp(logits, min=-cfg.logits_clip, max=cfg.logits_clip)
        logits = logits.masked_fill(~feasible, fill_value)

        probs = F.softmax(logits, dim=1)
        probs = probs * feasible.to(probs.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=EPS)
        uniform = torch.full_like(probs, 1.0 / probs.shape[1])
        return torch.where(has_feasible, probs, uniform)


# 6. Single-level PPO agent

class SingleLevelPPOAgent:
    """PPO (clipped surrogate objective) single-level agent that trains the encoder,
    the policy and the value network jointly."""

    def __init__(self, encoder, policy, value, cfg):
        self.encoder = encoder
        self.policy = policy
        self.value = value
        self.cfg = cfg
        self.entropy_coef = cfg.entropy_coef
        self.beta = cfg.beta_start
        self.beta_lr = cfg.beta_lr
        self.kl_ema = None
        # Training curve of this run.  It is written into the rolling checkpoint so that a
        # restart continues the curve instead of losing the episodes already trained.
        self.kl_alpha = 0.9
        self.curve_returns = []
        self.curve_norm_returns = []
        # Validation bookkeeping: the validation curve, the reference objective of every
        # validation instance and the best value seen so far survive an interruption.
        self.val_curve = []
        self.val_reference = []
        self.best_val = float('inf')
        self.converged = False
        self.params = (list(encoder.parameters()) + list(policy.parameters())
                       + list(value.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay)

    def save(self, path):
        '''Write the weights together with everything a restart needs.

        The optimiser moments, the scheduler state, the entropy coefficient,
        the episode counter and the best normalised return are stored as well,
        so that an interrupted run is continued instead of being restarted.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_path = path + ".tmp"
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'entropy_coef': self.entropy_coef,
            'beta': float(self.beta),
            'episode_counter': getattr(self, 'episode_counter', 0),
            'best_reward': getattr(self, 'best_reward', -float('inf')),
            'curve_returns': list(self.curve_returns),
            'curve_norm_returns': list(self.curve_norm_returns),
            'val_curve': [tuple(entry) for entry in self.val_curve],
            'val_reference': list(self.val_reference),
            'best_val': self.best_val,
            'converged': self.converged,
        }, _atomic_path)
        os.replace(_atomic_path, path)
        print(f"Model saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"Model file {path} not found, starting from scratch.")
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(path, map_location=DEVICE)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.encoder = self.encoder.to(DEVICE).float()
        self.policy = self.policy.to(DEVICE).float()
        self.value = self.value.to(DEVICE).float()
        self.params = (list(self.encoder.parameters()) + list(self.policy.parameters())
                       + list(self.value.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=self.cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay)
        # Checkpoints written by save() carry the optimiser moments and the
        # bookkeeping of the interrupted run; the older checkpoints hold the
        # weights only and are accepted with a cold optimiser.
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
        self.episode_counter = int(checkpoint.get('episode_counter', 0) or 0)
        self.best_reward = float(checkpoint.get('best_reward', -float('inf')))
        self.entropy_coef = float(checkpoint.get('entropy_coef', self.entropy_coef))
        self.beta = float(checkpoint.get('beta', self.beta))
        self.curve_returns = list(checkpoint.get('curve_returns', []) or [])
        self.curve_norm_returns = list(checkpoint.get('curve_norm_returns', []) or [])
        self.val_curve = [tuple(entry) for entry in (checkpoint.get('val_curve', []) or [])]
        self.val_reference = list(checkpoint.get('val_reference', []) or [])
        self.best_val = float(checkpoint.get('best_val', float('inf')))
        self.converged = bool(checkpoint.get('converged', False))
        print(f"Model loaded from {path}")
        return True

    def act(self, state, inst, n_ops, n_machines, deterministic=False):
        """Sample one (operation, machine) joint action for the current state."""
        with torch.no_grad():
            # The state goes through the same packer as a batch of states, because the packer is
            # what turns the static edge arrays of a state into the visible edge set the encoder
            # consumes.  A batch of one produces exactly the graph the single state describes.
            batch_state_dict, _, _ = batch_states([state], n_ops, n_machines)
            h_op, h_mac = self._encode_batch(batch_state_dict, n_ops, n_machines)
            g_emb = h_op.mean(dim=0)
            joint_mask = build_joint_mask(inst, state, n_ops, n_machines)
            probs = self.policy(g_emb, h_op, h_mac, joint_mask)
            dist = torch.distributions.Categorical(probs)
            index = torch.argmax(probs) if deterministic else dist.sample()
            log_prob = dist.log_prob(index)
            value = self.value(g_emb).squeeze()
        return index, log_prob.detach(), value.detach()

    def act_batch(self, states, insts, n_ops, n_machines, deterministic=False):
        """
        Sample one (operation, machine) joint action for every state of ``states``.

        ``insts[i]`` is the instance of ``states[i]``.  The states are concatenated into a single
        graph batch, so the encoder, the joint policy and the value network run once for the whole
        batch.  The actions, the log probabilities and the value estimates are returned as one
        tensor each, in the order of ``states``.
        """
        with torch.no_grad():
            batch_state_dict, _, _ = batch_states(states, n_ops, n_machines)
            h_op_all, h_mac_all = self._encode_batch(batch_state_dict, n_ops, n_machines)
            h_op = h_op_all.reshape(len(states), n_ops, -1)
            h_mac = h_mac_all.reshape(len(states), n_machines, -1)
            g_emb = h_op.mean(dim=1)
            joint_mask = build_joint_mask_batch(insts, states, n_ops, n_machines)
            probs = self.policy.forward_batch(g_emb, h_op, h_mac, joint_mask)
            dist = torch.distributions.Categorical(probs)
            indices = torch.argmax(probs, dim=1) if deterministic else dist.sample()
            log_probs = dist.log_prob(indices)
            values = self.value(g_emb).squeeze(-1)
        return indices, log_probs.detach(), values.detach()

    def _encode_batch(self, batch_state_dict, n_ops, n_machines):
        """Run the encoder on a graph batch and return the node embeddings of the batch."""
        return self.encoder(batch_state_dict)

    def update(self, trajectory, inst):
        """Update the networks from one full episode trajectory (GAE + clipped PPO objective)."""
        T = len(trajectory)
        if T == 0:
            return {}
        states = [t['state'] for t in trajectory]
        n_ops = states[0]['op_feats'].shape[0]
        n_machines = states[0]['mac_feats'].shape[0]

        # The rewards stay on the device: converting every reward to a python float inside the
        # collection loop would stop the host at every step.
        rewards = torch.stack([t['reward'] for t in trajectory]).to(DEVICE).float()
        actions = torch.stack([t['action'] for t in trajectory]).detach()
        old_log_probs = torch.stack([t['log_prob'] for t in trajectory]).detach()

        # Generalized advantage estimation
        with torch.no_grad():
            values = torch.stack([t['value'] for t in trajectory]).detach()
            advantages = torch.zeros(T, dtype=torch.float32, device=DEVICE)
            last_advantage = torch.zeros((), dtype=torch.float32, device=DEVICE)
            for t in reversed(range(T)):
                next_value = values[t + 1] if t + 1 < T else torch.zeros((), device=DEVICE)
                non_terminal = 0.0 if trajectory[t]['done'] else 1.0
                delta = rewards[t] + self.cfg.gamma * next_value * non_terminal - values[t]
                last_advantage = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * last_advantage
                advantages[t] = last_advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            # Tighter clip of the normalised advantage.  The advantage of this update is
            # estimated from a single episode (one environment per update, see the update call
            advantages = advantages.clamp(-3.0, 3.0)

        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'kl': 0.0, 'updates': 0}

        for _ in range(self.cfg.update_epochs):
            for start in range(0, T, self.cfg.chunk_size):
                end = min(start + self.cfg.chunk_size, T)
                chunk_states = states[start:end]

                batch_state_dict, _, _ = batch_states(chunk_states, n_ops, n_machines)
                h_op_all, h_mac_all = self._encode_batch(batch_state_dict, n_ops, n_machines)

                # Every state of a batch occupies one block of n_ops operation rows and n_machines
                # machine rows, so the embeddings can be reshaped into ``[batch, ...]`` instead of
                batch_size = len(chunk_states)
                h_op = h_op_all.reshape(batch_size, n_ops, -1)
                h_mac = h_mac_all.reshape(batch_size, n_machines, -1)
                g_emb = h_op.mean(dim=1)
                joint_mask = build_joint_mask_batch([inst] * batch_size, chunk_states,
                                                    n_ops, n_machines)
                probs = self.policy.forward_batch(g_emb, h_op, h_mac, joint_mask)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions[start:end])
                new_values = self.value(g_emb).squeeze(-1)
                entropy = dist.entropy().mean()

                advantage = advantages[start:end]
                ratio = torch.exp(new_log_probs - old_log_probs[start:end])
                surrogate = torch.min(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantage)
                policy_loss = -surrogate.mean()

                # Approximate KL divergence between the sampling policy and the updated one; the
                # total loss below weights this term with beta.
                old_probs = torch.exp(old_log_probs[start:end]).clamp(EPS, 1.0 - EPS)
                curr_probs = torch.exp(new_log_probs).clamp(EPS, 1.0 - EPS)
                kl = (old_probs * (torch.log(old_probs + EPS) - torch.log(curr_probs + EPS))).mean()
                kl = torch.clamp(kl, min=0.0, max=10.0)
                value_loss = F.mse_loss(new_values, returns[start:end])
                loss = (policy_loss + self.cfg.value_coef * value_loss
                        + self.beta * kl - self.entropy_coef * entropy)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm)
                self.opt.step()

                stats['policy_loss'] += float(policy_loss.detach())
                stats['value_loss'] += float(value_loss.detach())
                stats['entropy'] += float(entropy.detach())
                stats['kl'] += float(kl.detach())
                stats['updates'] += 1

        self.scheduler.step()

        # Adapt the KL coefficient to the measured divergence, exactly as the proposed method does
        avg_kl = stats['kl'] / max(1, stats['updates'])
        if self.kl_ema is None:
            self.kl_ema = avg_kl
        else:
            self.kl_ema = self.kl_alpha * self.kl_ema + (1.0 - self.kl_alpha) * avg_kl
        self.beta *= (1.0 + self.beta_lr * (self.kl_ema - self.cfg.kl_target))
        self.beta = float(np.clip(self.beta, 0.01, 10.0))

        self.entropy_coef *= self.cfg.entropy_decay
        self.entropy_coef = max(self.entropy_coef, getattr(self.cfg, 'entropy_min', 0.0))
        return stats


# 7. Training / evaluation entry points

def build_models(cfg=cfg):
    """Build the encoder, the single-level policy and the value network."""
    encoder = HGNNEncoder(cfg.dim_op, cfg.dim_mac, cfg.hidden_dim, cfg.enc_layers).to(DEVICE)
    policy = JointPolicy(cfg.hidden_dim, cfg.policy_hidden_dim).to(DEVICE)
    value = ValueNetwork(cfg.hidden_dim, cfg.policy_hidden_dim).to(DEVICE)
    return encoder, policy, value


class ValueNetwork(nn.Module):
    """State value network (same structure as the critic of the proposed method)."""

    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(emb_dim, hidden_dim), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0))

    def forward(self, g_emb):
        return self.net(g_emb)


def run_scheduler(file_path, model_path, seed_list, global_max_vals):
    """
    Run the trained model on a single instance and return [Cmax, LB, Tardy] per random seed.

    ``global_max_vals`` is the tuple (max_jobs, max_machines, max_total_ops,
    max_proc_time, max_due_date, max_capability) computed over the evaluation
    dataset. It is shared with the proposed method, Experiment2 and the ablation
    study so that every method sees identically normalised features.
    """
    (max_jobs, max_machines, max_total_ops,
     global_max_proc_time, global_max_due_date, global_max_capability) = global_max_vals
    jobs, machine_capabilities, due_dates, file_name, size = read_fjsp_instance(file_path)
    inst = MOFJSPInstance(jobs, machine_capabilities, due_dates, file_name, size,
                          global_max_proc_time, global_max_capability)

    encoder, policy, value = build_models()
    agent = SingleLevelPPOAgent(encoder, policy, value, cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    agent.load(model_path)
    agent.encoder.eval()
    agent.policy.eval()
    agent.value.eval()

    objectives_list = []
    runs = [(seed, False) for seed in seed_list]
    if seed_list:
        runs.append((seed_list[0], True))
    # The greedy rollout is appended after the sampled ones: greedy decoding is the
    # canonical inference of a policy and it dominates the sampled trajectories, so the
    for seed, deterministic in runs:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Build the environment from the global maxima (identical to training and to
        # the proposed method); using instance-local sizes would change the job-index
        # normalisation and the action-mask dimensions, i.e. the policy input domain.
        env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                             global_max_proc_time, global_max_due_date, HGATConfig)
        state = env.reset()
        n_ops = state['op_feats'].shape[0]
        n_mac = state['mac_feats'].shape[0]

        done = False
        steps = 0
        while not done and steps < cfg.max_steps_per_episode:
            index, _, _ = agent.act(state, inst, n_ops, n_mac, deterministic=deterministic)
            action = joint_index_to_env_action(index, n_mac, inst)
            state, _, done = env.step(action)
            steps += 1

        job_completion = env.job_completion_time.cpu().numpy()
        machine_loads = env.machine_load.cpu().numpy()
        cmax = float(job_completion.max())
        lb = float(np.std(machine_loads))
        tardy = float(np.sum(np.maximum(0, job_completion - np.array(due_dates))))
        objectives_list.append([cmax, lb, tardy])

    return objectives_list


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
                        max_proc_time, max_due_date, env_cfg=None, seed=0):
    """Rollouts of the validation instances; returns the best [Cmax, LB, Tardy] per instance.

    The objective is the same statistic the comparison reports (the best of one deterministic and
    fifty sampled rollouts), so the selected weights are the best of the run under the reported
    protocol and not the draw of a single greedy decode.
    """
    env_cfg = env_cfg if env_cfg is not None else HGATConfig
    agent.encoder.eval()
    agent.policy.eval()
    agent.value.eval()
    objectives = []
    for inst in validation:
        best = None
        for run_index in range(VALIDATION_ROLLOUTS):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                                 max_proc_time, max_due_date, env_cfg)
            state = env.reset()
            n_ops = state['op_feats'].shape[0]
            n_mac = state['mac_feats'].shape[0]
            steps = 0
            done = False
            while not done and steps < agent.cfg.max_steps_per_episode:
                index, _, _ = agent.act(state, inst, n_ops, n_mac,
                                        deterministic=(run_index == 0))
                state, _, done = env.step(joint_index_to_env_action(index, n_mac, inst))
                steps += 1
            job_completion = env.job_completion_time.cpu().numpy()
            machine_loads = env.machine_load.cpu().numpy()
            candidate = [float(job_completion.max()),
                         float(np.std(machine_loads)),
                         float(np.sum(np.maximum(0, job_completion
                                                 - np.array(inst.due_dates))))]
            if best is None or weighted_objective(candidate) < weighted_objective(best):
                best = candidate
        objectives.append(best)
    agent.encoder.train()
    agent.policy.train()
    agent.value.train()
    return objectives


def train(config=None):
    """Train on the self-built training set and keep the weights with the best validation objective."""
    config = config or cfg
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)

    (instance_list, max_jobs, max_machines, max_total_ops,
     max_proc_time, max_due_date, max_capability) = load_all_instances(INSTANCE_DIR, FILE_PATTERN)
    # Normalise the machine capability levels by the training-set maximum, exactly as
    # the proposed method does, so that training and evaluation use the same features.
    instances = [MOFJSPInstance(jobs, caps, due_dates, fname, size, max_proc_time, max_capability)
                 for (jobs, caps, due_dates, fname, size) in instance_list]
    sampler = CurriculumSampler(instances, HGATConfig)
    # The reported weights are selected on a fixed validation set, one that the curriculum
    # sampler cannot move: the sampler changes the instance distribution over the run, so the
    val_tuples = load_all_instances(INSTANCE_DIR, "mo_fjsp_*_val.txt")[0]
    pool_tuples = load_all_instances(INSTANCE_DIR, "mo_fjsp_*_train.txt")[0]
    validation_instances = [MOFJSPInstance(jobs, caps, due_dates, fname, size, max_proc_time,
                                           max_capability)
                            for (jobs, caps, due_dates, fname, size) in val_tuples]
    pool_instances = [MOFJSPInstance(jobs, caps, due_dates, fname, size, max_proc_time,
                                           max_capability)
                            for (jobs, caps, due_dates, fname, size) in pool_tuples]

    validation_set = build_validation_set(validation_instances, per_size=8)
    # Two instances of every size group of the training pool next to the eight held-out ones:
    # the wave that selected on the held-out split alone produced checkpoints that measured
    print(f"Validation set: {[inst.file_name for inst in validation_set]}")

    encoder, policy, value = build_models(config)
    agent = SingleLevelPPOAgent(encoder, policy, value, config)

    best_reward = -np.inf
    best_episode = 0
    trained_episodes = 0
    returns_history = []
    norm_returns_history = []
    # Validation bookkeeping: the relative objective of the fixed validation set selects the
    # reported weights, and the number of checks without an improvement stops the run.
    val_curve = []
    val_reference = None
    best_val = float('inf')
    stale_checks = 0
    model_path = os.path.join(config.model_save_dir, config.model_name)
    resume_path = os.path.join(config.model_save_dir, config.resume_name)

    # The rolling checkpoint is preferred over the best model: it also carries
    # the optimiser state and the episode counter, so a restart continues where
    # the interrupted run stopped instead of retraining from episode 0.
    start_episode = 0
    if os.path.exists(resume_path) and agent.load(resume_path) and agent.episode_counter > 0:
        start_episode = min(int(agent.episode_counter), config.n_episodes)
        # The rolling checkpoint carries the training curve of the interrupted run
        returns_history = list(agent.curve_returns)
        norm_returns_history = list(agent.curve_norm_returns)
        best_episode = start_episode
        if np.isfinite(agent.best_reward):
            best_reward = agent.best_reward
        val_curve = list(agent.val_curve)
        val_reference = list(agent.val_reference) or None
        best_val = float(agent.best_val)
        # The patience of the early stop survives a restart.  A queue that restarts a run after
        # every shutdown would otherwise grant it a fresh patience window and train it thousands
        stale_checks = 0
        running_best = float('inf')
        for _episode, _score in val_curve:
            if _score < running_best * (1.0 - config.val_tol):
                running_best = _score
                stale_checks = 0
            else:
                stale_checks += 1
        # An early stop is the decision of the run, not an interruption of it: the validation
        # objective of the model stopped improving, and more episodes would only be trained past
        if agent.converged:
            print(f"The rolling checkpoint already holds a converged run "
                  f"({start_episode} episodes); nothing to train.")
            return agent
        print(f"Continuing from episode {start_episode}/{config.n_episodes} "
              f"(best_reward={best_reward:.6f})")
    elif os.path.exists(resume_path):
        print(f"Rolling checkpoint {resume_path} has no episode counter, "
              f"training starts from scratch.")
    else:
        print(f"No rolling checkpoint {resume_path} found, training starts from scratch.")

    log_interval = max(1, int(getattr(config, 'log_interval', 100)))
    # Reference of the validation objective.  It is the weighted objective of a deterministic
    # earliest-completion dispatch of the same instance (dispatch_reference.py) instead of the
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from dispatch_reference import reference_objectives
    val_reference = [weighted_objective(obj) for obj in reference_objectives(
        [inst.file_name for inst in validation_set], INSTANCE_DIR)]
    # A curve written by the earlier frame holds a ratio to the first check of that run and is
    # not comparable with this one, so the run starts a fresh curve instead of mixing the two.
    restored_reference = list(getattr(agent, 'val_reference', []) or [])
    if val_curve and (abs(val_curve[0][1] - 1.0) < 1e-12
                      or (restored_reference and restored_reference != val_reference)):
        print("The restored validation curve is in an old frame or on another "
              "selection set and is reset.")
        val_curve = []
        best_val = float('inf')
        stale_checks = 0
    checkpoint_interval = max(1, int(getattr(config, 'checkpoint_interval', 25)))
    pbar = tqdm(total=config.n_episodes, initial=start_episode, desc="Training", unit="ep")

    # The instances are collected in lockstep instead of one at a time: the encoder, the joint
    # policy and the value network then process n_envs states per forward pass, which amortises the
    n_envs = max(1, int(getattr(config, 'n_envs', 1)))
    envs, insts, states = [], [], []
    trajectories, step_counts = [], []
    sampler.update_stage(min(1.0, start_episode / max(1, config.curriculum_episodes)))
    for _ in range(n_envs):
        inst = sampler.sample()
        envs.append(HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                                   max_proc_time, max_due_date, HGATConfig))
        insts.append(inst)
        states.append(envs[-1].reset())
        trajectories.append([])
        step_counts.append(0)

    n_ops = states[0]['op_feats'].shape[0]
    n_machines = states[0]['mac_feats'].shape[0]

    episode = start_episode
    # The validation and the early stop advance on the interval the episode counter falls into
    # instead of on an exact multiple of it.  The counter advances once per collected episode and
    last_val_interval = episode // max(1, config.val_interval)
    converged = False
    while episode < config.n_episodes and not converged:
        indices, log_probs, values_est = agent.act_batch(states, insts, n_ops, n_machines)

        for i in range(n_envs):
            if episode >= config.n_episodes:
                break

            index = indices[i]
            action = joint_index_to_env_action(index, n_machines, insts[i])
            next_state, reward, done = envs[i].step(action)
            trajectories[i].append({
                'state': states[i],
                'action': index.detach(),
                'log_prob': log_probs[i],
                'value': values_est[i],
                'reward': reward,
                'done': bool(done),
            })
            step_counts[i] += 1

            if not done and step_counts[i] < config.max_steps_per_episode:
                states[i] = next_state
                continue

            # The environment reached the end of its episode (or its step budget), so its
            # trajectory is complete: the networks are updated on it and the environment is reset
            # with a new instance of the curriculum.
            trajectory = trajectories[i]
            # A truncated episode (step budget reached) is treated as terminal so that no advantage
            # is bootstrapped from a non-existent successor state
            trajectory[-1]['done'] = True
            stats = agent.update(trajectory, insts[i])

            episode_return = float(torch.stack([t['reward'] for t in trajectory]).sum())
            returns_history.append(episode_return)

            # The return is normalised by the number of operations: the raw return of a long
            # episode is simply more negative, so a selection on the raw return would always keep
            # the weights of a small training instance.
            norm_return = episode_return / max(1, insts[i].total_ops)
            norm_returns_history.append(norm_return)
            episode += 1
            trained_episodes = episode

            # The smoothed return is the curve that the figure reports; it no longer selects the
            # weights, because two episodes of different curriculum stages are not comparable.
            if len(norm_returns_history) >= config.plateau_window:
                smoothed = float(np.mean(norm_returns_history[-config.plateau_window:]))
                if smoothed > best_reward:
                    best_reward = smoothed
                    best_episode = episode

                # Validation block.  It selects the reported weights and it stops the run as
                # soon as the average objective stops improving, i.e. the run is never trained
                # past the point where the policy keeps getting better.
                val_interval_index = episode // max(1, config.val_interval)
                if episode >= config.min_episodes and val_interval_index > last_val_interval:
                    last_val_interval = val_interval_index
                    objectives = evaluate_validation(
                        agent, validation_set, max_jobs, max_total_ops, max_machines,
                        max_proc_time, max_due_date, env_cfg=HGATConfig,
                        seed=int(getattr(config, 'val_seed', 0)))
                    scores = [weighted_objective(obj) for obj in objectives]
                    if val_reference is None:
                        # The first check defines the reference of every instance, so the score
                        # is a relative objective that does not depend on the instance size
                        # which dominates the raw weighted value.
                        val_reference = [value if abs(value) > 1e-9 else 1.0 for value in scores]
                    val_score = float(np.mean([value / ref
                                               for value, ref in zip(scores, val_reference)]))
                    val_curve.append((episode, val_score))
                    print(f"Validation {len(val_curve):3d} at episode {episode}: "
                          f"relative objective {val_score:.6f} (best {best_val:.6f})")
                    if val_score < best_val * (1.0 - config.val_tol):
                        best_val = val_score
                        stale_checks = 0
                        agent.episode_counter = episode
                        agent.best_reward = best_reward
                        agent.curve_returns = returns_history
                        agent.curve_norm_returns = norm_returns_history
                        agent.val_curve = list(val_curve)
                        agent.val_reference = list(val_reference)
                        agent.best_val = best_val
                        agent.converged = False
                        agent.save(model_path)
                    else:
                        stale_checks += 1
                        if stale_checks >= config.val_patience:
                            print(f"Early stop after {episode} episodes: the validation "
                                  f"objective did not improve for {stale_checks} checks "
                                  f"(best={best_val:.6f}).")
                            converged = True

            pbar.set_postfix({
                'return': f"{episode_return:.3f}",
                'best': f"{best_reward:.3f}",
                'steps': step_counts[i],
                'policy_loss': f"{stats.get('policy_loss', 0.0):.4f}",
                'value_loss': f"{stats.get('value_loss', 0.0):.4f}",
            })
            pbar.update(1)

            # Rolling checkpoint: it bounds the work a restart has to repeat to
            # checkpoint_interval episodes.
            if episode % checkpoint_interval == 0:
                agent.episode_counter = episode
                agent.best_reward = best_reward
                agent.curve_returns = returns_history
                agent.curve_norm_returns = norm_returns_history
                agent.save(resume_path)

            if episode % log_interval == 0:
                print(f"[{episode:5d}/{config.n_episodes}] return={episode_return:.3f} "
                      f"best={best_reward:.3f} steps={step_counts[i]} "
                      f"policy_loss={stats.get('policy_loss', 0.0):.4f} "
                      f"value_loss={stats.get('value_loss', 0.0):.4f}")

            if converged:
                break

            # Restart the environment that just finished with a new curriculum instance.
            sampler.update_stage(min(1.0, episode / max(1, config.curriculum_episodes)))
            inst = sampler.sample()
            envs[i] = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                                     max_proc_time, max_due_date, HGATConfig)
            insts[i] = inst
            states[i] = envs[i].reset()
            trajectories[i] = []
            step_counts[i] = 0

    pbar.close()

    # The finished run is checkpointed as well: a later restart of this script
    # finds every episode already trained and only refreshes the training curve.
    agent.episode_counter = trained_episodes
    agent.best_reward = best_reward
    agent.curve_returns = returns_history
    agent.curve_norm_returns = norm_returns_history
    agent.val_curve = list(val_curve)
    agent.val_reference = list(val_reference) if val_reference else []
    agent.best_val = best_val
    agent.converged = bool(converged)
    agent.save(resume_path)
    print("Training completed." if not converged else "Training stopped early (converged).")
    print(f"Best validation objective: {best_val:.6f}")

    # Validation curve: the relative objective of the fixed validation set.  It is the quantity
    # that selects the reported weights and that stops the run, so it is plotted next to the
    # training curve.
    if val_curve:
        fig_val, ax_val = plt.subplots(figsize=(8, 3.2))
        ax_val.plot([e for e, _ in val_curve], [s for _, s in val_curve],
                    marker='o', linewidth=1.2, color='tab:red')
        ax_val.set_xlabel('Episode')
        ax_val.set_ylabel('Relative validation objective')
        ax_val.set_title('Validation curve (lower is better)')
        plt.tight_layout()
        plt.savefig(os.path.join(config.figure_dir, 'validation_curve.png'), dpi=150)
        plt.close(fig_val)

    # Training curve
    # The raw return grows with the instance size, so the upper panel mostly shows the curriculum
    window = 25
    fig, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax_raw.plot(returns_history, linewidth=0.7, color='tab:grey')
    ax_raw.set_ylabel('Episode return (raw)')
    ax_raw.set_title('HGNN training curve')
    ax_norm.plot(norm_returns_history, linewidth=0.7, color='tab:blue', alpha=0.45)
    if len(norm_returns_history) >= window:
        moving = np.convolve(norm_returns_history, np.ones(window) / window, mode='valid')
        ax_norm.plot(range(window - 1, len(norm_returns_history)), moving,
                     linewidth=1.6, color='tab:blue')
    ax_norm.set_ylabel('Return per operation')
    ax_norm.set_xlabel('Episode')
    plt.tight_layout()
    plt.tight_layout()
    curve_path = os.path.join(config.figure_dir, 'training_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {curve_path}")
    print(f"Best episode return: {best_reward:.6f} (smoothed over {config.plateau_window} episodes, normalised by the number of operations)")
    return agent


def main():
    """Training entry point: runs Config.n_episodes episodes by default, --episodes overrides it."""
    parser = argparse.ArgumentParser(description="Train the single-level baseline.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override Config.n_episodes (used to fit the CPU time budget)")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes > 0:
        cfg.n_episodes = args.episodes
        print(f"n_episodes overridden to {cfg.n_episodes}")
    train()


if __name__ == "__main__":
    main()

