# -*- coding: utf-8 -*-
"""Evaluators for the three non-heterogeneous learned baselines.

The public experiment needs MLP+DQN, MLP+PPO and GAT+PPO in addition to the
HGNN/DAN/RESCHED adapters in ``baseline_adapter.py``.  These three modules use
different state classes and environment wrappers, so each evaluator is kept
next to the module it wraps instead of pretending that one generic agent
interface fits all of them.

The evaluators use the checkpoints under ``model/`` and the family-local public
test split.  They do not train on the public split; the three graph baselines
retain their fine-tuning path in ``baseline_adapter.py``.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

COMMON = Path(__file__).resolve().parent.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

import public_benchmark as pb

EXPANDED = ("MLP+DQN", "MLP+PPO", "GAT+PPO")
MODEL_MAX_JOBS = 100
MODEL_MAX_MACHINES = 30
CHECKPOINTS = {
    "MLP+DQN": ("model", "dqn", "dqn_model.pth"),
    "MLP+PPO": ("model", "ppo", "ppo_model.pth"),
    "GAT+PPO": ("model", "ppo_gat", "ppo_gat_best.pth"),
}


def _module(project_root: Path, name: str):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    module_name = {
        "MLP+DQN": "DRL.DQN_MLP",
        "MLP+PPO": "DRL.PPO_MLP",
        "GAT+PPO": "DRL.PPO_GAT",
    }[name]
    return importlib.import_module(module_name)


def _checkpoint(project_root: Path, name: str) -> Path:
    return project_root.joinpath(*CHECKPOINTS[name])


def _internal_paths(code_data: Path, instances: Sequence[pb.BenchInstance]):
    return pb.materialize_internal_files(instances, code_data / "internal")


def _dqn_instances(module, instances, paths, geometry):
    cfg = module.Config()
    cfg.max_jobs, cfg.max_machines = MODEL_MAX_JOBS, MODEL_MAX_MACHINES
    cfg.global_max_proc_time, cfg.global_max_due_date = geometry[3], geometry[4]
    objects = []
    for item in instances:
        data = module.read_fjsp_instance_structured(str(paths[item.name]))
        objects.append(module.JobShopInstance(
            data, cfg.max_jobs, cfg.max_machines, item.name, item.size_label))
    return cfg, objects


def _evaluate_dqn(project_root: Path, code_data: Path, instances,
                  geometry, rollouts: int, seed: int) -> Dict[str, float]:
    import torch

    module = _module(project_root, "MLP+DQN")
    cfg, objects = _dqn_instances(module, instances, _internal_paths(code_data, instances), geometry)
    agent = module.DQN_Agent(cfg)
    checkpoint = torch.load(str(_checkpoint(project_root, "MLP+DQN")),
                            map_location=cfg.device, weights_only=False)
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.policy_net.eval()
    state_norm = None
    if all(key in checkpoint for key in ("norm_mean", "norm_var", "norm_count")):
        state_norm = module.RunningMeanStd(
            shape=(agent.state_dim,), device=cfg.device)
        state_norm.mean = torch.as_tensor(checkpoint["norm_mean"], device=cfg.device)
        state_norm.var = torch.as_tensor(checkpoint["norm_var"], device=cfg.device)
        state_norm.count = float(checkpoint["norm_count"])

    result = {}
    for inst in objects:
        best = None
        for run_index in range(rollouts):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = module.MOFJSP_Env(cfg, inst)
            raw_state = env.reset()
            done = False
            info = {}
            steps = 0
            limit = max(1, inst.total_ops + 1)
            while not done and steps < limit:
                state_t = torch.from_numpy(np.asarray(raw_state, dtype=np.float32)).to(cfg.device)
                if state_norm is not None:
                    state_t = state_norm.normalize(state_t)
                with torch.no_grad():
                    q_values = agent.policy_net(state_t.unsqueeze(0)).squeeze(0)
                if run_index == 0:
                    rule = int(q_values.argmax().item())
                else:
                    rule = int(torch.multinomial(torch.softmax(q_values, dim=0), 1).item())
                legal = module.rule_actions(env)
                action = int(legal[rule]) if len(legal) else 0
                next_raw, _, done, info = env.step(action)
                if next_raw is not None:
                    raw_state = next_raw
                steps += 1
            metrics = info.get("final_metrics") if done else None
            if metrics is None:
                remaining = max(0, int(inst.total_ops) - int(env.finished_ops_count))
                horizon = float(env.machine_avail_time[:inst.num_machines].max()) if inst.num_machines else 0.0
                horizon += remaining * float(inst.max_proc_time)
                metrics = {"Cmax": horizon}
            candidate = float(metrics["Cmax"])
            best = candidate if best is None else min(best, candidate)
        result[inst.file_name] = float(best)
    return result


def _ppo_mlp_instances(module, instances, paths, geometry):
    cfg = module.Config()
    cfg.max_jobs, cfg.max_machines = MODEL_MAX_JOBS, MODEL_MAX_MACHINES
    cfg.global_max_proc_time, cfg.global_max_due_date = geometry[3], geometry[4]
    objects = []
    for item in instances:
        jobs, mids, dues = module.read_fjsp_instance(str(paths[item.name]))
        objects.append(module.JobShopInstance(
            jobs, mids, dues, cfg.max_jobs, cfg.max_machines, item.name, item.size_label))
    return cfg, objects


def _evaluate_ppo_mlp(project_root: Path, code_data: Path, instances,
                      geometry, rollouts: int, seed: int) -> Dict[str, float]:
    import torch

    module = _module(project_root, "MLP+PPO")
    cfg, objects = _ppo_mlp_instances(
        module, instances, _internal_paths(code_data, instances), geometry)
    state_dim = 2 * cfg.max_machines + 3 * cfg.max_jobs
    action_dim = cfg.max_jobs * cfg.max_machines
    policy = module.PolicyNetwork(
        state_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers,
        use_mask_input=False).to(module.DEVICE)
    # The rolling checkpoint stores the training Config class from __main__.
    import __main__
    __main__.Config = module.Config
    checkpoint = torch.load(str(_checkpoint(project_root, "MLP+PPO")),
                            map_location=module.DEVICE, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    result = {}
    for inst in objects:
        best = None
        for run_index in range(rollouts):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = module.MOFJSP_Env(cfg, inst)
            state = env.reset()
            done = False
            info = {}
            steps = 0
            while not done and steps < max(1, inst.total_ops):
                mask = env.get_action_mask()
                if not mask.any():
                    break
                state_t = torch.from_numpy(state).to(module.DEVICE).unsqueeze(0)
                mask_t = torch.from_numpy(mask).to(module.DEVICE, dtype=torch.bool).unsqueeze(0)
                with torch.no_grad():
                    _, logits = policy(state_t, mask_t)
                    probs = torch.softmax(logits, dim=-1)
                    action = (int(torch.argmax(logits, dim=-1).item())
                              if run_index == 0 else int(torch.multinomial(probs, 1).item()))
                state, _, done, info = env.step(action)
                steps += 1
            metrics = info.get("final_metrics", {}) if done else {}
            candidate = float(metrics.get("Cmax", 0.0))
            if candidate > 0:
                best = candidate if best is None else min(best, candidate)
        result[inst.file_name] = float(best) if best is not None else float("nan")
    return result


def _to_device(value, device):
    import torch
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _ppo_gat_instances(module, instances, paths):
    objects = []
    for item in instances:
        jobs, mids, dues = module.read_fjsp_instance(str(paths[item.name]))
        objects.append(module.MOFJSPInstance(jobs, mids, dues, item.name, item.size_label))
    return objects


def _evaluate_ppo_gat(project_root: Path, code_data: Path, instances,
                      geometry, rollouts: int, seed: int) -> Dict[str, float]:
    import torch

    module = _module(project_root, "GAT+PPO")
    cfg = module.Config()
    objects = _ppo_gat_instances(
        module, instances, _internal_paths(code_data, instances))
    gat = module.HomogeneousGAT(
        8, 1, cfg.gat_hidden_dim, cfg.gat_out_dim,
        num_layers=cfg.gat_num_layers, num_heads=cfg.gat_num_heads).to(module.DEVICE)
    policy = module.PolicyNetwork(
        cfg.gat_out_dim, cfg.policy_hidden_dim, temperature=0.5).to(module.DEVICE)
    checkpoint = torch.load(str(_checkpoint(project_root, "GAT+PPO")),
                            map_location=module.DEVICE, weights_only=False)
    gat.load_state_dict(checkpoint["gat"])
    policy.load_state_dict(checkpoint["policy"])
    gat.eval()
    policy.eval()
    state_norm = None
    if checkpoint.get("state_norm_mean") is not None:
        state_norm = module.TorchRunningMeanStd(
            shape=(cfg.gat_out_dim,), device=module.DEVICE)
        state_norm.mean = checkpoint["state_norm_mean"].to(module.DEVICE).float()
        state_norm.var = checkpoint["state_norm_var"].to(module.DEVICE).float()
        state_norm.count = float(checkpoint.get("state_norm_count", state_norm.count))

    result = {}
    for inst in objects:
        best = None
        for run_index in range(rollouts):
            random.seed(seed + run_index)
            np.random.seed(seed + run_index)
            torch.manual_seed(seed + run_index)
            env = module.HomogeneousDisjunctiveGraphEnv(
                inst, MODEL_MAX_JOBS, MODEL_MAX_MACHINES, geometry[3], geometry[4], cfg)
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < cfg.max_steps_per_episode:
                with torch.no_grad():
                    state_device = _to_device(state, module.DEVICE)
                    global_emb, node_embs = gat(state_device)
                    emb = state_norm.normalize(global_emb.unsqueeze(0)).squeeze(0) \
                        if state_norm is not None else global_emb
                    probs = policy(emb, node_embs[state_device["op_node_idx"]],
                                   node_embs[state_device["mac_node_idx"]],
                                   state_device["action_mask"])
                    if torch.isnan(probs).any():
                        mask = state_device["action_mask"]
                        valid = torch.nonzero(mask).squeeze(-1)
                        probs = torch.zeros_like(probs)
                        probs[valid] = 1.0 / max(1, len(valid))
                    action = (int(torch.argmax(probs).item()) if run_index == 0
                              else int(torch.multinomial(probs, 1).item()))
                state, _, done, _ = env.step(action)
                steps += 1
            metrics = env.get_final_metrics()
            candidate = float(metrics["Cmax"])
            best = candidate if best is None else min(best, candidate)
        result[inst.file_name] = float(best) if best is not None else float("nan")
    return result


EVALUATORS = {
    "MLP+DQN": _evaluate_dqn,
    "MLP+PPO": _evaluate_ppo_mlp,
    "GAT+PPO": _evaluate_ppo_gat,
}


def cache_path(code_data: Path, family: str, method: str, rollouts: int, seed: int) -> Path:
    return code_data / "cache" / family / ("%s_r%d_s%d.json" % (method, rollouts, seed))


def evaluate_all(project_root: Path, code_data: Path, instances,
                 geometry, rollouts: int, seed: int = 42,
                 cache_tag: str = None) -> Dict[str, Dict[str, float]]:
    family = cache_tag or (instances[0].family if instances else "unknown")
    outputs = {}
    for method, evaluator in EVALUATORS.items():
        path = cache_path(code_data, family, method, rollouts, seed)
        if path.exists():
            outputs[method] = {key: float(value) for key, value in
                               json.loads(path.read_text(encoding="utf-8")).items()}
            print("    %s: cached" % method, flush=True)
            continue
        values = evaluator(project_root, code_data, instances, geometry, rollouts, seed)
        outputs[method] = values
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values, indent=1), encoding="utf-8")
    return outputs
