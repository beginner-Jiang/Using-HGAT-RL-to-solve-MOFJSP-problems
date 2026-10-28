# -*- coding: utf-8 -*-
"""Adapters for the three learning baselines: HGNN, DAN and RESCHED.

The baselines share the environment and the single-level PPO agent implemented
in the project, but each owns a different encoder.  This module gives them the
same public-benchmark split, a makespan-only reward, the same rollout count and
the same single seed as the proposed method.

The public files are never rewritten.  A one-based working copy with two
metadata lines is generated in the experiment data folder for the original
``read_fjsp_instance`` reader.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

COMMON = Path(__file__).resolve().parent.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

import public_benchmark as pb
import expanded_baselines

BASELINES = ("MLP+DQN", "MLP+PPO", "GAT+PPO", "HGNN", "DAN", "RESCHED")
TRAINABLE = ("HGNN", "DAN", "RESCHED")
MODULE_NAMES = {
    "HGNN": "DRL.HGNN",
    "DAN": "DRL.DAN",
    "RESCHED": "DRL.RESCHED",
}
CHECKPOINT_FILES = {
    "MLP+DQN": ("model", "dqn", "dqn_model.pth"),
    "MLP+PPO": ("model", "ppo", "ppo_model.pth"),
    "GAT+PPO": ("model", "ppo_gat", "ppo_gat_best.pth"),
    "HGNN": ("model", "hgnn", "hgnn_best.pth"),
    "DAN": ("model", "dan", "dan_best.pth"),
    "RESCHED": ("model", "resched", "resched_best.pth"),
}


def import_baseline(project_root: Path, name: str):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    module = importlib.import_module(MODULE_NAMES[name])
    # The reward and the feature tables are module globals in the shared
    # environment.  A makespan-only run must set them before any rollout.
    module.W1 = 1.0
    module.W2 = 0.0
    module.W3 = 0.0
    base = importlib.import_module("HGHRL.HGHRL")
    base.W1 = 1.0
    base.W2 = 0.0
    base.W3 = 0.0
    return module


def load_agent(module, checkpoint: Path, lr: float = 2e-4):
    import torch

    pb._allow_foreign_numpy_checkpoint()
    encoder, policy, value = module.build_models()
    agent = module.SingleLevelPPOAgent(encoder, policy, value, module.cfg)
    checkpoint = torch.load(checkpoint, map_location=module.DEVICE, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    value.load_state_dict(checkpoint["value"], strict=True)
    encoder = encoder.to(module.DEVICE).float()
    policy = policy.to(module.DEVICE).float()
    value = value.to(module.DEVICE).float()
    agent.entropy_coef = float(checkpoint.get("entropy_coef", agent.entropy_coef))
    agent.beta = float(checkpoint.get("beta", agent.beta))
    for group in agent.opt.param_groups:
        group["lr"] = float(lr)
    return agent, checkpoint


def _instance_objects(module, public_instances: Sequence[pb.BenchInstance],
                      internal_paths: Dict[str, Path], geometry) -> List:
    objects = []
    for item in public_instances:
        jobs, caps, due_dates, file_name, size = module.read_fjsp_instance(
            str(internal_paths[item.name]))
        objects.append(module.MOFJSPInstance(
            jobs, caps, due_dates, file_name, size, geometry[3], 1.0))
    return objects


def _new_env(module, inst, geometry):
    return module.HeteroGraphEnv(
        inst, geometry[0], geometry[1], geometry[2], geometry[3], geometry[4],
        module.HGATConfig,
    )


def evaluate_agent(module, agent, parameter_instances: Sequence,
                   geometry, rollouts: int = 11, batch: int = 4) -> Dict[str, float]:
    """Best makespan of one deterministic and ``rollouts-1`` sampled policies."""
    import torch

    agent.encoder.eval()
    agent.policy.eval()
    agent.value.eval()
    result = {}
    with torch.no_grad():
        for inst in parameter_instances:
            best = None

            def traverse(width: int, deterministic: bool):
                nonlocal best
                envs = [_new_env(module, inst, geometry) for _ in range(width)]
                states = [env.reset() for env in envs]
                finished = [False] * width
                n_ops = states[0]["op_feats"].shape[0]
                n_mac = states[0]["mac_feats"].shape[0]
                while not all(finished):
                    live = [index for index in range(width) if not finished[index]]
                    indices, _, _ = agent.act_batch(
                        [states[index] for index in live], [inst] * len(live),
                        n_ops, n_mac, deterministic=deterministic,
                    )
                    for slot, index in enumerate(live):
                        action = module.joint_index_to_env_action(indices[slot], n_mac, inst)
                        next_state, _, done = envs[index].step(action)
                        states[index] = next_state
                        finished[index] = bool(done)
                for env in envs:
                    cmax = float(env.job_completion_time.max().item())
                    best = cmax if best is None else min(best, cmax)

            traverse(1, True)
            remaining = max(0, rollouts - 1)
            while remaining > 0:
                width = min(batch, remaining)
                traverse(width, False)
                remaining -= width
            result[inst.file_name.replace("public_", "").rsplit("_", 1)[0]] = float(best)
    return result


def _relative_score(values: Dict[str, float], parameter_instances: Sequence) -> float:
    gaps = []
    for item in parameter_instances:
        name = item.file_name.replace("public_", "").rsplit("_", 1)[0]
        bks = (pb.BRANDIMARTE_BKS.get(name) if name.startswith("mk")
               else pb.LAWRENCE_BKS.get(name))
        if name in values and bks:
            gaps.append((values[name] - bks) / bks * 100.0)
    return float(np.mean(gaps)) if gaps else float("nan")


def fine_tune(module, agent, train_instances: Sequence, validation_instances: Sequence,
              geometry, episodes: int, envs_count: int, lr: float, seed: int,
              checkpoint_dir: Path, name: str, val_every: int = 100,
              val_rollouts: int = 2) -> Path:
    """Fine-tune one baseline with the same single-seed makespan-only protocol."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for group in agent.opt.param_groups:
        group["lr"] = float(lr)
    module.cfg.gamma = 1.0
    module.cfg.gae_lambda = 0.95
    agent.encoder.train()
    agent.policy.train()
    agent.value.train()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / ("%s_public_best.pth" % name.lower())
    last_path = checkpoint_dir / ("%s_public_last.pth" % name.lower())
    order = list(train_instances)
    cursor = 0

    envs, insts, states, trajectories = [], [], [], []
    for _ in range(max(1, envs_count)):
        inst = order[cursor % len(order)]
        cursor += 1
        env = _new_env(module, inst, geometry)
        envs.append(env)
        insts.append(inst)
        states.append(env.reset())
        trajectories.append([])
    n_ops = states[0]["op_feats"].shape[0]
    n_mac = states[0]["mac_feats"].shape[0]

    best_score = float("inf")
    episode = 0
    last_val = 0
    started = time.time()
    while episode < episodes:
        indices, log_probs, values_est = agent.act_batch(states, insts, n_ops, n_mac)
        for index in range(len(envs)):
            action = module.joint_index_to_env_action(indices[index], n_mac, insts[index])
            next_state, reward, done = envs[index].step(action)
            trajectories[index].append({
                "state": states[index],
                "action": indices[index].detach(),
                "log_prob": log_probs[index],
                "value": values_est[index],
                "reward": reward,
                "done": bool(done),
            })
            if not done and len(trajectories[index]) < agent.cfg.max_steps_per_episode:
                states[index] = next_state
                continue

            trajectory = trajectories[index]
            trajectory[-1]["done"] = True
            agent.update(trajectory, insts[index])
            episode += 1
            if episode % max(1, episodes // 5) == 0 or episode == episodes:
                elapsed = time.time() - started
                print("    %s: episode %d/%d (%.2f s/ep)"
                      % (name, episode, episodes, elapsed / max(1, episode)), flush=True)
            if episode >= episodes:
                break

            inst = order[cursor % len(order)]
            cursor += 1
            envs[index] = _new_env(module, inst, geometry)
            insts[index] = inst
            states[index] = envs[index].reset()
            trajectories[index] = []

        if episode >= last_val + val_every:
            last_val = episode
            val_values = evaluate_agent(module, agent, validation_instances, geometry,
                                        rollouts=val_rollouts, batch=min(2, val_rollouts))
            score = _relative_score(val_values, validation_instances)
            print("    %s validation at %d: mean gap %.3f%%" % (name, episode, score), flush=True)
            agent.episode_counter = episode
            if math.isfinite(score) and score < best_score:
                best_score = score
                agent.save(str(best_path))
                print("    %s saved new best checkpoint" % name, flush=True)
            agent.encoder.train()
            agent.policy.train()
            agent.value.train()

    agent.episode_counter = episode
    agent.save(str(last_path))
    if not best_path.exists():
        agent.save(str(best_path))
    return best_path


def default_checkpoint(project_root: Path, name: str) -> Path:
    return project_root.joinpath(*CHECKPOINT_FILES[name])


def train_all(project_root: Path, code_data: Path, train: Sequence[pb.BenchInstance],
              validation: Sequence[pb.BenchInstance], geometry, episodes: int,
              envs_count: int, lr: float, seed: int, family: str) -> Dict[str, str]:
    internal_dir = code_data / "internal"
    internal_paths = pb.materialize_internal_files(list(train) + list(validation), internal_dir)
    outputs = {}
    for name in TRAINABLE:
        print("== Fine-tuning %s ==" % name, flush=True)
        module = import_baseline(project_root, name)
        train_objects = _instance_objects(module, train, internal_paths, geometry)
        val_objects = _instance_objects(module, validation, internal_paths, geometry)
        checkpoint = default_checkpoint(project_root, name)
        agent, _ = load_agent(module, checkpoint, lr=lr)
        best = fine_tune(module, agent, train_objects, val_objects, geometry,
                         episodes=episodes, envs_count=envs_count, lr=lr,
                         seed=seed, checkpoint_dir=code_data / "checkpoints" / family, name=name)
        outputs[name] = str(best)
        # Release the CUDA graph between baselines.
        del agent, module, train_objects, val_objects
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    for name in expanded_baselines.EXPANDED:
        outputs[name] = str(expanded_baselines._checkpoint(project_root, name))
    return outputs


def selection_map(code_data: Path, family: str) -> Dict[str, str]:
    """Return the validation-selected checkpoint of each trainable baseline."""
    path = code_data / "checkpoints" / family / "selection.json"
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(key): str(value) for key, value in blob.get("checkpoints", {}).items()}


def evaluate_all(project_root: Path, code_data: Path, test: Sequence[pb.BenchInstance],
                 geometry, rollouts: int, seed: int = 42,
                 family: str = "Lawrence",
                 selection: Dict[str, str] = None,
                 cache_tag: str = None) -> Dict[str, Dict[str, float]]:
    internal_dir = code_data / "internal"
    internal_paths = pb.materialize_internal_files(list(test), internal_dir)
    outputs = expanded_baselines.evaluate_all(
        project_root, code_data, test, geometry, rollouts=rollouts, seed=seed,
        cache_tag=cache_tag)
    selection = dict(selection or selection_map(code_data, family))
    tag = cache_tag or family
    for name in TRAINABLE:
        cache = (code_data / "cache" / tag
                 / ("%s_r%d_s%d.json" % (name, rollouts, seed)))
        if cache.exists():
            outputs[name] = {key: float(value) for key, value in
                             json.loads(cache.read_text(encoding="utf-8")).items()}
            print("    %s: cached" % name, flush=True)
            continue
        print("== Evaluating %s ==" % name, flush=True)
        module = import_baseline(project_root, name)
        objects = _instance_objects(module, test, internal_paths, geometry)
        fine_tuned = code_data / "checkpoints" / family / ("%s_public_best.pth" % name.lower())
        chosen = selection.get(name)
        if chosen and Path(chosen).exists():
            checkpoint = Path(chosen)
        elif fine_tuned.exists():
            checkpoint = fine_tuned
        else:
            checkpoint = default_checkpoint(project_root, name)
        print("    checkpoint: %s" % checkpoint, flush=True)
        agent, _ = load_agent(module, checkpoint, lr=2e-4)
        outputs[name] = evaluate_agent(module, agent, objects, geometry,
                                       rollouts=rollouts, batch=min(8, rollouts))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(outputs[name], indent=1), encoding="utf-8")
        del agent, module, objects
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return outputs

