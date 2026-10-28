# -*- coding: utf-8 -*-
"""Joint-vs-isolated hyperparameter comparison (4000 episodes).

Two configurations:
  C0     : beta=0.2,   gamma=1.0,  K=4, lr=2e-4  (adopted)
  Cjoint : beta=0.005, gamma=0.95, K=8, lr=5e-4  (isolated optima)

All non-parameter settings are identical: 4000 episodes, 30/30/40 curriculum,
the same data split, the same instance draw per seed, 4 environments, the same
learning-rate halving at episode 2000, and the same 51-rollout evaluation.
"""
import argparse
import json
import os
import random
import sys
import time

PARAM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Param_Experiment")
sys.path.insert(0, PARAM_DIR)
import numpy as np
import torch
import Param

CONFIGS = {
    "C0": dict(beta=0.2, gamma=1.0, K=4, lr=2e-4),
    "Cjoint": dict(beta=0.005, gamma=0.95, K=8, lr=5e-4),
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(CONFIGS))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=4000)
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfgv = CONFIGS[args.config]
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

instances, max_jobs, max_machines, max_ops, max_proc, max_due, max_cap = \
    Param.load_all_instances(Param.INSTANCE_DIR, Param.FILE_PATTERN)
rng = random.Random(42)
rng.shuffle(instances)
n = len(instances)
train_end, val_end = int(0.6 * n), int(0.8 * n)
train_t, val_t, test_t = instances[:train_end], instances[train_end:val_end], instances[val_end:]
train_i = [Param.MOFJSPInstance(j, c, d, f, s, max_proc, max_cap) for j, c, d, f, s in train_t]
val_i = [Param.MOFJSPInstance(j, c, d, f, s, max_proc, max_cap) for j, c, d, f, s in val_t]
test_i = [Param.MOFJSPInstance(j, c, d, f, s, max_proc, max_cap) for j, c, d, f, s in test_t]

small = [i for i in train_i if getattr(i, "size", "") == "small"] or train_i
medium = [i for i in train_i if getattr(i, "size", "") == "medium"]
large = [i for i in train_i if getattr(i, "size", "") == "large"]
probe_inst = val_i[0] if val_i else test_i[0]


class Config:
    def __init__(self):
        self.w1, self.w2, self.w3 = 0.4, 0.3, 0.3
        self.gamma = cfgv["gamma"]
        self.beta = cfgv["beta"]
        self.K = cfgv["K"]
        self.reward_scaling = Param.REWARD_SCALING
        self.reward_clip = Param.REWARD_CLIP


cfg = Config()
geometry = (max_jobs, max_ops, max_machines, max_proc, max_due)
gat = Param.HeteroGAT(3, 3, 256, 256, num_heads=cfg.K).to(Param.DEVICE)
upper = Param.UpperPolicy(256, 512).to(Param.DEVICE)
lower = Param.LowerPolicy(256, 512).to(Param.DEVICE)
cu = Param.Critic(256, 512).to(Param.DEVICE)
cl = Param.Critic(256, 512).to(Param.DEVICE)
sn = Param.TorchRunningMeanStd(shape=(256,), device=Param.DEVICE)
agent = Param.HierarchicalPPOAgent(None, gat, upper, lower, cu, cl,
                                   gamma=cfg.gamma, beta=cfg.beta, lr=cfgv["lr"],
                                   lam=1.0, state_norm=sn)

halved = {"done": False}
inst_rng = random.Random(args.seed)


def next_instance(index):
    if not halved["done"] and index >= 2000:
        for group in agent.opt.param_groups:
            group["lr"] *= 0.5
        halved["done"] = True
    progress = index / max(1, args.epochs)
    if progress < 0.3 or not medium:
        pool = small
    elif progress < 0.6 or not large:
        pool = small + medium
    else:
        pool = small + medium + large
    return inst_rng.choice(pool)


def probe():
    agent.gat.eval(); agent.upper.eval(); agent.lower.eval()
    agent.critic_u.eval(); agent.critic_l.eval()
    env = Param.HeteroGraphEnv(probe_inst, max_jobs, max_ops, max_machines,
                               max_proc, max_due, cfg)
    state = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            actions, _, _ = agent.get_action_batch([state], [probe_inst],
                                                   deterministic=True, check_invalid=False)
            state, _, done = env.step(actions[0])
    c = max(env.job_completion_time).item()
    lb = torch.std(env.machine_load).item() if env.machine_load.numel() else 0.0
    t = torch.sum(torch.clamp(env.job_completion_time - env.due_dates_tensor, min=0)).item()
    return cfg.w1 * c + cfg.w2 * lb + cfg.w3 * t, c, lb, t


curve = []
orig_update = agent.update
counter = {"n": 0}


def wrapped_update(traj, inst):
    orig_update(traj, inst)
    counter["n"] += 1
    if counter["n"] % 500 == 0 or counter["n"] == args.epochs:
        f, c, lb, t = probe()
        curve.append(dict(episode=counter["n"], F=float(f), Cmax=float(c),
                          LB=float(lb), Tardy=float(t)))
        print("probe", counter["n"], round(f, 2), flush=True)


agent.update = wrapped_update
t0 = time.time()
Param.train_collector(agent, cfg, next_instance, args.epochs, geometry,
                      num_envs=args.envs, label="%s-s%d" % (args.config, args.seed),
                      report_every=250)
elapsed = time.time() - t0
agent.update = orig_update
mean_f, mean_c, mean_lb, mean_t, f_list = Param.evaluate(
    agent, test_i, max_jobs, max_ops, max_machines, max_proc, max_due, cfg,
    deterministic=True)

row = dict(config=args.config, seed=args.seed, epochs=args.epochs,
           beta=cfgv["beta"], gamma=cfgv["gamma"], K=cfgv["K"], lr=cfgv["lr"],
           meanF=float(mean_f), meanCmax=float(mean_c), meanLB=float(mean_lb),
           meanT=float(mean_t), elapsed=float(elapsed),
           F_list=[float(x) for x in f_list], curve=curve)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as handle:
    json.dump(row, handle, indent=2)
print(json.dumps({k: row[k] for k in ("config", "seed", "meanF", "elapsed")}))
