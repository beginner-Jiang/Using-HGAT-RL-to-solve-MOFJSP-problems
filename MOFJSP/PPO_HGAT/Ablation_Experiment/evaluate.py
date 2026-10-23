"""
PPO+HGAT Ablation Study Evaluation
Loads models from all experiments, tests them on the comparison dataset, outputs a table and saves as CSV.
Supports checkpoint resume: experiment results are saved in the Check_Point directory, finally aggregated into a CSV table.
Enforces that model files for all experiments must exist, otherwise exits with error.
If A5 model fails to load, automatically copies from the pre-trained path.
"""

import torch
import numpy as np
import random
import os
import sys
import glob
import time
import pickle
import shutil
from tqdm import tqdm

# Try to import pandas, give a warning but continue if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed, cannot export CSV files, will print results only.")

# Try to import scipy for the statistical significance tests
try:
    from scipy.stats import (
        wilcoxon,
        friedmanchisquare,
        rankdata,
        t as t_dist,
        f as f_dist,
    )
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, the statistical significance tests will be skipped.")

from common import (
    DEVICE, EPS, W1, W2, W3, SEED,
    COMPARISON_DIR, EXPERIMENTS,
    read_fjsp_instance, MOFJSPInstance,
    HeteroGraphEnv, HomogeneousGAT, HeteroGAT, MLPEncoder,
    UpperPolicy, LowerPolicy, Critic, JointPolicy,
    TorchRunningMeanStd, PPOAgent, cfg, PROJECT_ROOT, A5_MODEL_PATH
)

# Checkpoint and final result save paths
CHECK_POINT_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "PPO_HGAT", "Ablation_Experiment", "Check_Point")
FINAL_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "PPO_HGAT", "Ablation_Experiment", "Final")
os.makedirs(CHECK_POINT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

ABLATION_EVAL_BATCH = max(1, int(os.environ.get("ABLATION_EVAL_BATCH", "8")))

# Helper function: extract encoder state_dict from checkpoint
def extract_encoder_state_dict(checkpoint, exp_config):
    """
    Attempt to extract encoder state_dict from checkpoint, supporting multiple possible key names.
    If fails, raise RuntimeError.
    """
    possible_keys = ['encoder', 'gat', 'model', 'state_dict']
    if isinstance(checkpoint, dict):
        for key in possible_keys:
            if key in checkpoint:
                return checkpoint[key]
        # If the checkpoint itself is a state_dict (e.g., directly saved model parameters)
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError(f"Cannot extract encoder state_dict from checkpoint, available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")

# Evaluation function
def evaluate_experiment(exp_config, instance_objs, global_max_vals, num_runs=50):
    """
    instance_objs: list of (file_path, MOFJSPInstance)
    global_max_vals: (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability)
    num_runs: number of runs per instance
    Returns: (exp_name, results_dict)  results_dict: {filename: {'objectives': list, 'time': total_time}}
    """
    exp_name = exp_config['name']
    # A sharded run appends a tag to the checkpoint name of this variant so that several shards of the same variant never overwrite each other
    checkpoint_base = exp_name.replace(' ', '_').replace('(', '').replace(')', '')
    if _ABL_TAG:
        checkpoint_base = f"{checkpoint_base}_{_ABL_TAG}"
    checkpoint_path = os.path.join(CHECK_POINT_DIR, f"{checkpoint_base}.pkl")

    model_dir = exp_config['model_dir']
    model_path = os.path.join(model_dir, "best.pth")

    # A cache is valid only when it is newer than the selected checkpoint.  Reusing a cache
    # written from an older arm mixed generations in the previous ablation table.
    if os.path.exists(checkpoint_path):
        if os.path.exists(model_path) and os.path.getmtime(checkpoint_path) > os.path.getmtime(model_path):
            print(f"\n[Checkpoint] Experiment {exp_name} already has results, loading: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                results = pickle.load(f)
            return exp_name, results
        print(f"\n[Stale cache] {checkpoint_path} is older than {model_path}; re-evaluating.")

    print(f"\nEvaluating experiment: {exp_name}")

    # Check if model file exists
    if not os.path.exists(model_path):
        # If it's A5 and the pre-trained path exists, try to copy
        if exp_config.get('pretrained_path') and os.path.exists(A5_MODEL_PATH):
            print(f"Warning: A5 model file {model_path} does not exist, copying from pre-trained path {A5_MODEL_PATH}...")
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy(A5_MODEL_PATH, model_path)
            print(f"Copy completed.")
        else:
            raise FileNotFoundError(f"Model file does not exist: {model_path}, please train experiment {exp_name} first.")

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Error loading model file {model_path}: {e}")

    # Extract encoder state_dict
    try:
        encoder_state_dict = extract_encoder_state_dict(checkpoint, exp_config)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to extract encoder from checkpoint: {e}")

    # Infer dimensions based on encoder type
    original_hidden = cfg.gat_hidden_dim
    original_out = cfg.gat_out_dim
    original_policy_hidden = cfg.policy_hidden_dim
    try:
        if exp_config['use_homogeneous']:
            # Homogeneous GAT, get output dimension from the first linear layer
            found = False
            for key in encoder_state_dict.keys():
                if 'W.weight' in key and 'layers.0' in key:
                    out_dim = encoder_state_dict[key].shape[0]
                    found = True
                    break
            if not found:
                raise RuntimeError("Cannot infer output dimension from checkpoint")
            hidden_dim = out_dim  # assume hidden_dim = out_dim
            # Check for LayerNorm keys
            has_layer_norm = any('ln.weight' in key for key in encoder_state_dict.keys())
        elif exp_config['use_mlp_encoder']:
            # MLP encoder, infer from first linear layer of op_net
            out_dim = encoder_state_dict['op_net.0.weight'].shape[0]
            hidden_dim = out_dim
            has_layer_norm = False  # MLP has no LayerNorm
        else:
            # Heterogeneous GAT, infer from l1.W_op.weight
            out_dim = encoder_state_dict['l1.W_op.weight'].shape[0]
            hidden_dim = out_dim
            # Check for LayerNorm keys (note: heterogeneous GAT LayerNorm keys are l1.ln_op.weight etc.)
            has_layer_norm = any('ln' in key for key in encoder_state_dict.keys())

        # Temporarily modify cfg dimensions
        cfg.gat_hidden_dim = hidden_dim
        cfg.gat_out_dim = out_dim
        print(f"Adjusted model dimensions from checkpoint: hidden_dim={hidden_dim}, out_dim={out_dim}, use_layer_norm={has_layer_norm}")

        # Infer policy_hidden_dim
        if exp_config['use_hierarchical']:
            # Check upper dimensions
            if 'upper' in checkpoint and checkpoint['upper'] is not None:
                upper_state = checkpoint['upper']
                # net.0.weight shape: (hidden_dim, input_dim)
                input_dim_upper = upper_state['net.0.weight'].shape[1]
                hidden_dim_policy = upper_state['net.0.weight'].shape[0]
                # Verify input_dim equals cfg.gat_out_dim * 2
                expected_input = cfg.gat_out_dim * 2
                if input_dim_upper != expected_input:
                    print(f"Warning: upper input dimension {input_dim_upper} does not match expected {expected_input}, using inferred hidden dimension {hidden_dim_policy}")
            else:
                hidden_dim_policy = cfg.policy_hidden_dim  # default
        else:
            # Joint policy, infer from joint_policy.net.0.weight
            if 'joint_policy' in checkpoint and checkpoint['joint_policy'] is not None:
                joint_state = checkpoint['joint_policy']
                hidden_dim_policy = joint_state['net.0.weight'].shape[0]
            else:
                hidden_dim_policy = cfg.policy_hidden_dim

        cfg.policy_hidden_dim = hidden_dim_policy
        print(f"Adjusted policy_hidden_dim from checkpoint to {hidden_dim_policy}")

        # Comparing variants whose networks have a different capacity is not a controlled
        # experiment, so a mismatch between the width declared in EXPERIMENTS and the width found
        # in the checkpoint is reported explicitly instead of being silently accepted.
        declared_dims = (exp_config.get('gat_hidden_dim'), exp_config.get('gat_out_dim'),
                         exp_config.get('policy_hidden_dim'))
        found_dims = (cfg.gat_hidden_dim, cfg.gat_out_dim, cfg.policy_hidden_dim)
        if None not in declared_dims and tuple(declared_dims) != tuple(found_dims):
            print(f"WARNING: capacity mismatch for {exp_name}: the checkpoint holds {found_dims} while "
                  f"EXPERIMENTS declares {tuple(declared_dims)}; this variant is not capacity-matched.")

    except Exception as e:
        print(f"Failed to infer dimensions from checkpoint: {e}, using current cfg settings (hidden_dim={cfg.gat_hidden_dim}, out_dim={cfg.gat_out_dim}, policy_hidden_dim={cfg.policy_hidden_dim})")
        has_layer_norm = True  # assume LayerNorm by default

    max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = global_max_vals

    # The feature width follows arm A5 (the copy of the selected proposed model), so the
    # ablation follows a 3/3, 5/3 or 7/4 full model without editing the evaluation code.
    dim_op, dim_mac, feature_mode = 3, 3, "legacy"
    a5_path = os.path.join(os.path.dirname(exp_config["model_dir"]), "A5", "best.pth")
    if os.path.exists(a5_path):
        try:
            a5_encoder = torch.load(a5_path, map_location="cpu").get("encoder", {})
            if "l1.W_op.weight" in a5_encoder and "l1.W_mac.weight" in a5_encoder:
                dim_op = int(a5_encoder["l1.W_op.weight"].shape[1])
                dim_mac = int(a5_encoder["l1.W_mac.weight"].shape[1])
                feature_mode = {3: "legacy", 5: "augmented", 7: "full"}.get(dim_op, "full")
        except Exception as error:  # noqa: BLE001 - a missing A5 keeps the 7/4 default
            print(f"Could not infer the feature width from A5 ({error}); using 7/4")

    if exp_config['use_homogeneous']:
        encoder = HomogeneousGAT(
            in_dim=dim_op,
            hidden_dim=cfg.gat_hidden_dim,
            out_dim=cfg.gat_out_dim,
            num_layers=2,
            num_heads=4,
            use_edge_feat=exp_config['use_edge_feat'],
            use_layer_norm=has_layer_norm  # Use LayerNorm based on checkpoint
        ).to(DEVICE)
    elif exp_config['use_mlp_encoder']:
        encoder = MLPEncoder(dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim).to(DEVICE)
    else:
        encoder = HeteroGAT(
            dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim,
            use_edge_feat=exp_config['use_edge_feat'],
            use_layer_norm=has_layer_norm  # Pass has_layer_norm
        ).to(DEVICE)

    if exp_config['use_hierarchical']:
        upper = UpperPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
        lower = LowerPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
        critic_u = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
        critic_l = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
        joint_policy = None
    else:
        upper = None
        lower = None
        critic_u = Critic(cfg.gat_out_dim, cfg.policy_hidden_dim).to(DEVICE)
        critic_l = None
        joint_policy = JointPolicy(cfg.gat_out_dim, cfg.policy_hidden_dim,
                                   max_total_ops, max_machines).to(DEVICE)

    state_norm = TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=DEVICE)

    dummy_inst = instance_objs[0][1]
    agent = PPOAgent(dummy_inst, encoder, upper, lower, critic_u, critic_l, cfg,
                     state_norm, joint_policy, use_hierarchical=exp_config['use_hierarchical'])

    if not agent.load(model_path):
        raise RuntimeError(f"PPOAgent failed to load model: {model_path}")

    class DummyConfig:
        def __init__(self):
            self.reward_scaling = cfg.reward_scaling
            self.reward_clip = cfg.reward_clip
            # No disjunctive switch: HeteroGraphEnv builds the competing operation pairs of the
            # manuscript unconditionally and this script imports the same class as the trainer of
            # the arms, so the graph of the evaluation is the graph of the training.
    eval_cfg = DummyConfig()

    results = {}
    # Use tqdm to show instance progress
    for fpath, inst in tqdm(instance_objs, desc=f"Instance evaluation", leave=False):
        env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                             max_proc_time, max_due_date, eval_cfg,
                             feature_mode=feature_mode)
        agent.inst = inst
        objectives_list = []
        total_time = 0.0
        def objective_of(item):
            job_completion = item.job_completion_time.cpu().numpy()
            machine_loads = item.machine_load.cpu().numpy()
            cmax = float(job_completion.max())
            lb = float(np.std(machine_loads))
            tardy = float(np.sum(np.maximum(
                0, job_completion - np.array(inst.due_dates))))
            return [cmax, lb, tardy]

        agent.inst = inst
        objectives_list = []
        total_time = 0.0
        # The greedy rollout is the canonical inference of the policy.
        random.seed(SEED + num_runs)
        np.random.seed(SEED + num_runs)
        torch.manual_seed(SEED + num_runs)
        started = time.time()
        state = env.reset()
        done = False
        while not done:
            actions, _, _ = agent.get_action_batch(
                [state], [inst], deterministic=True)
            next_state, _, done = env.step(actions[0])
            state = next_state
        total_time += time.time() - started
        objectives_list.append(objective_of(env))

        # The sampled rollouts are decoded in batches, while every rollout still runs one
        # episode from its own reset and contributes one candidate to the same set.
        pool = [HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                               max_proc_time, max_due_date, eval_cfg,
                               feature_mode=feature_mode)
                for _ in range(min(ABLATION_EVAL_BATCH, num_runs))]
        for start in range(0, num_runs, ABLATION_EVAL_BATCH):
            width = min(ABLATION_EVAL_BATCH, num_runs - start)
            seed = SEED + start
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            envs = pool[:width]
            states = [item.reset() for item in envs]
            finished = [False] * width
            started = time.time()
            while not all(finished):
                live = [index for index in range(width) if not finished[index]]
                actions, _, _ = agent.get_action_batch(
                    [states[index] for index in live], [inst] * len(live),
                    deterministic=False)
                for slot, index in enumerate(live):
                    next_state, _, done = envs[index].step(actions[slot])
                    states[index] = next_state
                    if done:
                        finished[index] = True
            total_time += time.time() - started
            objectives_list.extend(objective_of(item) for item in envs)

        results[os.path.basename(fpath)] = {
            'objectives': objectives_list,
            'time': total_time
        }

    # Save checkpoint file
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Experiment results saved to checkpoint file: {checkpoint_path}")

    # Restore original cfg values
    cfg.gat_hidden_dim = original_hidden
    cfg.gat_out_dim = original_out
    cfg.policy_hidden_dim = original_policy_hidden
    return exp_name, results


# Multi-objective metric calculations
def _hypervolume_2d(points, ref_x, ref_y):
    """
    Exact area of the union of the rectangles [x, ref_x] x [y, ref_y] (minimization).
    """
    area = 0.0
    best_y = ref_y          # Upper border of the union at the current x (lowest y of the skyline)
    prev_x = None
    for x, y in sorted(points):
        if prev_x is not None and x > prev_x:
            area += (x - prev_x) * (ref_y - best_y)
        if y < best_y:
            best_y = y
        prev_x = x
    if prev_x is not None:
        area += (ref_x - prev_x) * (ref_y - best_y)
    return area

def compute_hypervolume(points, ref_point):
    """
    Exact three-dimensional hypervolume (all objectives are minimized).

    Every solution p defines the hyper-box [p, ref_point] and the HV is the volume of their
    union. For three objectives the volume can be obtained exactly by slicing along z: after
    sorting the solutions by z, the cross-section of the union inside a slice equals the area
    of the 2-D union of the "active" solutions (those with z <= the lower bound of the slice),
    so summing area * slice height over all slices gives the exact volume. Compared with a
    fixed-sample Monte-Carlo estimator this removes the sampling error and avoids ties between
    variants whose fronts are small relative to the reference box.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 3:
        raise ValueError("Only 3 objectives supported for hypervolume.")
    ref = np.asarray(ref_point, dtype=float)
    if np.any(np.isnan(pts)) or np.any(np.isnan(ref)):
        return np.nan

    # Only solutions strictly inside the reference box define a non-empty hyper-box
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    # Drop dominated solutions: their boxes are already covered by the others
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        if np.any(np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)):
            keep[i] = False
    pts = pts[keep]

    z_values = np.unique(pts[:, 2])
    volume = 0.0
    for idx, z_low in enumerate(z_values):
        z_high = float(z_values[idx + 1]) if idx + 1 < len(z_values) else float(ref[2])
        if z_high <= z_low:
            continue
        # Only solutions with z <= z_low cover the whole thickness of this slice
        active = pts[pts[:, 2] <= z_low][:, :2]
        volume += (z_high - z_low) * _hypervolume_2d([tuple(row) for row in active],
                                                     float(ref[0]), float(ref[1]))
    return float(volume)

def compute_gd(pf, reference_pf):
    pf = np.array(pf)
    ref = np.array(reference_pf)
    if len(pf) == 0 or len(ref) == 0:
        return np.nan
    distances = []
    for p in pf:
        dist = np.min(np.linalg.norm(ref - p, axis=1))
        distances.append(dist)
    return np.mean(distances)

def compute_sp(pf):
    pf = np.array(pf)
    if len(pf) < 2:
        return np.nan
    # Identical copies must not be scored as a perfectly uniform front: collapse them first
    # and report NaN when fewer than two distinct points remain.
    pf = np.unique(np.round(np.asarray(pf, dtype=float), 6), axis=0)
    # A two-point front has a spacing of zero by construction, so it carries no information
    # about uniformity; fewer than three distinct points are therefore not reported.
    if len(pf) < 3:
        return np.nan
    dists = []
    for i in range(len(pf)):
        others = np.concatenate([pf[:i], pf[i+1:]])
        if len(others) == 0:
            continue
        d = np.min(np.linalg.norm(others - pf[i], axis=1))
        dists.append(d)
    if len(dists) == 0:
        return np.nan
    mean_d = np.mean(dists)
    sp = np.sqrt(np.mean((dists - mean_d)**2))
    return sp

def nondominated_sort(objectives):
    objectives = np.array(objectives)
    n = len(objectives)
    if n == 0:
        return []
    dominated_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                dominated_solutions[i].append(j)
            elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for idx in fronts[i]:
            for j in dominated_solutions[idx]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts[0] if fronts and fronts[0] else []


# Ablation experiment: helpers required by the statistical significance tests

ABLATION_REFERENCE = next((e['name'] for e in EXPERIMENTS if 'A5' in e['name']), EXPERIMENTS[-1]['name'])
ABLATION_METHOD_ORDER = [e['name'] for e in EXPERIMENTS]

# Sharded (parallel) run support: 5 variants x 9 scales x 10 instances x 50 samples takes very long on CPU, whereas the evaluation
# results of the variants are independent (HV/GD/SP and the statistical tests are post-processed inside main), so the environment
_SHARD_VARIANTS = [v.strip().upper() for v in os.environ.get("ABLATION_VARIANTS", "").split(",")
                   if v.strip()]
# Sharding by scale is allowed in the same way (comma separated, e.g. "50x25,75x25,100x30"); the evaluation then covers only those scales, but the global maximum
# tuple is still computed over all instances so that every shard sees exactly the same feature normalisation as an unsharded run.
_ABL_SCALES = [s.strip() for s in os.environ.get("ABLATION_SCALES", "").split(",") if s.strip()]
# A shard may write its results into a tagged pkl (e.g. Check_Point/<variant>_10x5.pkl) so that several shards of the same variant
# never overwrite each other; once all shards finish, _Review/_merge_ablation_checkpoints.py merges them back into <variant>.pkl.
_ABL_TAG = os.environ.get("ABLATION_CHECKPOINT_TAG", "").strip()
_SHARDED = bool(_SHARD_VARIANTS or _ABL_SCALES or _ABL_TAG)


def selected_experiments():
    """Filter the variants to evaluate according to ABLATION_VARIANTS; return all variants when it is unset."""
    if not _SHARD_VARIANTS:
        return list(EXPERIMENTS)
    return [exp for exp in EXPERIMENTS
            if any(exp['name'].upper().startswith(v) for v in _SHARD_VARIANTS)]


def _aligned_metric(dim_data, method, list_key):
    """Convert the per-instance metrics of a variant on a scale into {instance name: metric value} so that the paired tests align strictly by instance."""
    data = dim_data.get(method, {})
    values = list(data.get(list_key, []))
    names = list(data.get('names', []))
    if names and len(names) == len(values):
        return {name: value for name, value in zip(names, values)}
    return {idx: value for idx, value in enumerate(values)}


def _common_instance_keys(metric_maps, methods):
    """Take the instance names shared by all variants, ordered according to the first variant in methods."""
    if not methods:
        return []
    reference_map = metric_maps[methods[0]]
    return [key for key in reference_map if all(key in metric_maps[m] for m in methods)]


def _paired_values(sample_a, sample_b):
    """Drop instances on which any variant has an empty (NaN) value while keeping the paired structure (lengths are already aligned by the shared instance names)."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return a[:0], b[:0]
    a, b = a[:n], b[:n]
    valid = ~(np.isnan(a) | np.isnan(b))
    return a[valid], b[valid]


# Ablation experiment statistical significance tests (the reference variant is the full model A5)
#

def paired_wilcoxon(sample_a, sample_b):
    """
    Paired Wilcoxon signed-rank test; returns (statistic, p-value).
    Returns (nan, nan) / (nan, 1.0) when the sample is too small or all differences are 0.
    """
    if not HAS_SCIPY:
        return np.nan, np.nan
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return np.nan, np.nan
    a, b = a[:n], b[:n]
    if np.allclose(a, b):
        return np.nan, 1.0
    try:
        statistic, p_value = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(statistic), float(p_value)
    except Exception:
        return np.nan, np.nan


def cliffs_delta(sample_a, sample_b):
    """
    Cliff's delta effect size, ranging in [-1, 1]. Common thresholds for |delta|: <0.147 negligible,
    <0.33 small, <0.474 medium, otherwise large. A positive value means sample a is overall larger.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    greater = float(np.sum(diff > 0))
    less = float(np.sum(diff < 0))
    return (greater - less) / (len(a) * len(b))


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction; returns the corrected p-values in the same order as the input."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    adjusted = np.full(m, np.nan)
    if m == 0:
        return adjusted
    order = np.argsort(np.where(np.isnan(p_values), np.inf, p_values))
    running_max = 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p_values[idx]):
            continue
        value = min(1.0, (m - rank) * float(p_values[idx]))
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def friedman_analysis(sample_matrix):
    """
    Friedman omnibus test. sample_matrix has shape (number of instances, number of variants) and holds the metric value of
    each variant on the corresponding instance. Returns (Friedman statistic, Friedman p, Iman-Davenport F, Iman-Davenport p, mean ranks).
    """
    if not HAS_SCIPY:
        return np.nan, np.nan, np.nan, np.nan, []
    data = np.asarray(sample_matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        return np.nan, np.nan, np.nan, np.nan, []
    # Drop instances that contain missing values
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < 2:
        return np.nan, np.nan, np.nan, np.nan, []

    n_instances, n_methods = data.shape
    ranks = np.apply_along_axis(rankdata, 1, data)
    average_ranks = ranks.mean(axis=0)

    try:
        statistic, p_value = friedmanchisquare(*[data[:, j] for j in range(n_methods)])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, average_ranks

    # Iman-Davenport correction: an F statistic that is more accurate than the chi-squared approximation
    denominator = n_instances * (n_methods - 1) - statistic
    if denominator > 1e-12:
        f_statistic = (n_instances - 1) * statistic / denominator
        # Use the survival function instead of 1-cdf: for a large F the subtraction cancels
        # catastrophically and saturates at ~1.1e-16, hiding the true p-value.
        p_f = float(f_dist.sf(f_statistic, n_methods - 1,
                              (n_methods - 1) * (n_instances - 1)))
    elif denominator > -1e-12:
        # When the variant ranking is identical on every instance the Friedman statistic reaches its upper bound,
        # the Iman-Davenport F statistic then diverges and the corresponding p-value tends to 0
        f_statistic, p_f = np.inf, 0.0
    else:
        f_statistic, p_f = np.nan, np.nan
    return float(statistic), float(p_value), float(f_statistic), p_f, average_ranks


def mean_std_ci95(samples):
    """Return (mean, standard deviation, 95% confidence interval half-width) using the t distribution critical value."""
    data = np.asarray(samples, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(data))
    if len(data) > 1:
        std = float(np.std(data, ddof=1))
        half = 0.0
        if HAS_SCIPY:
            half = float(t_dist.ppf(0.975, len(data) - 1) * std / np.sqrt(len(data)))
    else:
        std, half = 0.0, 0.0
    return mean, std, half


def compute_statistical_tests(dim_metrics, reference=ABLATION_REFERENCE):
    """
    Produce the statistical test results for every scale and every metric.

    Returns (pairwise_rows, friedman_rows):
        pairwise_rows: per-instance paired test results of the reference variant against the other variants
        friedman_rows: omnibus test results of all variants on that scale
    """
    pairwise_rows = []
    friedman_rows = []

    metric_specs = [('F', 'F_list'), ('HV', 'HV_list'), ('GD', 'GD_list'), ('SP', 'SP_list')]

    def dimension_sort_key(scale):
        jobs, machines = scale.split('x')
        return int(jobs), int(machines)

    for dim in sorted(dim_metrics.keys(), key=dimension_sort_key):
        dim_data = dim_metrics[dim]
        if reference not in dim_data:
            print(f"Warning: scale {dim} has no baseline {reference}; the test is skipped.")
            continue
        methods_here = [m for m in ABLATION_METHOD_ORDER if m in dim_data]

        for metric_name, list_key in metric_specs:
            # Align the per-instance metrics of the variants by instance name: the length of a per-instance list depends on the instances that
            # each variant actually returned successfully, and pairing by index would shift everything as soon as one variant misses an instance and would
            # force the Friedman test below to truncate by index; aligning by instance name keeps the three experiments (comparison 1/2, ablation) consistent.
            metric_maps = {m: _aligned_metric(dim_data, m, list_key) for m in methods_here}
            common_keys = _common_instance_keys(metric_maps, methods_here)
            if len(common_keys) < 2:
                continue
            ref_samples = np.asarray([metric_maps[reference][k] for k in common_keys], dtype=float)

            # (1) paired Wilcoxon signed-rank test + Holm-Bonferroni correction
            entries = []
            raw_p_values = []
            for method in methods_here:
                if method == reference:
                    continue
                samples = np.asarray([metric_maps[method][k] for k in common_keys], dtype=float)
                # Drop instances on which either value is empty (NaN) while keeping the paired structure
                ref_paired, samples_paired = _paired_values(ref_samples, samples)
                n = len(ref_paired)
                if n < 2:
                    continue
                statistic, p_value = paired_wilcoxon(ref_paired, samples_paired)
                raw_p_values.append(p_value)
                entries.append({
                    'Scale': dim,
                    'Metric': metric_name,
                    'Variant': method,
                    'Paired samples': n,
                    'W statistic': statistic,
                    'p-value': p_value,
                    'Holm-adjusted p': np.nan,
                    'Significant (0.05)': '',
                    'Cliff delta': cliffs_delta(ref_paired, samples_paired),
                    'Baseline mean': float(np.mean(ref_paired)),
                    'Baseline SD': float(np.std(ref_paired, ddof=1)),
                    'Variant mean': float(np.mean(samples_paired)),
                    'Variant SD': float(np.std(samples_paired, ddof=1)),
                })
            for entry, p_adjusted in zip(entries, holm_bonferroni(raw_p_values)):
                entry['Holm-adjusted p'] = p_adjusted
                entry['Significant (0.05)'] = 'Yes' if (not np.isnan(p_adjusted) and p_adjusted < 0.05) else 'No'
                pairwise_rows.append(entry)

            # (2) Friedman omnibus test
            if len(methods_here) >= 3:
                # All variants use the same set of instances (aligned by instance name) and are no longer truncated by index
                transposed = np.array([[metric_maps[m][k] for k in common_keys]
                                       for m in methods_here], dtype=float).T
                # Drop instances missing any variant value so that the Friedman test uses a complete paired sample
                transposed = transposed[~np.isnan(transposed).any(axis=1)]
                if transposed.shape[0] < 2:
                    continue
                statistic, p_value, f_statistic, p_f, average_ranks = friedman_analysis(transposed)
                friedman_rows.append({
                    'Scale': dim,
                    'Metric': metric_name,
                    'Variants': len(methods_here),
                    'Instances': int(transposed.shape[0]),
                    'Friedman statistic': statistic,
                    'Friedman p': p_value,
                    'Iman-Davenport F': f_statistic,
                    'Iman-Davenport p': p_f,
                    'Mean rank': '/'.join(f"{m}:{r:.2f}" for m, r in zip(methods_here, average_ranks))
                              if len(average_ranks) else '',
                })

    return pairwise_rows, friedman_rows


def print_statistical_tests(pairwise_rows, friedman_rows):
    """Print the statistical significance results to the console."""
    print("\n" + "=" * 120)
    print("Statistical significance results")
    print("=" * 120)

    print("\n[Paired Wilcoxon signed-rank test] full model A5 vs variants (Holm-Bonferroni)")
    header = (f"{'Scale':<8} | {'Metric':<5} | {'Variant':<12} | {'W':>10} | {'p-value':>10} | "
              f"{'Holm p':>10} | {'Sig.':<6} | {'Cliff delta':>11}")
    print(header)
    print("-" * len(header))
    for row in pairwise_rows:
        w_str = f"{row['W statistic']:.3f}" if not np.isnan(row['W statistic']) else "NaN"
        p_str = f"{row['p-value']:.4e}" if not np.isnan(row['p-value']) else "NaN"
        padj_str = f"{row['Holm-adjusted p']:.4e}" if not np.isnan(row['Holm-adjusted p']) else "NaN"
        delta_value = row['Cliff delta']
        delta_str = f"{delta_value:.3f}" if not np.isnan(delta_value) else "NaN"
        # Variant names are long, so the console only shows the index (A1~A5)
        variant_label = row['Variant'].split()[0]
        print(f"{row['Scale']:<8} | {row['Metric']:<5} | {variant_label:<12} | {w_str:>10} | {p_str:>10} | "
              f"{padj_str:>10} | {row['Significant (0.05)']:<6} | {delta_str:>11}")

    print("\n[Friedman test] differences among methods at each scale")
    header2 = (f"{'Scale':<8} | {'Metric':<5} | {'Variants':>5} | {'Instances':>5} | {'Friedman chi2':>12} | "
               f"{'p-value':>10} | {'ID-F':>10} | {'ID-p-value':>10}")
    print(header2)
    print("-" * len(header2))
    for row in friedman_rows:
        chi_str = f"{row['Friedman statistic']:.3f}" if not np.isnan(row['Friedman statistic']) else "NaN"
        p_str = f"{row['Friedman p']:.4e}" if not np.isnan(row['Friedman p']) else "NaN"
        f_str = f"{row['Iman-Davenport F']:.3f}" if not np.isnan(row['Iman-Davenport F']) else "NaN"
        pf_str = f"{row['Iman-Davenport p']:.4e}" if not np.isnan(row['Iman-Davenport p']) else "NaN"
        print(f"{row['Scale']:<8} | {row['Metric']:<5} | {row['Variants']:>5} | {row['Instances']:>5} | "
              f"{chi_str:>12} | {p_str:>10} | {f_str:>10} | {pf_str:>10}")


# Export final aggregated CSV
def export_final_csv(dim_metrics, a5_avg_F, file_to_size, dim_to_files, stats=None):
    """
    Generates three CSV tables (small, medium, large scale), saves them in FINAL_DIR.
    Table rows are experiment names, columns are flattened "dimension_metric".
    Values are formatted with specified precision.
    stats = (pairwise_rows, friedman_rows) additionally writes the statistical
    significance results to two separate CSV files.
    """
    if not HAS_PANDAS:
        print("pandas not installed, cannot export CSV.")
        return

    # Metric names (consistent with table); the dispersion measures required by
    # the reviewers (standard deviation and 95% confidence interval) are added
    # right after the average F value.
    metric_names = ['Avg F', 'F Std', 'F 95%CI', 'GAP(%)', 'HV', 'GD', 'SP', 'Total Time(s)']
    # Experiment name list (in original order)
    exp_names = [e['name'] for e in EXPERIMENTS]

    # Group dimensions by scale
    size_to_dims = {'small': [], 'medium': [], 'large': []}
    for dim in dim_metrics.keys():
        if dim in dim_to_files and dim_to_files[dim]:
            sample_fname = os.path.basename(sorted(dim_to_files[dim])[0])
            size = file_to_size.get(sample_fname, 'unknown')
            if size in size_to_dims:
                size_to_dims[size].append(dim)

    # Generate CSV for each scale
    for size, dims in size_to_dims.items():
        if not dims:
            continue
        dims = sorted(dims)
        # Create flattened column names, e.g., "10x5_Avg F"
        flat_columns = [f"{dim}_{metric}" for dim in dims for metric in metric_names]
        df = pd.DataFrame(index=exp_names, columns=flat_columns)

        for dim in dims:
            base = a5_avg_F.get(dim, None)
            for exp_name in exp_names:
                if exp_name not in dim_metrics[dim]:
                    continue
                m = dim_metrics[dim][exp_name]
                gap = (m['avg_F'] - base) / base * 100 if base is not None and base != 0 else 0.0

                # Dispersion of the weighted objective across the 10 instances
                _, f_std, f_half = mean_std_ci95(m.get('F_list', []))

                # Format values
                df.loc[exp_name, f"{dim}_Avg F"] = f"{m['avg_F']:.2f}"
                df.loc[exp_name, f"{dim}_F Std"] = f"{f_std:.4f}" if not np.isnan(f_std) else ""
                df.loc[exp_name, f"{dim}_F 95%CI"] = f"{f_half:.4f}" if not np.isnan(f_half) else ""
                df.loc[exp_name, f"{dim}_GAP(%)"] = f"{gap:.2f}"
                df.loc[exp_name, f"{dim}_HV"] = f"{m['HV']:.2e}"
                df.loc[exp_name, f"{dim}_GD"] = f"{m['GD']:.4f}"
                df.loc[exp_name, f"{dim}_SP"] = f"{m['SP']:.4f}"
                df.loc[exp_name, f"{dim}_Total Time(s)"] = f"{m['time']:.3f}"

        # Save as CSV
        csv_path = os.path.join(FINAL_DIR, f"{size.capitalize()}.csv")
        df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"Exported {size} scale results to: {csv_path}")

    # Statistical significance results (paired Wilcoxon + Friedman) as separate files
    if stats is not None:
        pairwise_rows, friedman_rows = stats
        if pairwise_rows:
            csv_path = os.path.join(FINAL_DIR, "Wilcoxon_Test.csv")
            pd.DataFrame(pairwise_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Exported paired Wilcoxon test results to: {csv_path}")
        if friedman_rows:
            csv_path = os.path.join(FINAL_DIR, "Friedman_Test.csv")
            pd.DataFrame(friedman_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Exported Friedman test results to: {csv_path}")


# Main function
def main():
    # Check comparison dataset directory
    if not os.path.exists(COMPARISON_DIR):
        print(f"Error: Comparison dataset directory {COMPARISON_DIR} does not exist. Please run Generate_DataSet2.py first to generate data.")
        sys.exit(1)

    # Collect all test instances and group by dimension
    pattern = "comp_*.txt"
    all_files = glob.glob(os.path.join(COMPARISON_DIR, pattern))
    if not all_files:
        print(f"Error: No instance files found in {COMPARISON_DIR}.")
        sys.exit(1)

    # Group by dimension (e.g., '10x5', '15x10', etc.)
    dim_to_files = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        if len(parts) >= 3:
            dim = parts[2]  # e.g., '10x5'
            if dim not in dim_to_files:
                dim_to_files[dim] = []
            dim_to_files[dim].append(fpath)

    # Keep only dimensions with at least 10 instances
    valid_dims = [dim for dim, flist in dim_to_files.items() if len(flist) >= 10]
    instance_files = []
    for dim in sorted(valid_dims):
        flist = sorted(dim_to_files[dim])[:10]
        instance_files.extend(flist)

    print(f"Collected {len(instance_files)} test instances (10 per dimension).")

    # Read all instances and build MOFJSPInstance objects, also compute global maximums
    all_instances = []  # (file_path, MOFJSPInstance)
    file_to_size = {}   # filename -> scale
    max_jobs = 0
    max_machines = 0
    max_proc_time = 0
    max_due_date = 0
    max_total_ops = 0
    max_capability = 0

    # First pass: read every instance and accumulate the global maxima
    parsed_instances = []
    for fpath in instance_files:
        jobs, caps, due_dates, fname, size = read_fjsp_instance(fpath)
        total_ops = sum(len(j) for j in jobs)
        max_jobs = max(max_jobs, len(jobs))
        max_machines = max(max_machines, len(caps))
        for job_ops in jobs:
            for op in job_ops:
                for t in op.machine_times.values():
                    max_proc_time = max(max_proc_time, t)
        max_due_date = max(max_due_date, max(due_dates) if due_dates else 0)
        max_total_ops = max(max_total_ops, total_ops)
        max_capability = max(max_capability, max(caps) if caps else 0)
        parsed_instances.append((fpath, jobs, caps, due_dates, fname, size))
        file_to_size[fname] = size

    # Second pass: build the MOFJSPInstance objects with the final global maxima.
    # Building them inside the loop above would normalise the instances read first
    all_instances = []
    for fpath, jobs, caps, due_dates, fname, size in parsed_instances:
        inst = MOFJSPInstance(jobs, caps, due_dates, fname, size, max_proc_time, max_capability)
        all_instances.append((fpath, inst))

    # Sharded run: the evaluation only covers the scales given by ABLATION_SCALES, but the global maximum above is computed over all instances
    # (exactly as in an unsharded run), so the feature normalisation of every shard is not shifted.
    if _ABL_SCALES:
        all_instances = [(fpath, inst) for fpath, inst in all_instances
                         if os.path.basename(fpath).split('_')[2] in _ABL_SCALES]
        print(f"Shard run: only scales {sorted(_ABL_SCALES)} are evaluated ({len(all_instances)} instances).")

    global_max_vals = (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability)
    print(f"Test instances global maximums: max_jobs={max_jobs}, max_machines={max_machines}, max_ops={max_total_ops}, "
          f"max_proc={max_proc_time:.2f}, max_due={max_due_date:.2f}, max_cap={max_capability}")

    # Before evaluation, check that model files for all experiments exist (A5 will attempt auto-copy)
    for exp in selected_experiments():
        model_path = os.path.join(exp['model_dir'], "best.pth")
        if exp.get('pretrained_path') and not os.path.exists(model_path):
            if os.path.exists(A5_MODEL_PATH):
                print(f"A5 model does not exist, copying from pre-trained path...")
                os.makedirs(exp['model_dir'], exist_ok=True)
                shutil.copy(A5_MODEL_PATH, model_path)
            else:
                raise FileNotFoundError(f"A5 pre-trained model {A5_MODEL_PATH} also does not exist, cannot evaluate.")
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for experiment {exp['name']} does not exist: {model_path}, please run train.py first.")

    # Evaluate all experiments (supports checkpoint resume)
    all_results = {}  # exp_name -> {filename: {'objectives': list, 'time': total_time}}
    for exp in selected_experiments():
        # Set num_runs to 50 to ensure enough samples for metrics like SP
        name, res = evaluate_experiment(exp, all_instances, global_max_vals, num_runs=50)
        if res is not None:
            all_results[name] = res

    if not all_results:
        print("No experiments successfully evaluated.")
        return

    # Sharded run: this process only writes its own variants into Check_Point while the post-processing and the export are left to the full run,
    # avoiding several shards rewriting the same summary table at the same time
    if _SHARDED:
        print(f"Shard run: variants {sorted(all_results.keys())} evaluated; aggregation and export skipped.")
        return

    # Reference point of every scale (1.1 x the largest solution of that scale).
    # The original implementation used the global maximum solution over all scales as a single reference point: the front of a small instance then sits in the corner of a huge reference box,
    dim_ref_points = {}
    for dim, flist in dim_to_files.items():
        if len(flist) < 10:
            continue
        dim_fnames = {os.path.basename(f) for f in sorted(flist)[:10]}
        dim_objs = []
        for exp_res in all_results.values():
            for fname, data in exp_res.items():
                if fname in dim_fnames:
                    dim_objs.extend(data['objectives'])
        dim_objs = np.array(dim_objs)
        dim_ref_points[dim] = (np.max(dim_objs, axis=0) * 1.1 if len(dim_objs) > 0
                               else np.array([1000.0, 1000.0, 1000.0]))

    # Compute true frontiers for each instance (union of all experiments' solutions)
    true_frontiers = {}
    for fpath, _ in all_instances:
        fname = os.path.basename(fpath)
        all_objs_inst = []
        for exp_res in all_results.values():
            if fname in exp_res:
                all_objs_inst.extend(exp_res[fname]['objectives'])
        if all_objs_inst:
            true_idx = nondominated_sort(all_objs_inst)
            true_frontiers[fname] = [all_objs_inst[i] for i in true_idx]
        else:
            true_frontiers[fname] = []

    # Compute average metrics per dimension per experiment
    dim_metrics = {}  # dim -> {exp_name: {'avg_F': , 'HV': , 'GD': , 'SP': , 'time': }}
    for dim, flist in dim_to_files.items():
        if len(flist) < 10:
            continue
        fnames = [os.path.basename(f) for f in sorted(flist)[:10]]
        ref_point = dim_ref_points.get(dim, np.array([1000.0, 1000.0, 1000.0]))
        for exp_name, exp_res in all_results.items():
            avg_F_list = []
            hv_list = []
            gd_list = []
            sp_list = []
            inst_names = []
            time_total = 0.0
            for fname in fnames:
                if fname not in exp_res:
                    continue
                data = exp_res[fname]
                obj_list = data['objectives']
                time_total += data['time']
                if len(obj_list) == 0:
                    continue
                w_objs = [W1*o[0] + W2*o[1] + W3*o[2] for o in obj_list]
                best_F = min(w_objs)
                avg_F_list.append(best_F)
                inst_names.append(fname)

                # Keep the same convention as comparison experiments 1/2: HV/GD/SP are computed uniformly on the obtained non-dominated front.
                # In this experiment every variant returns all solutions from 50 samples, including dominated ones; without filtering them first, GD/SP would be
                # noticeably inflated (HV is unaffected because of monotonicity) and the row of the full model A5 would not match HeteroGAT+PPO in comparison experiment 2.
                front_idx = nondominated_sort(obj_list)
                front = [obj_list[i] for i in front_idx] if front_idx else list(obj_list)

                hv = compute_hypervolume(front, ref_point)
                hv_list.append(hv)

                sp = compute_sp(front)
                if not np.isfinite(sp):
                    # A single non-dominated point has no spacing of its own.  Reporting zero
                    # would claim a perfectly uniform front, so the statistic is measured on
                    sampled = obj_list[1:] if len(obj_list) > 1 else list(obj_list)
                    sampled_idx = nondominated_sort(sampled)
                    sampled_front = [sampled[i] for i in sampled_idx] if sampled_idx else sampled
                    sp = compute_sp(sampled_front)
                    if not np.isfinite(sp):
                        # Even the sampled front is a single point: report the measured
                        # dispersion of the sampled solutions instead of a missing value.
                        sp = compute_sp(sampled)
                sp_list.append(sp)

                true_pf = true_frontiers.get(fname, [])
                if len(true_pf) > 0:
                    gd = compute_gd(front, true_pf)
                    gd_list.append(gd)
                else:
                    gd_list.append(np.nan)

            if avg_F_list:
                if dim not in dim_metrics:
                    dim_metrics[dim] = {}
                dim_metrics[dim][exp_name] = {
                    'avg_F': np.mean(avg_F_list),
                    'HV': np.mean(hv_list),
                    'GD': np.nanmean(gd_list),
                    'SP': np.nanmean(sp_list),
                    'time': time_total,
                    # Per-instance metric values kept for the statistical tests
                    'F_list': list(avg_F_list),
                    'HV_list': list(hv_list),
                    'GD_list': list(gd_list),
                    'SP_list': list(sp_list),
                    # The instance name is stored together with the per-instance metrics so that the paired samples of the statistical tests align strictly to the same instance
                    'names': list(inst_names)
                }

    # Get A5's avg_F per dimension as baseline
    a5_avg_F = {}
    for dim, metrics in dim_metrics.items():
        for exp_name, m in metrics.items():
            if 'A5' in exp_name:
                a5_avg_F[dim] = m['avg_F']
                break

    # Print table
    print("\n" + "="*120)
    print("Ablation Study Comparison Results (Average per Dimension)")
    print("="*120)

    header = f"{'Dim':<8} {'Experiment':<40} {'Avg F':>10} {'GAP(%)':>7} {'HV':>12} {'GD':>12} {'SP':>12} {'Total Time(s)':>10}"
    print(header)
    print("-" * len(header))

    for dim in sorted(dim_metrics.keys()):
        base = a5_avg_F.get(dim, None)
        for exp_name in [e['name'] for e in EXPERIMENTS]:
            if exp_name not in dim_metrics[dim]:
                continue
            m = dim_metrics[dim][exp_name]
            gap = (m['avg_F'] - base) / base * 100 if base is not None and base != 0 else 0.0
            print(f"{dim:<8} {exp_name:<40} {m['avg_F']:>10.2f} {gap:>7.2f} {m['HV']:>12.4f} {m['GD']:>12.4f} {m['SP']:>12.4f} {m['time']:>10.2f}")

    # Statistical significance tests (paired Wilcoxon against A5 + Friedman over variants)
    pairwise_rows, friedman_rows = compute_statistical_tests(dim_metrics)
    print_statistical_tests(pairwise_rows, friedman_rows)

    # Export final CSV summary tables
    if HAS_PANDAS:
        export_final_csv(dim_metrics, a5_avg_F, file_to_size, dim_to_files,
                         stats=(pairwise_rows, friedman_rows))
    else:
        print("pandas not installed, skipping CSV export.")


if __name__ == "__main__":
    main()



