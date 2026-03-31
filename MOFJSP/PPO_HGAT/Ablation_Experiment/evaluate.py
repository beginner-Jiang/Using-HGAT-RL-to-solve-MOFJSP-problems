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

from common import (
    DEVICE, EPS, W1, W2, W3, SEED,
    COMPARISON_DIR, EXPERIMENTS,
    read_fjsp_instance, MOFJSPInstance,
    HeteroGraphEnv, HomogeneousGAT, HeteroGAT, MLPEncoder,
    UpperPolicy, LowerPolicy, Critic, JointPolicy,
    TorchRunningMeanStd, PPOAgent, cfg, PROJECT_ROOT, A5_MODEL_PATH
)

# Checkpoint and final result save paths
CHECK_POINT_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "Ablation_Experiment", "Check_Point")
FINAL_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "Ablation_Experiment", "Final")
os.makedirs(CHECK_POINT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

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
    checkpoint_path = os.path.join(CHECK_POINT_DIR, f"{exp_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

    # If checkpoint file exists, load and return
    if os.path.exists(checkpoint_path):
        print(f"\n[Checkpoint] Experiment {exp_name} already has results, loading: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            results = pickle.load(f)
        return exp_name, results

    print(f"\nEvaluating experiment: {exp_name}")

    model_dir = exp_config['model_dir']
    model_path = os.path.join(model_dir, "best.pth")

    # Check if model file exists
    if not os.path.exists(model_path):
        # If it's A5 and the pre-trained path exists, try to copy
        if exp_name == 'A5 (异构+带边特征+GAT+分层PPO) 完整模型' and os.path.exists(A5_MODEL_PATH):
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

    except Exception as e:
        print(f"Failed to infer dimensions from checkpoint: {e}, using current cfg settings (hidden_dim={cfg.gat_hidden_dim}, out_dim={cfg.gat_out_dim}, policy_hidden_dim={cfg.policy_hidden_dim})")
        has_layer_norm = True  # assume LayerNorm by default

    max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = global_max_vals

    dim_op, dim_mac = 3, 3

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
            self.use_disjunctive_edges = False
    eval_cfg = DummyConfig()

    results = {}
    # Use tqdm to show instance progress
    for fpath, inst in tqdm(instance_objs, desc=f"Instance evaluation", leave=False):
        env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                             max_proc_time, max_due_date, eval_cfg)
        agent.inst = inst
        objectives_list = []
        total_time = 0.0
        # Run each instance num_runs times
        for run in tqdm(range(num_runs), desc=f"Running {os.path.basename(fpath)}", leave=False):
            seed = SEED + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            state = env.reset()
            done = False
            start_time = time.time()
            while not done:
                # Use deterministic=False to maintain randomness
                actions, _, _ = agent.get_action_batch([state], [inst], deterministic=False)
                action = actions[0]
                next_state, _, done = env.step(action)
                state = next_state
            elapsed = time.time() - start_time
            total_time += elapsed

            job_completion = env.job_completion_time.cpu().numpy()
            machine_loads = env.machine_load.cpu().numpy()
            cmax = float(job_completion.max())
            lb = float(np.std(machine_loads))
            tardy = float(np.sum(np.maximum(0, job_completion - np.array(inst.due_dates))))
            objectives_list.append([cmax, lb, tardy])

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
def compute_hypervolume(points, ref_point):
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    n_samples = 10000
    samples = np.random.uniform(0, 1, (n_samples, 3)) * (ref_point - 0) + 0
    dominated = np.zeros(n_samples, dtype=bool)
    for p in points:
        dominated |= np.all(samples >= p, axis=1)
    hv = dominated.mean() * np.prod(ref_point)
    return hv

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


# Export final aggregated CSV
def export_final_csv(dim_metrics, a5_avg_F, file_to_size, dim_to_files):
    """
    Generates three CSV tables (small, medium, large scale), saves them in FINAL_DIR.
    Table rows are experiment names, columns are flattened "dimension_metric".
    Values are formatted with specified precision.
    """
    if not HAS_PANDAS:
        print("pandas not installed, cannot export CSV.")
        return

    # Metric names (consistent with table)
    metric_names = ['Avg F', 'GAP(%)', 'HV', 'GD', 'SP', 'Total Time(s)']
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

                # Format values
                df.loc[exp_name, f"{dim}_Avg F"] = f"{m['avg_F']:.2f}"
                df.loc[exp_name, f"{dim}_GAP(%)"] = f"{gap:.2f}"
                df.loc[exp_name, f"{dim}_HV"] = f"{m['HV']:.2e}"
                df.loc[exp_name, f"{dim}_GD"] = f"{m['GD']:.4f}"
                df.loc[exp_name, f"{dim}_SP"] = f"{m['SP']:.4f}"
                df.loc[exp_name, f"{dim}_Total Time(s)"] = f"{m['time']:.3f}"

        # Save as CSV
        csv_path = os.path.join(FINAL_DIR, f"{size.capitalize()}.csv")
        df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"Exported {size} scale results to: {csv_path}")


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
        inst = MOFJSPInstance(jobs, caps, due_dates, fname, size, max_proc_time, max_capability)
        all_instances.append((fpath, inst))
        file_to_size[fname] = size

    global_max_vals = (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability)
    print(f"Test instances global maximums: max_jobs={max_jobs}, max_machines={max_machines}, max_ops={max_total_ops}, "
          f"max_proc={max_proc_time:.2f}, max_due={max_due_date:.2f}, max_cap={max_capability}")

    # Before evaluation, check that model files for all experiments exist (A5 will attempt auto-copy)
    for exp in EXPERIMENTS:
        model_path = os.path.join(exp['model_dir'], "best.pth")
        if exp['name'] == 'A5 (异构+带边特征+GAT+分层PPO) 完整模型' and not os.path.exists(model_path):
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
    for exp in EXPERIMENTS:
        # Set num_runs to 50 to ensure enough samples for metrics like SP
        name, res = evaluate_experiment(exp, all_instances, global_max_vals, num_runs=50)
        if res is not None:
            all_results[name] = res

    if not all_results:
        print("No experiments successfully evaluated.")
        return

    # Determine global reference point (maximum of all solutions)
    all_objectives = []
    for exp_res in all_results.values():
        for fname, data in exp_res.items():
            all_objectives.extend(data['objectives'])
    all_objectives = np.array(all_objectives)
    ref_point = np.max(all_objectives, axis=0) * 1.1 if len(all_objectives) > 0 else np.array([1000, 1000, 1000])

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
        for exp_name, exp_res in all_results.items():
            avg_F_list = []
            hv_list = []
            gd_list = []
            sp_list = []
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

                hv = compute_hypervolume(obj_list, ref_point)
                hv_list.append(hv)

                sp = compute_sp(obj_list)
                sp_list.append(sp)

                true_pf = true_frontiers.get(fname, [])
                if len(true_pf) > 0:
                    gd = compute_gd(obj_list, true_pf)
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
                    'time': time_total
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

    # Export final CSV summary tables
    if HAS_PANDAS:
        export_final_csv(dim_metrics, a5_avg_F, file_to_size, dim_to_files)
    else:
        print("pandas not installed, skipping CSV export.")


if __name__ == "__main__":
    main()