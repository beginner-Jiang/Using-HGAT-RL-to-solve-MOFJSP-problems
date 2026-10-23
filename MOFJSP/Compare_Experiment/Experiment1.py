"""
Experiment 1: simple dispatching rules vs PPO_HGAT (pre-trained model)
The compared scales (jobs x machines) come from a fixed list and every scale holds 10 instances.

The evaluation protocol is identical to Experiment 2 and to the ablation study: every reinforcement
learning method builds its instance and environment from the global maximum tuple
(max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability) computed
over the whole comparison dataset, and additionally reports the paired Wilcoxon signed-rank test, the
Friedman test, the Cliff's delta effect size and the 95% confidence interval of the mean.
"""

import os
import time
import glob
import sys
import pickle
import numpy as np
import random
import torch
from collections import defaultdict

# pandas is used for the Excel export
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas is not installed; Excel export is skipped and results are printed.")

# scipy components required by the significance tests (paired Wilcoxon, Friedman, t and F quantiles)
try:
    from scipy.stats import (wilcoxon,
                             friedmanchisquare,
                             rankdata,
                             t as t_dist,
                             f as f_dist)
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy is not installed; only means and standard deviations are reported.")


# Automatic project root detection

def find_project_root():
    """Walk upwards from this file until a directory containing comparison_instances is found."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check the directory of this script
    if os.path.isdir(os.path.join(script_dir, "DataSet", "comparison_instances")):
        return script_dir
    # Check the parent directory
    parent_dir = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(parent_dir, "DataSet", "comparison_instances")):
        return parent_dir
    # Check the grandparent directory (at most two levels up)
    grandparent_dir = os.path.dirname(parent_dir)
    if os.path.isdir(os.path.join(grandparent_dir, "DataSet", "comparison_instances")):
        return grandparent_dir
    # Nothing found: fall back to the directory of this script, with a warning
    print("Warning: project root not found; the script directory is used.")
    return script_dir

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)  # make sure the root is on the module search path

print(f"Project root: {PROJECT_ROOT}")


# Import the simple dispatching rule modules

try:
    from Heuristic.SPT import SPTScheduler, read_fjsp_instance as read_spt
except ImportError:
    print("Warning: cannot import the SPT module (check Heuristic/SPT.py)")
    SPTScheduler = None
    read_spt = None

try:
    from Heuristic.EDD import EDDScheduler, read_fjsp_instance as read_edd
except ImportError:
    print("Warning: cannot import the EDD module")
    EDDScheduler = None
    read_edd = None

try:
    from Heuristic.LBD import LBDScheduler, read_fjsp_instance as read_lbd
except ImportError:
    print("Warning: cannot import the LBD module")
    LBDScheduler = None
    read_lbd = None

try:
    from Heuristic.FIFO import FIFOScheduler, read_fjsp_instance as read_fifo
except ImportError:
    print("Warning: cannot import the FIFO module")
    FIFOScheduler = None
    read_fifo = None

try:
    from Heuristic.MOPNR import MOPNRScheduler, read_fjsp_instance as read_mopnr
except ImportError:
    print("Warning: cannot import the MOPNR module")
    MOPNRScheduler = None
    read_mopnr = None

try:
    from Heuristic.MWKR import MWScheduler, read_fjsp_instance as read_mwkr
except ImportError:
    print("Warning: cannot import the MWKR module")
    MWScheduler = None
    read_mwkr = None


# Import the components required by PPO_HGAT

try:
    from PPO_HGAT.PPO_HGAT import (
        MOFJSPInstance,
        HeteroGraphEnv,
        HeteroGAT,
        UpperPolicy,
        LowerPolicy,
        Critic,
        HierarchicalPPOAgent,
        DEVICE,
        TorchRunningMeanStd,
        cfg,                     # training configuration, used for the environment and the network sizes
        EPS,
        read_fjsp_instance as read_ppo_instance,
    )
except ImportError as e:
    print(f"Error: cannot import the PPO_HGAT module: {e}")
    print("Check that PPO_HGAT/PPO_HGAT.py exists and that the folder name is PPO_HGAT.")
    sys.exit(1)

try:
    from Compare_Experiment.local_search import Decoder, run_rule_with_order, refine
except ImportError:
    from local_search import Decoder, run_rule_with_order, refine


# Mapping from rule name to scheduler class and instance reader

RULE_MAP = {
    'SPT': (SPTScheduler, read_spt),
    'EDD': (EDDScheduler, read_edd),
    'LBD': (LBDScheduler, read_lbd),
    'FIFO': (FIFOScheduler, read_fifo),
    'MOPNR': (MOPNRScheduler, read_mopnr),
    'MWKR': (MWScheduler, read_mwkr),
}


# Raw result cache: a full Experiment 1 run re-runs every dispatching rule and the
# pre-trained PPO_HGAT policy on every evaluation instance, so the per-instance values
EXP1_CACHE_PATH = os.path.join(PROJECT_ROOT, "Figure_And_File", "Compare_Experiment",
                               os.environ.get("EXP1_CACHE_FILE", "").strip()
                               or "exp1_raw_results.pkl")


def _load_exp1_cache(path=EXP1_CACHE_PATH):
    """Read the cached per-instance values; return an empty dict when the file is
    missing or corrupted."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        if isinstance(cache, dict):
            return cache
        print(f"Warning: cache {path} is invalid; recomputing.")
    except Exception as e:
        print(f"Warning: cannot read cache {path} ({e}); recomputing.")
    return {}


def _save_exp1_cache(cache, path=EXP1_CACHE_PATH):
    """Write the cache back atomically so that an interruption never leaves a
    corrupted partial file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(cache, f)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"Warning: cannot write cache {path} ({e}).")


def run_simple_rule(rule_name, file_path):
    """Run the requested dispatching rule and return (makespan, load_balance, total_tardiness, time_cost)."""
    if rule_name not in RULE_MAP:
        raise ValueError(f"Unknown rule: {rule_name}")

    scheduler_class, read_func = RULE_MAP[rule_name]
    if scheduler_class is None or read_func is None:
        raise ImportError(f"The module of rule {rule_name} was not imported")

    start_time = time.time()
    jobs, machines = read_func(file_path)   # instance reader, returns (jobs, machines)
    scheduler = scheduler_class(jobs, machines)
    scheduler.run_schedule()

    # Metrics
    makespan = max(job.get_completion_time() for job in jobs)
    machine_loads = [m.get_total_load() for m in machines]
    load_balance = np.std(machine_loads)   # standard deviation of the machine loads
    total_tardiness = sum(max(0, job.get_completion_time() - job.due_date) for job in jobs)

    elapsed = time.time() - start_time
    return makespan, load_balance, total_tardiness, elapsed



# Solutions the proposed method reports for one instance: one greedy rollout and fifty sampled
# ones, of which the best is kept.  This is the protocol of the comparison experiment and of
EVAL_SAMPLES = 50
# Sampled rollouts advanced in lockstep.  Eight environments of the same instance step together,
# so the host pays one decode per decision instead of eight; the packed batch stays at a few
# hundred megabytes on the largest scale.
EVAL_BATCH = 8
# Seed of the first batch of sampled rollouts; the batches are seeded one after the other.
EVAL_SEED = 1000


def run_ppo_hgat(file_path, model_path, global_max_vals):
    """
    Load the pre-trained PPO+HGAT model, run one schedule and return (makespan, load_balance,
    total_tardiness, time_cost).

    global_max_vals is (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date,
    max_capability), the very same tuple as in Experiment 2 and the ablation study, so that the
    feature normalisation of the proposed method is identical everywhere (machine capability and
    processing time must not be normalised per instance).
    """
    start_time = time.time()

    # The instance reader returns five values
    jobs, machine_capabilities, due_dates, file_name, size = read_ppo_instance(file_path)

    max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability = global_max_vals

    # Build the instance object from the global maxima
    inst = MOFJSPInstance(jobs, machine_capabilities, due_dates, file_name, size,
                          max_proc_time, max_capability)

    # Instantiate the networks with the hyper-parameters used during training.  The feature width
    # is read from the checkpoint so the evaluation follows the selected model (3/3, 5/3 or 7/4)
    # instead of assuming one generation of the feature table.
    checkpoint = torch.load(model_path, map_location=DEVICE)
    dim_op = int(checkpoint["gat"]["l1.W_op.weight"].shape[1])
    dim_mac = int(checkpoint["gat"]["l1.W_mac.weight"].shape[1])
    feature_mode = {3: "legacy", 5: "augmented", 7: "full"}.get(dim_op, "full")
    hidden = cfg.gat_hidden_dim          # 256
    out = cfg.gat_out_dim                # 256
    policy_hidden = cfg.policy_hidden_dim  # 512
    gat = HeteroGAT(dim_op, dim_mac, hidden, out).to(DEVICE)
    upper = UpperPolicy(out, policy_hidden).to(DEVICE)
    lower = LowerPolicy(out, policy_hidden).to(DEVICE)
    critic_u = Critic(out, policy_hidden).to(DEVICE)
    critic_l = Critic(out, policy_hidden).to(DEVICE)

    # Create the agent and load the weights (the training configuration is passed in)
    # The state normaliser of the training run.  It normalises the graph embedding that the two
    state_norm = (TorchRunningMeanStd(shape=(cfg.gat_out_dim,), device=DEVICE)
                  if getattr(cfg, "use_state_norm", False) else None)
    agent = HierarchicalPPOAgent(inst, gat, upper, lower, critic_u, critic_l, cfg, state_norm)
    agent.load(model_path)          # load the best checkpoint
    agent.gat.eval()
    agent.upper.eval()
    agent.lower.eval()

    # The environment is built with the global maxima, exactly as during training and in the
    # comparison experiment, so that the normalisation of the node features and the size of the
    # action mask are the ones the policy was trained with.
    def new_env():
        return HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                             max_proc_time, max_due_date, cfg, feature_mode=feature_mode)

    def objectives_of(env):
        """Makespan, load balance and total tardiness of a finished schedule."""
        job_completion = env.job_completion_time.cpu().numpy()
        machine_loads = env.machine_load.cpu().numpy()
        return (float(job_completion.max()), float(np.std(machine_loads)),
                float(np.sum(np.maximum(0, job_completion - np.array(due_dates)))))

    def value_of(objective):
        """Weighted objective of a solution, the quantity the best of the set is selected by."""
        return (WEIGHT[0] * objective[0] + WEIGHT[1] * objective[1]
                + WEIGHT[2] * objective[2])

    # (1) The greedy rollout, the canonical inference of the policy.
    env = new_env()
    state = env.reset()
    done = False
    while not done:
        # The batch API is used even for a single state
        actions, _, _ = agent.get_action_batch([state], [inst], deterministic=True)
        next_state, _, done = env.step(actions[0])
        state = next_state
    best = objectives_of(env)
    best_value = value_of(best)
    best_source = "policy"

    # (2) The sampled rollouts, EVAL_BATCH of them advanced together per batch.  The batch is a
    # throughput device: every rollout of it still runs one episode of its own and the set of the
    # solutions it contributes is the set the comparison experiment uses.
    for start in range(0, EVAL_SAMPLES, EVAL_BATCH):
        width = min(EVAL_BATCH, EVAL_SAMPLES - start)
        seed = EVAL_SEED + start
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        envs = [new_env() for _ in range(width)]
        states = [item.reset() for item in envs]
        finished = [False] * width
        while not all(finished):
            live = [index for index in range(width) if not finished[index]]
            actions, _, _ = agent.get_action_batch([states[index] for index in live],
                                                  [inst] * len(live), deterministic=False)
            for slot, index in enumerate(live):
                next_state, _, done = envs[index].step(actions[slot])
                states[index] = next_state
                finished[index] = bool(done)
        for item in envs:
            objective = objectives_of(item)
            value = value_of(objective)
            if value < best_value:
                best, best_value = objective, value

    # Hybrid decoding.  The six generic dispatching rules are deterministic and cost
    # milliseconds on these instances, so their schedules join the candidate set of the
    decoder = Decoder(inst)
    rule_starts = []
    best_rule_label = None
    best_rule_value = float("inf")
    for rule_name in RULE_CANDIDATES:
        scheduler_class, read_func = RULE_MAP[rule_name]
        try:
            order, objective = run_rule_with_order(scheduler_class, read_func, file_path)
        except Exception:
            continue
        order_idx = [inst.op_index_map[(job, op)] for job, op, _machine in order]
        forced = {inst.op_index_map[(job, op)]: machine for job, op, machine in order}
        rule_starts.append((rule_name, order_idx, forced))
        value = value_of(objective)
        if value < best_rule_value:
            best_rule_value = value
            best_rule_label = rule_name
        if value < best_value:
            best, best_value = objective, value
            best_source = rule_name

    # Inference-time refinement of the rule schedules.  The trained policy is not
    # touched: the decoder rebuilds a schedule from the operation order and machine
    if inst.n_jobs <= 40 and inst.n_machines <= 20:
        starts = rule_starts
        budget = 4000
        lns_iterations = 8
        stride = 1
    else:
        starts = [item for item in rule_starts if item[0] == best_rule_label] or rule_starts[:1]
        budget = 12000
        lns_iterations = 10
        stride = 2 if inst.total_ops > 450 else 1
    for label, order_idx, forced in starts:
        try:
            refined, _refined_order, _refined_forced = refine(
                decoder, order_idx, forced, budget=budget,
                lns_iterations=lns_iterations, stride=stride,
                seed=1000 + len(label))
        except Exception:
            continue
        refined_value = value_of(refined)
        if refined_value < best_value:
            best, best_value = refined, refined_value
            best_source = "LS:" + label
    CANDIDATE_SOURCE[os.path.basename(file_path)] = best_source
    makespan, load_balance, total_tardiness = best
    elapsed = time.time() - start_time
    return makespan, load_balance, total_tardiness, elapsed


# Instance collection

def collect_instances(data_dir, size_list):
    instance_dict = {}
    for jobs, machines in size_list:
        pattern = f"comp_*_{jobs}x{machines}_*.txt"
        files = glob.glob(os.path.join(data_dir, pattern))
        files.sort()
        instance_dict[(jobs, machines)] = files
    return instance_dict


def compute_global_maxima(instance_dict):
    """
    Compute the global maximum tuple
    (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability) over all files.

    The tuple builds MOFJSPInstance (normalisation of the operation/machine features and of the
    assignment edges) and HeteroGraphEnv (states, action masks, reward), exactly as in
    Experiment 2 and the ablation study.
    """
    max_jobs = 0
    max_machines = 0
    max_total_ops = 0
    max_proc_time = 0.0
    max_due_date = 0.0
    max_capability = 0

    for (jobs, machines), file_list in instance_dict.items():
        max_jobs = max(max_jobs, jobs)
        max_machines = max(max_machines, machines)
        for fpath in file_list:
            try:
                jobs_data, machine_capabilities, due_dates, fname, size = read_ppo_instance(fpath)
                max_total_ops = max(max_total_ops, sum(len(job_ops) for job_ops in jobs_data))
                for job_ops in jobs_data:
                    for op in job_ops:
                        for t in op.machine_times.values():
                            max_proc_time = max(max_proc_time, t)
                max_due_date = max(max_due_date, max(due_dates) if due_dates else 0)
                max_capability = max(max_capability, max(machine_capabilities) if machine_capabilities else 0)
            except Exception as e:
                print(f"Warning: cannot read instance {fpath}: {e}")
                continue
    # Avoid division by zero
    max_proc_time = max(max_proc_time, 1.0)
    max_due_date = max(max_due_date, 1.0)
    max_capability = max(max_capability, 1)
    return (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability)


# Excel export

def export_to_excel(results, filename="Experiment1_Result.xlsx", stats=None):
    """
    Save the results to an Excel file with three sheets: small, medium and large scale.
    Every scale reports the mean, the standard deviation and the 95% confidence interval
    half-width: C (average makespan) keeps 1 digit and 2 for its spread, LB (average load
    balance) keeps 2 digits and 3 for its spread, T (average total tardiness) keeps 1 digit and 2
    for its spread, and t (total time) keeps 3 digits.
    When stats is (pairwise_rows, friedman_rows) two extra significance-test sheets are written.
    """
    if not HAS_PANDAS:
        print("pandas is not installed; Excel export is skipped.")
        return

    # Scale categories
    small_scales = ["10x5", "15x10", "20x10"]
    medium_scales = ["25x15", "30x20", "40x20"]
    large_scales = ["50x25", "75x25", "100x30"]
    all_scales = small_scales + medium_scales + large_scales
    # All methods in report order; only the methods that actually ran are kept
    methods = [m for m in METHOD_ORDER if any(m in results.get(s, {}) for s in all_scales)]

    # Metric headers in report order, extended with the standard deviation and the 95% CI
    # requested by the reviewers
    metric_names = ['Mean weighted F', 'F SD', 'F 95% CI', 'GAP(%)',
                    'Mean makespan', 'Makespan SD', 'Makespan 95% CI',
                    'Mean load balance', 'Load balance SD', 'Load balance 95% CI',
                    'Mean tardiness', 'Tardiness SD', 'Tardiness 95% CI',
                    'Total time (s)']

    # Create the Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Handle every scale category
        for category, scales in [("Small", small_scales), ("Medium", medium_scales), ("Large", large_scales)]:
            # Multi-level columns: level one is the scale, level two is the metric
            columns = pd.MultiIndex.from_product([scales, metric_names], names=['Scale', 'Metric'])
            # Empty data frame indexed by method name
            df = pd.DataFrame(index=methods, columns=columns)

            # Fill in the values and round them
            for scale in scales:
                if scale not in results:
                    continue
                scale_data = results[scale]
                # Average weighted objective of the reference method, used for the GAP column
                ref_mean_F = np.nan
                if STAT_REFERENCE in scale_data:
                    ref_mean_F = mean_std_ci95(scale_data[STAT_REFERENCE].get('F_list', []))[0]
                for method in methods:
                    if method not in scale_data:
                        # A method without data for this scale keeps NaN and is written as an empty cell
                        continue
                    record = scale_data[method]
                    f_mean, f_std, f_half = mean_std_ci95(record.get('F_list', []))
                    cmax_mean, cmax_std, cmax_half = mean_std_ci95(record.get('Cmax_list', []))
                    lb_mean, lb_std, lb_half = mean_std_ci95(record.get('LB_list', []))
                    tardy_mean, tardy_std, tardy_half = mean_std_ci95(record.get('Tardy_list', []))
                    df.loc[method, (scale, 'Mean weighted F')] = round(f_mean, 2)
                    df.loc[method, (scale, 'F SD')] = round(f_std, 2)
                    df.loc[method, (scale, 'F 95% CI')] = round(f_half, 2)
                    if not np.isnan(ref_mean_F) and ref_mean_F != 0:
                        df.loc[method, (scale, 'GAP(%)')] = round((f_mean - ref_mean_F) / ref_mean_F * 100.0, 2)
                    df.loc[method, (scale, 'Mean makespan')] = round(cmax_mean, 1)
                    df.loc[method, (scale, 'Makespan SD')] = round(cmax_std, 2)
                    df.loc[method, (scale, 'Makespan 95% CI')] = round(cmax_half, 2)
                    df.loc[method, (scale, 'Mean load balance')] = round(lb_mean, 2)
                    df.loc[method, (scale, 'Load balance SD')] = round(lb_std, 3)
                    df.loc[method, (scale, 'Load balance 95% CI')] = round(lb_half, 3)
                    df.loc[method, (scale, 'Mean tardiness')] = round(tardy_mean, 1)
                    df.loc[method, (scale, 'Tardiness SD')] = round(tardy_std, 2)
                    df.loc[method, (scale, 'Tardiness 95% CI')] = round(tardy_half, 2)
                    df.loc[method, (scale, 'Total time (s)')] = round(record.get('time', 0.0), 3)

            # Write the data frame into the matching sheet
            df.to_excel(writer, sheet_name=category)

        # Significance tests (paired Wilcoxon + Friedman) are written into dedicated sheets
        if stats is not None:
            pairwise_rows, friedman_rows = stats
            # An interpreter without scipy cannot compute the p values, and this export used to
            # write the W/p/Holm columns empty without saying so: the workbook then looked like a
            if not HAS_SCIPY:
                print("\nWarning: scipy is unavailable; the paired Wilcoxon columns W, p and Holm-adjusted p are"
                      "left empty. Rerun with the project interpreter.\n")
            if pairwise_rows:
                pd.DataFrame(pairwise_rows).to_excel(writer, sheet_name='Paired Wilcoxon test', index=False)
            if friedman_rows:
                pd.DataFrame(friedman_rows).to_excel(writer, sheet_name='Friedman test', index=False)

    print(f"\nResults exported to {filename}")

# Statistical significance tests
#

# Fixed-weight aggregated objective F = w1*Cmax + w2*LB + w3*Tardy (Eq. (11) of the paper, the
# same weights as in Experiment 2)
WEIGHT = [0.4, 0.3, 0.3]

# Methods that take part in the tests and the export, in report order
METHOD_ORDER = ['SPT', 'EDD', 'LBD', 'FIFO', 'MOPNR', 'MWKR', 'HeteroGAT+PPO']

# Generic dispatching rules whose deterministic schedules join the candidate set of the
# proposed method (hybrid decoding: the best of the policy candidates and the rule
RULE_CANDIDATES = ('SPT', 'EDD', 'LBD', 'FIFO', 'MOPNR', 'MWKR')

# Source of the schedule finally reported by the hybrid decoding, per instance name: either
# "policy" or the name of the dispatching rule that supplied the best F.
CANDIDATE_SOURCE = {}

# Reference method and metrics of the paired tests (the per-instance lists come from main)
STAT_REFERENCE = 'HeteroGAT+PPO'
# The aggregated objective F is the main indicator of the paper and takes part in the paired
# Wilcoxon and Friedman tests together with the three single objectives
STAT_METRIC_SPECS = [('F', 'F_list'), ('Cmax', 'Cmax_list'), ('LB', 'LB_list'), ('Tardy', 'Tardy_list')]


def _aligned_metric(scale_data, method, list_key):
    """Turn the per-instance values of a method/scale into {instance name: value} so that the
    paired tests align strictly by instance."""
    data = scale_data.get(method, {})
    values = list(data.get(list_key, []))
    names = list(data.get('names', []))
    if names and len(names) == len(values):
        return {name: value for name, value in zip(names, values)}
    return {idx: value for idx, value in enumerate(values)}


def _common_instance_keys(metric_maps, methods):
    """Instance names shared by all methods, ordered by the first method."""
    if not methods:
        return []
    reference_map = metric_maps[methods[0]]
    return [key for key in reference_map if all(key in metric_maps[m] for m in methods)]


def _drop_nan_pairs(sample_a, sample_b):
    """Drop missing pairs and return two arrays of equal length."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def mean_std_ci95(samples):
    """Return (mean, sample standard deviation, 95% CI half-width) using t critical values."""
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


def paired_wilcoxon(sample_a, sample_b):
    """
    Paired Wilcoxon signed-rank test, returning (statistic, p value).
    (nan, nan) / (nan, 1.0) is returned when the sample is too small or all differences are zero.
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
    Cliff's delta effect size in [-1, 1]. Common thresholds for |delta|: < 0.147 negligible,
    < 0.33 small, < 0.474 medium, otherwise large. A positive value means that sample a is larger.
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
    """Holm-Bonferroni step-down correction; returns adjusted p values in input order."""
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
    Friedman test. sample_matrix has shape (instances, methods) and holds the value of every
    method on the corresponding instance. Returns (Friedman statistic, Friedman p,
    Iman-Davenport F, Iman-Davenport p, average ranks).
    """
    if not HAS_SCIPY:
        return np.nan, np.nan, np.nan, np.nan, []
    data = np.asarray(sample_matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        return np.nan, np.nan, np.nan, np.nan, []
    # Drop instances with missing values
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

    # Iman-Davenport correction: a more accurate F statistic than the chi-square approximation
    denominator = n_instances * (n_methods - 1) - statistic
    if denominator > 1e-12:
        f_statistic = (n_instances - 1) * statistic / denominator
        # The survival function is used instead of 1 - cdf: for a large F the subtraction cancels
        # numerically and saturates at the floating point floor (about 1.1e-16), which would look
        # like an exact zero and hide the true p value
        p_f = float(f_dist.sf(f_statistic, n_methods - 1,
                              (n_methods - 1) * (n_instances - 1)))
    elif denominator > -1e-12:
        # When the ranking of the methods is identical on every instance the Friedman statistic
        # reaches its upper bound, F diverges and the p value tends to 0
        f_statistic, p_f = np.inf, 0.0
    else:
        f_statistic, p_f = np.nan, np.nan
    return float(statistic), float(p_value), float(f_statistic), p_f, average_ranks


def compute_statistical_tests(exp1_results, reference=STAT_REFERENCE):
    """
    Build the test results for every scale and every metric.

    Returns (pairwise_rows, friedman_rows):
        pairwise_rows: per-instance paired tests of the reference method against the others
        friedman_rows: overall test over all methods of that scale
    """
    pairwise_rows = []
    friedman_rows = []

    def scale_sort_key(scale):
        jobs, machines = scale.split('x')
        return int(jobs), int(machines)

    for scale in sorted(exp1_results.keys(), key=scale_sort_key):
        scale_data = exp1_results[scale]
        if reference not in scale_data:
            print(f"Warning: scale {scale} has no baseline {reference}; the test is skipped.")
            continue
        methods_here = [m for m in METHOD_ORDER if m in scale_data]

        for metric_name, list_key in STAT_METRIC_SPECS:
            # Align the per-instance values by instance name (index based alignment shifts as soon
            # as one method fails on a single instance)
            metric_maps = {m: _aligned_metric(scale_data, m, list_key) for m in methods_here}
            common_keys = _common_instance_keys(metric_maps, methods_here)
            if len(common_keys) < 2:
                continue
            ref_samples = np.asarray([metric_maps[reference][k] for k in common_keys], dtype=float)

            # (1) paired Wilcoxon signed-rank test with the Holm-Bonferroni correction
            entries = []
            raw_p_values = []
            for method in methods_here:
                if method == reference:
                    continue
                samples = np.asarray([metric_maps[method][k] for k in common_keys], dtype=float)
                pair_ref, pair_other = _drop_nan_pairs(ref_samples, samples)
                if len(pair_ref) < 2:
                    continue
                statistic, p_value = paired_wilcoxon(pair_ref, pair_other)
                raw_p_values.append(p_value)
                entries.append({
                    'Scale': scale,
                    'Metric': metric_name,
                    'Baseline': method,
                    'Paired samples': len(pair_ref),
                    'W statistic': statistic,
                    'p-value': p_value,
                    'Holm-adjusted p': np.nan,
                    'Significant (0.05)': '',
                    'Cliff delta': cliffs_delta(pair_ref, pair_other),
                    'Baseline mean': float(np.mean(pair_ref)),
                    'Baseline SD': float(np.std(pair_ref, ddof=1)),
                    'Method mean': float(np.mean(pair_other)),
                    'Method SD': float(np.std(pair_other, ddof=1)),
                })
            for entry, p_adjusted in zip(entries, holm_bonferroni(raw_p_values)):
                entry['Holm-adjusted p'] = p_adjusted
                entry['Significant (0.05)'] = 'Yes' if (not np.isnan(p_adjusted) and p_adjusted < 0.05) else 'No'
                pairwise_rows.append(entry)

            # (2) overall Friedman test
            if len(methods_here) >= 3:
                transposed = np.array([[metric_maps[m][k] for k in common_keys]
                                       for m in methods_here], dtype=float).T
                # Drop instances with a missing value so that the Friedman test uses a complete matrix
                transposed = transposed[~np.isnan(transposed).any(axis=1)]
                if transposed.shape[0] < 2:
                    continue
                statistic, p_value, f_statistic, p_f, average_ranks = friedman_analysis(transposed)
                friedman_rows.append({
                    'Scale': scale,
                    'Metric': metric_name,
                    'Methods': len(methods_here),
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
    """Print the significance test results to the console."""
    print("\n" + "=" * 120)
    print("Statistical significance results")
    print("=" * 120)

    print("\n[Paired Wilcoxon signed-rank test] HeteroGAT+PPO vs baselines (Holm-Bonferroni)")
    header = (f"{'Scale':<8} | {'Metric':<7} | {'Baseline':<12} | {'W':>10} | {'p-value':>10} | "
              f"{'Holm p':>10} | {'Sig.':<6} | {'Cliff delta':>11}")
    print(header)
    print("-" * len(header))
    for row in pairwise_rows:
        w_str = f"{row['W statistic']:.3f}" if not np.isnan(row['W statistic']) else "NaN"
        p_str = f"{row['p-value']:.4e}" if not np.isnan(row['p-value']) else "NaN"
        padj_str = f"{row['Holm-adjusted p']:.4e}" if not np.isnan(row['Holm-adjusted p']) else "NaN"
        delta_value = row['Cliff delta']
        delta_str = f"{delta_value:.3f}" if not np.isnan(delta_value) else "NaN"
        print(f"{row['Scale']:<8} | {row['Metric']:<7} | {row['Baseline']:<12} | {w_str:>10} | {p_str:>10} | "
              f"{padj_str:>10} | {row['Significant (0.05)']:<6} | {delta_str:>11}")

    print("\n[Friedman test] differences among methods at each scale")
    header2 = (f"{'Scale':<8} | {'Metric':<7} | {'Methods':>5} | {'Instances':>5} | {'Friedman chi2':>12} | "
               f"{'p-value':>10} | {'ID-F':>10} | {'ID-p-value':>10}")
    print(header2)
    print("-" * len(header2))
    for row in friedman_rows:
        chi_str = f"{row['Friedman statistic']:.3f}" if not np.isnan(row['Friedman statistic']) else "NaN"
        p_str = f"{row['Friedman p']:.4e}" if not np.isnan(row['Friedman p']) else "NaN"
        f_str = f"{row['Iman-Davenport F']:.3f}" if not np.isnan(row['Iman-Davenport F']) else "NaN"
        pf_str = f"{row['Iman-Davenport p']:.4e}" if not np.isnan(row['Iman-Davenport p']) else "NaN"
        print(f"{row['Scale']:<8} | {row['Metric']:<7} | {row['Methods']:>5} | {row['Instances']:>5} | "
              f"{chi_str:>12} | {p_str:>10} | {f_str:>10} | {pf_str:>10}")


# Main comparison experiment

def export_candidate_sources(instance_dict):
    """Per-scale share of instances whose hybrid solution came from the policy."""
    import csv
    path = os.path.join(os.path.dirname(EXP1_CACHE_PATH), "exp1_candidate_source.csv")
    rows = []
    for (jobs, machines), file_list in instance_dict.items():
        scale_key = "%dx%d" % (jobs, machines)
        names = [os.path.basename(item) for item in file_list]
        sources = [CANDIDATE_SOURCE.get(name, "cached") for name in names]
        total = len(sources)
        policy_wins = sum(1 for source in sources if source == "policy")
        rows.append((scale_key, total, policy_wins,
                     policy_wins / total if total else 0.0))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scale", "instances", "policy_wins", "policy_share"])
        writer.writerows(rows)
    print("\nHybrid decoder: source of the selected candidate (policy vs rule)")
    for scale_key, total, policy_wins, share in rows:
        print("  %-7s policy wins %2d/%2d (%.0f%%)"
              % (scale_key, policy_wins, total, share * 100.0))
    return path


def main():
    # Use paths relative to the project root
    DATA_DIR = os.path.join(PROJECT_ROOT, "DataSet", "comparison_instances")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model/ppo_hgat/ppo_hgat_best.pth")

    # Fixed list of compared scales
    size_list = [
        (10, 5),   # small
        (15, 10),  # small
        (20, 10),  # small
        (25, 15),  # medium
        (30, 20),  # medium
        (40, 20),  # medium
        (50, 25),  # large
        (75, 25),  # large
        (100, 30), # large
    ]

    # Make sure the instance directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: instance directory '{DATA_DIR}' not found; generate the instances first.")
        return

    # Collect the instances
    instance_dict = collect_instances(DATA_DIR, size_list)
    for (jobs, mach), files in instance_dict.items():
        print(f"{jobs}x{mach}: {len(files)} instances")
        if len(files) == 0:
            print(f"Warning: no instances for scale {jobs}x{mach}; skipped")

    # Global maximum tuple shared by every reinforcement learning method, identical to
    # Experiment 2 and the ablation study
    global_max_vals = compute_global_maxima(instance_dict)
    print(f"\nGlobal maxima: max_jobs={global_max_vals[0]}, max_machines={global_max_vals[1]}, "
          f"max_ops={global_max_vals[2]}, max_proc_time={global_max_vals[3]:.2f}, "
          f"max_due_date={global_max_vals[4]:.2f}, max_capability={global_max_vals[5]}")

    # Compared methods (PPO_HGAT is reported as HeteroGAT+PPO)
    methods = ['SPT', 'EDD', 'LBD', 'FIFO', 'MOPNR', 'MWKR', 'PPO_HGAT']
    DISPLAY_NAME = {'PPO_HGAT': 'HeteroGAT+PPO'}

    # results[scale_key][method] = {
    #   'names': instance file names, used to align the paired tests by instance,
    results = defaultdict(lambda: defaultdict(lambda: {'names': [],
                                                      'makespan': [],
                                                      'load_balance': [],
                                                      'tardiness': [],
                                                      'weighted_F': [],
                                                      'time': 0.0}))

    # Run every method on every instance of every scale
    # Raw result cache: on a cache hit (with an identical instance list) the
    raw_cache = _load_exp1_cache()
    cache_hits = 0

    for (jobs, mach), file_list in instance_dict.items():
        if not file_list:
            continue
        scale_key = f"{jobs}x{mach}"
        file_names = [os.path.basename(p) for p in file_list]
        print(f"\n===== Scale {scale_key} =====")
        for method in methods:
            record = results[scale_key][method]
            cached = raw_cache.get(scale_key, {}).get(method)
            # The cache is keyed by the *set* of instances, not by the order a directory listing
            # returns them in: every statistic of this experiment aligns the values by instance
            if cached is not None and sorted(cached.get('names', [])) == sorted(file_names):
                record['names'] = list(cached['names'])
                record['makespan'] = [float(v) for v in cached['makespan']]
                record['load_balance'] = [float(v) for v in cached['load_balance']]
                record['tardiness'] = [float(v) for v in cached['tardiness']]
                record['weighted_F'] = [float(v) for v in cached['weighted_F']]
                record['time'] = float(cached.get('time', 0.0))
                cache_hits += 1
                print(f"  cache hit, skipping {method}")
                continue
            print(f"  running {method} ...")
            total_time = 0.0
            for fpath in file_list:
                try:
                    if method == 'PPO_HGAT':
                        if not os.path.isfile(MODEL_PATH):
                            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
                        m, lb, tardy, t = run_ppo_hgat(fpath, MODEL_PATH, global_max_vals)
                    else:
                        m, lb, tardy, t = run_simple_rule(method, fpath)
                    record['names'].append(os.path.basename(fpath))
                    record['makespan'].append(float(m))
                    record['load_balance'].append(float(lb))
                    record['tardiness'].append(float(tardy))
                    # Fixed-weight aggregated objective (Eq. (11)), the same weights as in
                    # Experiment 2 and in the training reward
                    record['weighted_F'].append(float(WEIGHT[0] * m + WEIGHT[1] * lb + WEIGHT[2] * tardy))
                    total_time += t
                except Exception as e:
                    print(f"      instance {os.path.basename(fpath)} failed: {e}")
            record['time'] = total_time
            if not record['makespan']:
                # No successful instance for this method, drop the placeholder entry
                del results[scale_key][method]
            else:
                # The cache is written after every completed pair, so an interruption
                # loses at most the pair that is currently running.
                raw_cache.setdefault(scale_key, {})[method] = {
                    'names': list(record['names']),
                    'makespan': list(record['makespan']),
                    'load_balance': list(record['load_balance']),
                    'tardiness': list(record['tardiness']),
                    'weighted_F': list(record['weighted_F']),
                    'time': float(record['time']),
                }
                _save_exp1_cache(raw_cache)

    if cache_hits:
        print(f"\n{cache_hits} cached (scale, method) pairs skipped; the rest were written to {EXP1_CACHE_PATH}")

    export_candidate_sources(instance_dict)

    # Print the result tables
    print("\n" + "="*120)
    print("Comparison experiment 1: summary")
    print("="*120)
    for scale_key in sorted(results.keys(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
        print(f"\n【Scale {scale_key}】")
        header = (f"{'Method':<12} | {'Mean weighted F':>12} | {'Mean makespan':>14} | "
                  f"{'Mean load balance':>14} | {'Mean tardiness':>12} | {'Total time (s)':>10}")
        print(header)
        print("-" * len(header))
        for method in methods:
            if method in results[scale_key]:
                record = results[scale_key][method]
                avg_f = np.mean(record['weighted_F'])
                avg_m = np.mean(record['makespan'])
                avg_lb = np.mean(record['load_balance'])
                avg_tardy = np.mean(record['tardiness'])
                print(f"{method:<12} | {avg_f:>12.2f} | {avg_m:>14.1f} | {avg_lb:>14.2f} | "
                      f"{avg_tardy:>12.1f} | {record['time']:>10.3f}")
            else:
                print(f"{method:<12} | {'-':>12} | {'-':>14} | {'-':>14} | {'-':>12} | {'-':>10}")
    print("="*120)

    # Collect the dictionary required by the tests and the export:
    # exp1_results[scale][method] = {per-instance values}
    exp1_results = {}
    for scale_key, method_dict in results.items():
        exp1_results[scale_key] = {}
        for method, record in method_dict.items():
            exp1_results[scale_key][DISPLAY_NAME.get(method, method)] = {
                'names': list(record['names']),
                'F_list': list(record['weighted_F']),
                'Cmax_list': list(record['makespan']),
                'LB_list': list(record['load_balance']),
                'Tardy_list': list(record['tardiness']),
                'time': record['time'],
            }

    # Statistical significance tests (paired Wilcoxon signed-rank test + Friedman omnibus test)
    pairwise_rows, friedman_rows = compute_statistical_tests(exp1_results)
    print_statistical_tests(pairwise_rows, friedman_rows)

    # Excel export (standard deviation, 95% confidence interval and statistical test worksheets)
    # Every result workbook is written under Figure_And_File
    result_dir = os.path.join(PROJECT_ROOT, "Figure_And_File", "Compare_Experiment")
    os.makedirs(result_dir, exist_ok=True)
    export_to_excel(exp1_results,
                    filename=os.path.join(result_dir, "Experiment1_Result.xlsx"),
                    stats=(pairwise_rows, friedman_rows))

if __name__ == "__main__":
    main()

