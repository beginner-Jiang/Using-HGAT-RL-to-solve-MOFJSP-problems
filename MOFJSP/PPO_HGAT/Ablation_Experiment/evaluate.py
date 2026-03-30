"""
PPO+HGAT 消融实验评估
加载所有实验的模型，在对比数据集上测试，输出表格并保存为 CSV。
支持断点续跑：每个实验结果保存在 Check_Point 目录，最终汇总为 CSV 表格。
强制要求所有实验的模型文件必须存在，否则报错退出。
A5 模型若加载失败，自动从预训练路径重新复制。
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

# 尝试导入 pandas，若失败则给出警告但继续运行
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告：pandas 未安装，无法导出 CSV 文件，将继续打印结果。")

from common import (
    DEVICE, EPS, W1, W2, W3, SEED,
    COMPARISON_DIR, EXPERIMENTS,
    read_fjsp_instance, MOFJSPInstance,
    HeteroGraphEnv, HomogeneousGAT, HeteroGAT, MLPEncoder,
    UpperPolicy, LowerPolicy, Critic, JointPolicy,
    TorchRunningMeanStd, PPOAgent, cfg, PROJECT_ROOT, A5_MODEL_PATH
)

# 断点和最终结果保存路径
CHECK_POINT_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "Ablation_Experiment", "Check_Point")
FINAL_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "Ablation_Experiment", "Final")
os.makedirs(CHECK_POINT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# 辅助函数：从 checkpoint 中提取 encoder state_dict
def extract_encoder_state_dict(checkpoint, exp_config):
    """
    尝试从 checkpoint 中提取 encoder 的 state_dict，支持多种可能键名。
    如果失败，抛出 RuntimeError。
    """
    possible_keys = ['encoder', 'gat', 'model', 'state_dict']
    if isinstance(checkpoint, dict):
        for key in possible_keys:
            if key in checkpoint:
                return checkpoint[key]
        # 如果 checkpoint 本身就是 state_dict（例如直接保存的模型参数）
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError(f"无法从 checkpoint 中提取 encoder state_dict，可用键: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")

# 评估函数
def evaluate_experiment(exp_config, instance_objs, global_max_vals, num_runs=50):
    """
    instance_objs: list of (file_path, MOFJSPInstance)
    global_max_vals: (max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, max_capability)
    num_runs: 每个实例运行的次数
    返回: (exp_name, results_dict)  results_dict: {filename: {'objectives': list, 'time': total_time}}
    """
    exp_name = exp_config['name']
    checkpoint_path = os.path.join(CHECK_POINT_DIR, f"{exp_name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

    # 如果断点文件存在，直接加载并返回
    if os.path.exists(checkpoint_path):
        print(f"\n[断点] 实验 {exp_name} 已有结果，直接加载：{checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            results = pickle.load(f)
        return exp_name, results

    print(f"\n评估实验：{exp_name}")

    model_dir = exp_config['model_dir']
    model_path = os.path.join(model_dir, "best.pth")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        # 如果是 A5 且预训练路径存在，则尝试复制
        if exp_name == 'A5 (异构+带边特征+GAT+分层PPO) 完整模型' and os.path.exists(A5_MODEL_PATH):
            print(f"警告：A5 模型文件 {model_path} 不存在，将从预训练路径 {A5_MODEL_PATH} 复制...")
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy(A5_MODEL_PATH, model_path)
            print(f"复制完成。")
        else:
            raise FileNotFoundError(f"模型文件不存在：{model_path}，请先训练实验 {exp_name}。")

    # 加载 checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"加载模型文件 {model_path} 时出错: {e}")

    # 提取 encoder state_dict
    try:
        encoder_state_dict = extract_encoder_state_dict(checkpoint, exp_config)
    except RuntimeError as e:
        raise RuntimeError(f"从 checkpoint 提取 encoder 失败: {e}")

    # 根据 encoder 类型推断维度
    original_hidden = cfg.gat_hidden_dim
    original_out = cfg.gat_out_dim
    original_policy_hidden = cfg.policy_hidden_dim
    try:
        if exp_config['use_homogeneous']:
            # 同构 GAT，取第一层线性层的输出维度
            found = False
            for key in encoder_state_dict.keys():
                if 'W.weight' in key and 'layers.0' in key:
                    out_dim = encoder_state_dict[key].shape[0]
                    found = True
                    break
            if not found:
                raise RuntimeError("无法从 checkpoint 中推断出输出维度")
            hidden_dim = out_dim  # 假设 hidden_dim = out_dim
            # 检查是否有 LayerNorm 键
            has_layer_norm = any('ln.weight' in key for key in encoder_state_dict.keys())
        elif exp_config['use_mlp_encoder']:
            # MLP 编码器，从 op_net 的第一个线性层推断
            out_dim = encoder_state_dict['op_net.0.weight'].shape[0]
            hidden_dim = out_dim
            has_layer_norm = False  # MLP 没有 LayerNorm
        else:
            # 异构 GAT，从 l1.W_op.weight 推断
            out_dim = encoder_state_dict['l1.W_op.weight'].shape[0]
            hidden_dim = out_dim
            # 检查是否有 LayerNorm 键（注意异构 GAT 的 LayerNorm 键为 l1.ln_op.weight 等）
            has_layer_norm = any('ln' in key for key in encoder_state_dict.keys())

        # 临时修改 cfg 的维度
        cfg.gat_hidden_dim = hidden_dim
        cfg.gat_out_dim = out_dim
        print(f"根据 checkpoint 调整模型维度: hidden_dim={hidden_dim}, out_dim={out_dim}, use_layer_norm={has_layer_norm}")

        # 推断 policy_hidden_dim
        if exp_config['use_hierarchical']:
            # 检查 upper 的维度
            if 'upper' in checkpoint and checkpoint['upper'] is not None:
                upper_state = checkpoint['upper']
                # net.0.weight 形状: (hidden_dim, input_dim)
                input_dim_upper = upper_state['net.0.weight'].shape[1]
                hidden_dim_policy = upper_state['net.0.weight'].shape[0]
                # 验证 input_dim 是否等于 cfg.gat_out_dim * 2
                expected_input = cfg.gat_out_dim * 2
                if input_dim_upper != expected_input:
                    print(f"警告: upper 输入维度 {input_dim_upper} 与预期 {expected_input} 不符，将使用推断的隐藏维度 {hidden_dim_policy}")
            else:
                hidden_dim_policy = cfg.policy_hidden_dim  # 默认
        else:
            # 联合策略，从 joint_policy.net.0.weight 推断
            if 'joint_policy' in checkpoint and checkpoint['joint_policy'] is not None:
                joint_state = checkpoint['joint_policy']
                hidden_dim_policy = joint_state['net.0.weight'].shape[0]
            else:
                hidden_dim_policy = cfg.policy_hidden_dim

        cfg.policy_hidden_dim = hidden_dim_policy
        print(f"根据 checkpoint 调整 policy_hidden_dim 为 {hidden_dim_policy}")

    except Exception as e:
        print(f"从 checkpoint 推断维度失败: {e}，将使用当前 cfg 设置 (hidden_dim={cfg.gat_hidden_dim}, out_dim={cfg.gat_out_dim}, policy_hidden_dim={cfg.policy_hidden_dim})")
        has_layer_norm = True  # 默认假设有 LayerNorm

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
            use_layer_norm=has_layer_norm  # 根据 checkpoint 决定是否使用 LayerNorm
        ).to(DEVICE)
    elif exp_config['use_mlp_encoder']:
        encoder = MLPEncoder(dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim).to(DEVICE)
    else:
        encoder = HeteroGAT(
            dim_op, dim_mac, cfg.gat_hidden_dim, cfg.gat_out_dim,
            use_edge_feat=exp_config['use_edge_feat'],
            use_layer_norm=has_layer_norm  # 传递 has_layer_norm
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
        raise RuntimeError(f"PPOAgent 加载模型失败：{model_path}")

    class DummyConfig:
        def __init__(self):
            self.reward_scaling = cfg.reward_scaling
            self.reward_clip = cfg.reward_clip
            self.use_disjunctive_edges = False
    eval_cfg = DummyConfig()

    results = {}
    # 使用 tqdm 显示实例进度
    for fpath, inst in tqdm(instance_objs, desc=f"实例评估", leave=False):
        env = HeteroGraphEnv(inst, max_jobs, max_total_ops, max_machines,
                             max_proc_time, max_due_date, eval_cfg)
        agent.inst = inst
        objectives_list = []
        total_time = 0.0
        # 每个实例运行 num_runs 次
        for run in tqdm(range(num_runs), desc=f"运行 {os.path.basename(fpath)}", leave=False):
            seed = SEED + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            state = env.reset()
            done = False
            start_time = time.time()
            while not done:
                # 使用 deterministic=False 保持随机性
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

    # 保存断点文件
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"实验结果已保存至断点文件：{checkpoint_path}")

    # 恢复 cfg 原始值
    cfg.gat_hidden_dim = original_hidden
    cfg.gat_out_dim = original_out
    cfg.policy_hidden_dim = original_policy_hidden
    return exp_name, results


# 多目标指标计算
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


# 导出最终汇总 CSV
def export_final_csv(dim_metrics, a5_avg_F, file_to_size, dim_to_files):
    """
    生成三张 CSV 表（小、中、大规模），保存在 FINAL_DIR 目录。
    每张表行索引为实验名，列索引为扁平化的"维度_指标"。
    数值按指定精度格式化。
    """
    if not HAS_PANDAS:
        print("pandas 未安装，无法导出 CSV。")
        return

    # 指标名称（与表格一致）
    metric_names = ['平均F', 'GAP(%)', 'HV', 'GD', 'SP', '总耗时(s)']
    # 实验名称列表（按原顺序）
    exp_names = [e['name'] for e in EXPERIMENTS]

    # 按规模分组维度
    size_to_dims = {'small': [], 'medium': [], 'large': []}
    for dim in dim_metrics.keys():
        if dim in dim_to_files and dim_to_files[dim]:
            sample_fname = os.path.basename(sorted(dim_to_files[dim])[0])
            size = file_to_size.get(sample_fname, 'unknown')
            if size in size_to_dims:
                size_to_dims[size].append(dim)

    # 生成每个规模的文件
    for size, dims in size_to_dims.items():
        if not dims:
            continue
        dims = sorted(dims)
        # 创建扁平列名，如 "10x5_平均F"
        flat_columns = [f"{dim}_{metric}" for dim in dims for metric in metric_names]
        df = pd.DataFrame(index=exp_names, columns=flat_columns)

        for dim in dims:
            base = a5_avg_F.get(dim, None)
            for exp_name in exp_names:
                if exp_name not in dim_metrics[dim]:
                    continue
                m = dim_metrics[dim][exp_name]
                gap = (m['avg_F'] - base) / base * 100 if base is not None and base != 0 else 0.0

                # 格式化数值
                df.loc[exp_name, f"{dim}_平均F"] = f"{m['avg_F']:.2f}"
                df.loc[exp_name, f"{dim}_GAP(%)"] = f"{gap:.2f}"
                df.loc[exp_name, f"{dim}_HV"] = f"{m['HV']:.2e}"
                df.loc[exp_name, f"{dim}_GD"] = f"{m['GD']:.4f}"
                df.loc[exp_name, f"{dim}_SP"] = f"{m['SP']:.4f}"
                df.loc[exp_name, f"{dim}_总耗时(s)"] = f"{m['time']:.3f}"

        # 保存为 CSV
        csv_path = os.path.join(FINAL_DIR, f"{size.capitalize()}.csv")
        df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"已导出 {size} 规模结果至：{csv_path}")


# 主函数
def main():
    # 检查对比数据集目录
    if not os.path.exists(COMPARISON_DIR):
        print(f"错误：对比数据集目录 {COMPARISON_DIR} 不存在，请先运行 Generate_DataSet2.py 生成数据。")
        sys.exit(1)

    # 收集所有测试实例并按维度分组
    pattern = "comp_*.txt"
    all_files = glob.glob(os.path.join(COMPARISON_DIR, pattern))
    if not all_files:
        print(f"错误：在 {COMPARISON_DIR} 中未找到任何实例文件。")
        sys.exit(1)

    # 按规模维度分组（例如 '10x5', '15x10' 等）
    dim_to_files = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        if len(parts) >= 3:
            dim = parts[2]  # 例如 '10x5'
            if dim not in dim_to_files:
                dim_to_files[dim] = []
            dim_to_files[dim].append(fpath)

    # 只保留每个维度有10个实例的维度
    valid_dims = [dim for dim, flist in dim_to_files.items() if len(flist) >= 10]
    instance_files = []
    for dim in sorted(valid_dims):
        flist = sorted(dim_to_files[dim])[:10]
        instance_files.extend(flist)

    print(f"共收集 {len(instance_files)} 个测试实例（每个维度10个）。")

    # 读取所有实例并构建MOFJSPInstance对象，同时计算全局最大值
    all_instances = []  # (file_path, MOFJSPInstance)
    file_to_size = {}   # 文件名 -> 规模
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
    print(f"测试实例全局最大值：max_jobs={max_jobs}, max_machines={max_machines}, max_ops={max_total_ops}, "
          f"max_proc={max_proc_time:.2f}, max_due={max_due_date:.2f}, max_cap={max_capability}")

    # 在评估前检查所有实验的模型文件是否存在（A5 会尝试自动复制）
    for exp in EXPERIMENTS:
        model_path = os.path.join(exp['model_dir'], "best.pth")
        if exp['name'] == 'A5 (异构+带边特征+GAT+分层PPO) 完整模型' and not os.path.exists(model_path):
            if os.path.exists(A5_MODEL_PATH):
                print(f"A5 模型不存在，将从预训练路径复制...")
                os.makedirs(exp['model_dir'], exist_ok=True)
                shutil.copy(A5_MODEL_PATH, model_path)
            else:
                raise FileNotFoundError(f"A5 预训练模型 {A5_MODEL_PATH} 也不存在，无法评估。")
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"实验 {exp['name']} 的模型文件 {model_path} 不存在，请先运行 train.py 训练。")

    # 评估所有实验（支持断点续跑）
    all_results = {}  # exp_name -> {filename: {'objectives': list, 'time': total_time}}
    for exp in EXPERIMENTS:
        # 将 num_runs 从 1 改为 50，以确保 SP 等指标有足够样本
        name, res = evaluate_experiment(exp, all_instances, global_max_vals, num_runs=50)
        if res is not None:
            all_results[name] = res

    if not all_results:
        print("没有成功评估的实验。")
        return

    # 确定全局参考点（所有解的最大值）
    all_objectives = []
    for exp_res in all_results.values():
        for fname, data in exp_res.items():
            all_objectives.extend(data['objectives'])
    all_objectives = np.array(all_objectives)
    ref_point = np.max(all_objectives, axis=0) * 1.1 if len(all_objectives) > 0 else np.array([1000, 1000, 1000])

    # 为每个实例计算真实前沿（所有实验的解合并）
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

    # 按维度分组计算各实验的平均指标
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

    # 获取A5在各维度的avg_F作为基准
    a5_avg_F = {}
    for dim, metrics in dim_metrics.items():
        for exp_name, m in metrics.items():
            if 'A5' in exp_name:
                a5_avg_F[dim] = m['avg_F']
                break

    # 打印表格
    print("\n" + "="*120)
    print("消融实验对比结果（各维度平均）")
    print("="*120)

    header = f"{'维度':<8} {'实验':<40} {'平均F':>10} {'GAP(%)':>7} {'HV':>12} {'GD':>12} {'SP':>12} {'总耗时(s)':>10}"
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

    # 导出最终 CSV 汇总表
    if HAS_PANDAS:
        export_final_csv(dim_metrics, a5_avg_F, file_to_size, dim_to_files)
    else:
        print("pandas 未安装，跳过 CSV 导出。")


if __name__ == "__main__":
    main()