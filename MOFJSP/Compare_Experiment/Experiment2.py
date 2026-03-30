"""
对比实验2：元启发式与深度强化学习方法对比
对比的规模（jobs x machines）由固定列表定义，每个规模应有10个实例。
"""

import os
import time
import glob
import sys
import numpy as np
import torch
import random
from collections import defaultdict


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告：pandas未安装，无法导出Excel文件，将继续打印结果。")


# 自动定位项目根目录

def find_project_root():
    """从脚本所在目录开始向上查找，直到找到包含 comparison_instances 的目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 检查脚本所在目录
    if os.path.isdir(os.path.join(script_dir, "comparison_instances")):
        return script_dir
    # 检查父目录
    parent_dir = os.path.dirname(script_dir)
    if os.path.isdir(os.path.join(parent_dir, "comparison_instances")):
        return parent_dir
    # 再检查父目录的父目录（最多向上两层）
    grandparent_dir = os.path.dirname(parent_dir)
    if os.path.isdir(os.path.join(grandparent_dir, "comparison_instances")):
        return grandparent_dir
    # 如果都没找到，返回脚本所在目录（并给出警告）
    print("警告：无法自动定位项目根目录，将使用脚本所在目录。")
    return script_dir

PROJECT_ROOT = find_project_root()
sys.path.insert(0, PROJECT_ROOT)  # 确保根目录在模块搜索路径中

print(f"项目根目录设置为: {PROJECT_ROOT}")


# 通用数据读取函数（仅用于计算全局最大值）

try:
    from PPO_HGAT.PPO_HGAT import read_fjsp_instance as read_ppo_instance
except ImportError:
    print("错误：无法从 PPO_HGAT.PPO_HGAT 导入 read_fjsp_instance")
    sys.exit(1)


# MOEA/D 模块

try:
    from Meta_Heuristic.MOEA_D import (
        MOEADScheduler,
        read_fjsp_instance as read_moead,
        Job as MOEAJob,
        Operation as MOEAOp
    )
except ImportError:
    print("警告：无法导入 MOEA/D 模块，将跳过该方法。请检查文件夹名称是否为 Meta_Heuristic（下划线）。")
    MOEADScheduler = None
    read_moead = None


# NSGA-II 模块

try:
    from Meta_Heuristic.NSGA_II import (
        NSGA2Scheduler,
        read_fjsp_instance as read_nsga2,
        Job as NSGA2Job,
        Operation as NSGA2Op
    )
except ImportError:
    print("警告：无法导入 NSGA-II 模块，将跳过该方法。")
    NSGA2Scheduler = None
    read_nsga2 = None


# DQN+MLP 模块

try:
    from Deep_Learning.DQN_MLP import (
        read_fjsp_instance_structured as read_dqn_instance,
        JobShopInstance as DQNJobInstance,
        Config as DQNConfig,
        MOFJSP_Env as DQNEnv,
        DQN_Agent
    )
except ImportError as e:
    print(f"警告：无法导入 DQN+MLP 模块 ({e})，将跳过该方法。")
    read_dqn_instance = None
    DQNJobInstance = None
    DQNConfig = None
    DQNEnv = None
    DQN_Agent = None


# PPO+MLP 模块

try:
    from Deep_Learning.PPO_MLP import (
        read_fjsp_instance as read_ppo_mlp_instance,
        JobShopInstance as PPOJobInstance,
        Config as PPOConfig,
        MOFJSP_Env as PPOEnv,
        PolicyNetwork as PPOPolicy,
        ValueNetwork as PPOValue,
        PPOKLTrainer as PPOAgent,
        DEVICE as PPO_DEVICE
    )
    import __main__
    __main__.Config = PPOConfig
except ImportError as e:
    print(f"警告：无法导入 PPO+MLP 模块 ({e})，将跳过该方法。")
    read_ppo_mlp_instance = None
    PPOJobInstance = None
    PPOConfig = None
    PPOEnv = None
    PPOPolicy = None
    PPOValue = None
    PPOAgent = None
    PPO_DEVICE = None


# PPO+GAT 模块

try:
    from Deep_Learning.PPO_GAT import (
        read_fjsp_instance as read_ppo_gat_instance,
        MOFJSPInstance as GATInstance,
        Config as GATConfig,
        HomogeneousDisjunctiveGraphEnv as GATEnv,
        HomogeneousGAT,
        PolicyNetwork as GATPolicy,
        ValueNetwork as GATValue,
        DEVICE as GAT_DEVICE
    )
except ImportError as e:
    print(f"警告：无法导入 PPO+GAT 模块 ({e})，将跳过该方法。")
    read_ppo_gat_instance = None
    GATInstance = None
    GATConfig = None
    GATEnv = None
    HomogeneousGAT = None
    GATPolicy = None
    GATValue = None
    GAT_DEVICE = None


# PPO+HGAT 模块

try:
    from PPO_HGAT.PPO_HGAT import (
        HeteroGAT,
        UpperPolicy,
        LowerPolicy,
        Critic,
        HierarchicalPPOAgent,
        MOFJSPInstance as HGATInstance,
        HeteroGraphEnv as HGATEnv,
        cfg as HGATConfig,
        DEVICE as HGAT_DEVICE
    )
except ImportError as e:
    print(f"警告：无法导入 PPO+HGAT 模块 ({e})，将跳过该方法。")
    HeteroGAT = None
    UpperPolicy = None
    LowerPolicy = None
    Critic = None
    HierarchicalPPOAgent = None
    HGATInstance = None
    HGATEnv = None
    HGATConfig = None
    HGAT_DEVICE = None


# 配置参数


DATA_DIR = os.path.join(PROJECT_ROOT, "comparison_instances")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# 各方法模型路径（已经训练好）
DQN_MODEL_PATH = os.path.join(MODEL_DIR, "dqn/dqn_model.pth")
PPO_MLP_MODEL_PATH = os.path.join(MODEL_DIR, "ppo/ppo_model.pth")
PPO_GAT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_gat/ppo_gat_best.pth")
PPO_HGAT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hgat/ppo_hgat_best.pth")

# 固定规模列表
SIZE_LIST = [
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

# 多目标权重
W = [0.4, 0.3, 0.3]   # [Cmax, LB, Tardy]

# 强化学习方法运行时生成解的数量（与进化算法种群大小一致）
NUM_RUNS_RL = 50

# 随机种子列表（用于多次运行）
RL_SEEDS = [42 + i for i in range(NUM_RUNS_RL)]


# 辅助函数：计算加权目标值


def weighted_objective(obj):
    """obj = [makespan, load_balance, total_tardiness]"""
    return W[0] * obj[0] + W[1] * obj[1] + W[2] * obj[2]


# 辅助函数：多目标指标计算


def compute_hypervolume(points, ref_point):
    """
    计算三维超体积的近似值（蒙特卡洛方法）
    """
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError("Only 3 objectives supported for hypervolume.")
    n_samples = 10000
    samples = np.random.uniform(0, 1, (n_samples, 3)) * (ref_point - 0) + 0
    dominated = np.zeros(n_samples, dtype=bool)
    for p in points:
        dominated |= np.all(samples >= p, axis=1)
    hv = dominated.mean() * np.prod(ref_point)
    return hv

def compute_gd(pf, reference_pf):
    """
    世代距离 GD：pf到reference_pf的平均欧氏距离
    """
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
    """
    间距 SP：pf中相邻点之间距离的标准差（用于衡量均匀性）
    若解集少于2个点，返回NaN。
    """
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
    """
    快速非支配排序，返回第一前沿（非支配解）的索引
    """
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
            # 检查i是否支配j
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
    # 返回第一前沿的索引
    return fronts[0] if fronts and fronts[0] else []


# 计算全局最大值（用于环境初始化）

def compute_global_maxima(instance_dict):
    """遍历所有实例文件，计算全局最大作业数、机器数、加工时间、交货期"""
    max_jobs = 0
    max_machines = 0
    max_proc_time = 0.0
    max_due_date = 0.0

    for (jobs, mach), file_list in instance_dict.items():
        max_jobs = max(max_jobs, jobs)
        max_machines = max(max_machines, mach)
        for fpath in file_list:
            try:
                jobs_data, machine_capabilities, due_dates, fname, size = read_ppo_instance(fpath)
                for job_ops in jobs_data:
                    for op in job_ops:
                        for t in op.machine_times.values():
                            max_proc_time = max(max_proc_time, t)
                max_due_date = max(max_due_date, max(due_dates) if due_dates else 0)
            except Exception as e:
                print(f"警告：读取实例 {fpath} 失败：{e}")
                continue
    # 避免除零
    max_proc_time = max(max_proc_time, 1.0)
    max_due_date = max(max_due_date, 1.0)
    return max_jobs, max_machines, max_proc_time, max_due_date


# 运行各方法的函数


def run_moead(file_path):
    """运行MOEA/D，返回Pareto前沿的目标值列表"""
    if MOEADScheduler is None or read_moead is None:
        raise ImportError("MOEA/D 模块未正确导入")
    jobs, machines = read_moead(file_path)
    scheduler = MOEADScheduler(
        jobs=jobs,
        machines=machines,
        population_size=50,
        max_generations=100,
        crossover_rate=0.9,
        mutation_rate=0.1,
        neighbor_size=10,
        decomposition_method='ws'
    )
    scheduler.run()
    objectives = [ind.objectives for ind in scheduler.ep]
    return objectives

def run_nsga2(file_path):
    """运行NSGA-II，返回Pareto前沿的目标值列表"""
    if NSGA2Scheduler is None or read_nsga2 is None:
        raise ImportError("NSGA-II 模块未正确导入")
    jobs, machines = read_nsga2(file_path)
    scheduler = NSGA2Scheduler(
        jobs=jobs,
        machines=machines,
        population_size=50,
        max_generations=100,
        crossover_rate=0.9,
        mutation_rate=0.1
    )
    scheduler.run()
    objectives = [ind.objectives for ind in scheduler.best_solutions]
    return objectives

def run_dqn_mlp(file_path, model_path, seed_list, global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date):
    """
    运行MLP+DQN模型，使用多个随机种子生成一组解。
    为生成多样化近似Pareto解，采用softmax采样（温度=1.0）代替贪婪策略。
    """
    if DQN_Agent is None or DQNEnv is None:
        raise ImportError("DQN+MLP 模块未正确导入")

    # 使用 DQN 专用的读取函数读取实例
    data = read_dqn_instance(file_path)
    # 构建实例包装
    file_name = os.path.basename(file_path)
    size = data.size if hasattr(data, 'size') else "unknown"
    inst = DQNJobInstance(data, global_max_jobs, global_max_machines, file_name, size)

    # 创建配置（使用全局最大值）
    cfg = DQNConfig()
    cfg.max_jobs = global_max_jobs
    cfg.max_machines = global_max_machines
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model_path = model_path  # 设置模型路径，以便 __init__ 自动加载

    # 创建 agent（__init__ 会自动调用 _load_model 加载模型）
    agent = DQN_Agent(cfg)
    agent.policy_net.eval()
    print(f"已加载 DQN 模型: {model_path}")

    objectives_list = []
    temperature = 1.0  # softmax温度，控制随机性
    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = DQNEnv(cfg, inst)
        state = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            state_t = torch.from_numpy(state).float().to(cfg.device).unsqueeze(0)
            mask_t = torch.from_numpy(mask).float().to(cfg.device)
            with torch.no_grad():
                q_values = agent.policy_net(state_t).squeeze(0)
                q_values_masked = q_values.clone()
                q_values_masked[mask_t == 0] = -float('inf')
                # 使用带温度的softmax采样
                probs = torch.softmax(q_values_masked / temperature, dim=-1)
                if torch.isnan(probs).any():
                    # 处理异常：如果所有概率为NaN，则均匀选择有效动作
                    valid = torch.nonzero(mask_t).squeeze(-1)
                    if len(valid) == 0:
                        action = 0
                    else:
                        action = valid[torch.randint(len(valid), (1,))].item()
                else:
                    action = torch.multinomial(probs, 1).item()
            next_state, _, done, _ = env.step(action)
            state = next_state

        # 从环境提取指标
        cmax = max(env.machine_avail_time[:inst.num_machines])
        loads = env.machine_load[:inst.num_machines]
        avg_load = np.mean(loads)
        lb = np.sqrt(np.mean((loads - avg_load)**2))
        tardy = sum(max(0, env.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))
        objectives_list.append([float(cmax), float(lb), float(tardy)])

    return objectives_list

def run_ppo_mlp(file_path, model_path, seed_list, global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date):
    """运行MLP+PPO模型，返回多个解的目标值"""
    if PPOAgent is None or PPOEnv is None:
        raise ImportError("PPO+MLP 模块未正确导入")

    # 使用 PPO_MLP 专用的读取函数
    jobs, machine_ids, due_dates = read_ppo_mlp_instance(file_path)
    file_name = os.path.basename(file_path)
    size = "unknown"
    inst = PPOJobInstance(jobs, machine_ids, due_dates, global_max_jobs, global_max_machines, file_name, size)

    cfg = PPOConfig()
    cfg.max_jobs = global_max_jobs
    cfg.max_machines = global_max_machines
    state_dim = 2 * cfg.max_machines + 3 * cfg.max_jobs
    action_dim = cfg.max_jobs * cfg.max_machines

    policy = PPOPolicy(state_dim, action_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(PPO_DEVICE)
    value = PPOValue(state_dim, cfg.hidden_dim, cfg.num_hidden_layers).to(PPO_DEVICE)

    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在，PPO+MLP运行可能失败。")
        return []
    try:
        checkpoint = torch.load(model_path, map_location=PPO_DEVICE, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        value.load_state_dict(checkpoint['value_state_dict'])
        policy.eval()
        value.eval()
        print(f"已加载 PPO+MLP 模型: {model_path}")
    except Exception as e:
        print(f"警告：PPO+MLP模型加载失败：{e}")
        return []

    objectives_list = []
    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = PPOEnv(cfg, inst)
        state = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            state_t = torch.from_numpy(state).to(PPO_DEVICE).float().unsqueeze(0)
            mask_t = torch.from_numpy(mask).to(PPO_DEVICE).bool().unsqueeze(0)
            with torch.no_grad():
                probs, _ = policy(state_t, mask_t)
                dist = torch.distributions.Categorical(probs=probs)
                action_id = dist.sample().item()
            next_state, _, done, _ = env.step(action_id)
            state = next_state

        # 从环境提取指标
        cmax = max(env.machine_avail_time[:inst.num_machines])
        loads = env.machine_load[:inst.num_machines]
        avg_load = np.mean(loads)
        lb = np.sqrt(np.mean((loads - avg_load)**2))
        tardy = sum(max(0, env.job_avail_time[j] - inst.due_dates[j]) for j in range(inst.num_jobs))
        objectives_list.append([float(cmax), float(lb), float(tardy)])

    return objectives_list

def run_ppo_gat(file_path, model_path, seed_list, global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date):
    """运行GAT+PPO模型，返回多个解的目标值"""
    if HomogeneousGAT is None or GATEnv is None:
        raise ImportError("PPO+GAT 模块未正确导入")

    # 使用 PPO_GAT 专用的读取函数
    jobs, machine_ids, due_dates = read_ppo_gat_instance(file_path)
    file_name = os.path.basename(file_path)
    size = "unknown"
    inst = GATInstance(jobs, machine_ids, due_dates, file_name, size)

    # 初始化网络（与训练时一致）
    cfg = GATConfig()
    node_dim = 8
    edge_dim = 1
    gat = HomogeneousGAT(node_dim, edge_dim, cfg.gat_hidden_dim, cfg.gat_out_dim,
                         num_layers=cfg.gat_num_layers, num_heads=cfg.gat_num_heads).to(GAT_DEVICE)
    policy = GATPolicy(cfg.gat_out_dim, global_max_jobs * global_max_machines,
                       cfg.policy_hidden_dim, temperature=0.5).to(GAT_DEVICE)
    value = GATValue(cfg.gat_out_dim, cfg.policy_hidden_dim).to(GAT_DEVICE)

    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在，GAT+PPO运行可能失败。")
        return []
    try:
        checkpoint = torch.load(model_path, map_location=GAT_DEVICE, weights_only=False)
        gat.load_state_dict(checkpoint['gat'])
        policy.load_state_dict(checkpoint['policy'])
        value.load_state_dict(checkpoint['value'])
        gat.eval()
        policy.eval()
        value.eval()
        print(f"已加载 GAT+PPO 模型: {model_path}")
    except Exception as e:
        print(f"警告：GAT+PPO模型加载失败：{e}")
        return []

    # 状态归一化（如果需要，可以从 checkpoint 中加载，但这里简化，不使用归一化）
    state_norm = None

    objectives_list = []
    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = GATEnv(inst, global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date, cfg)
        state = env.reset()
        done = False
        while not done:
            # 准备输入
            state_gpu = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state_gpu[k] = v.to(GAT_DEVICE)
                else:
                    state_gpu[k] = v
            with torch.no_grad():
                global_emb, _ = gat(state_gpu)
                global_emb = global_emb.float()
                if state_norm is not None:
                    emb_norm = state_norm.normalize(global_emb.unsqueeze(0)).squeeze(0)
                else:
                    emb_norm = global_emb
                mask = state['action_mask'].to(GAT_DEVICE)
                probs = policy(emb_norm, mask)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()
            next_state, _, done, _ = env.step(action)
            state = next_state

        # 从环境提取指标
        cmax = env.current_time
        loads = env.machine_load.cpu().numpy() if torch.is_tensor(env.machine_load) else env.machine_load
        avg_load = np.mean(loads)
        lb = np.sqrt(np.mean((loads - avg_load)**2))
        tardy = 0
        for j in range(inst.n_jobs):
            due = inst.due_dates[j]
            comp = env.job_completion_time[j].item() if torch.is_tensor(env.job_completion_time[j]) else env.job_completion_time[j]
            tardy += max(0, comp - due)
        objectives_list.append([float(cmax), float(lb), float(tardy)])

    return objectives_list

def run_ppo_hgat(file_path, model_path, seed_list, global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date):
    """运行HeteroGAT+PPO模型，返回多个解的目标值"""
    if HGATInstance is None or HierarchicalPPOAgent is None:
        raise ImportError("PPO+HGAT 模块未正确导入")

    # 使用通用读取函数（PPO_HGAT 的读取函数返回5个值）
    jobs, machine_capabilities, due_dates, file_name, size = read_ppo_instance(file_path)

    # 创建实例对象，需要传入全局最大值
    inst = HGATInstance(jobs, machine_capabilities, due_dates, file_name, size,
                        global_max_proc_time, global_max_capability=1.0)  # 能力值暂时用1.0代替

    # 初始化网络（使用训练配置中的维度）
    dim_op, dim_mac = 3, 3
    hidden = HGATConfig.gat_hidden_dim
    out = HGATConfig.gat_out_dim
    policy_hidden = HGATConfig.policy_hidden_dim

    gat = HeteroGAT(dim_op, dim_mac, hidden, out).to(HGAT_DEVICE)
    upper = UpperPolicy(out, policy_hidden).to(HGAT_DEVICE)
    lower = LowerPolicy(out, policy_hidden).to(HGAT_DEVICE)
    critic_u = Critic(out, policy_hidden).to(HGAT_DEVICE)
    critic_l = Critic(out, policy_hidden).to(HGAT_DEVICE)

    agent = HierarchicalPPOAgent(inst, gat, upper, lower, critic_u, critic_l, HGATConfig)
    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在，HGAT+PPO运行可能失败。")
        return []
    try:
        agent.load(model_path)
        print(f"已加载 HGAT+PPO 模型: {model_path}")
    except Exception as e:
        print(f"警告：HGAT+PPO模型加载失败：{e}")
        return []
    agent.gat.eval()
    agent.upper.eval()
    agent.lower.eval()

    total_ops = inst.total_ops
    n_jobs = inst.n_jobs
    n_machines = inst.n_machines

    objectives_list = []
    for seed in seed_list:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = HGATEnv(inst, n_jobs, total_ops, n_machines,
                      global_max_proc_time, global_max_due_date, HGATConfig)
        state = env.reset()
        done = False
        while not done:
            actions, _, _ = agent.get_action_batch([state], [inst], deterministic=False)
            action_tuple = actions[0]
            next_state, _, done = env.step(action_tuple)
            state = next_state

        # 从环境属性提取结果
        job_completion = env.job_completion_time.cpu().numpy()
        machine_loads = env.machine_load.cpu().numpy()
        cmax = float(job_completion.max())
        lb = float(np.std(machine_loads))
        tardy = float(np.sum(np.maximum(0, job_completion - np.array(due_dates))))
        objectives_list.append([cmax, lb, tardy])

    return objectives_list


# 导出结果到Excel

def export_experiment2_to_excel(exp2_results, filename="Experiment2_Result.xlsx"):
    """
    将结果保存为Excel文件，包含三个工作表：小规模、中规模、大规模。
    每个工作表的格式与实验报告中的表26、27、28完全一致。
    数值格式：平均F、GAP、GD、SP保留两位小数，HV科学计数法两位小数，总耗时三位小数。
    """
    if not HAS_PANDAS:
        print("pandas未安装，无法导出Excel。")
        return

    # 定义规模分类（与实验报告一致）
    scale_categories = {
        '小规模': ['10x5', '15x10', '20x10'],
        '中规模': ['25x15', '30x20', '40x20'],
        '大规模': ['50x25', '75x25', '100x30']
    }

    # 方法列表（顺序与实验报告一致）
    methods = ['MOEA/D', 'NSGA-II', 'MLP+DQN', 'MLP+PPO', 'GAT+PPO', 'HeteroGAT+PPO']

    # 指标名称（与实验报告表头一致）
    metrics = ['平均F', 'GAP(%)', 'HV', 'GD', 'SP', '总耗时(s)']

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for category, scales in scale_categories.items():
            # 构建多级列索引：第一级为规模，第二级为指标
            columns = pd.MultiIndex.from_product([scales, metrics], names=['调度规模', '指标'])
            # 创建空DataFrame，行索引为方法名
            df = pd.DataFrame(index=methods, columns=columns)

            # 填充原始数值（用于后续格式化）
            for scale in scales:
                if scale not in exp2_results:
                    continue
                scale_data = exp2_results[scale]
                for method in methods:
                    if method in scale_data:
                        data = scale_data[method]
                        df.loc[method, (scale, '平均F')] = data['avg_F']
                        df.loc[method, (scale, 'GAP(%)')] = data['GAP']
                        df.loc[method, (scale, 'HV')] = data['HV']
                        df.loc[method, (scale, 'GD')] = data['GD']
                        df.loc[method, (scale, 'SP')] = data['SP']
                        df.loc[method, (scale, '总耗时(s)')] = data['time']

            # 格式化数值显示
            df_display = df.copy()
            for scale in scales:
                for metric in metrics:
                    col = (scale, metric)
                    if metric == 'HV':
                        df_display[col] = df[col].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else "")
                    elif metric == '总耗时(s)':
                        df_display[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
                    else:
                        df_display[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
            # 写入Excel
            df_display.to_excel(writer, sheet_name=category)

    print(f"\n结果已成功导出至Excel文件：{filename}")


# 主对比实验

def main():
    # 收集实例文件
    instance_dict = {}
    for jobs, mach in SIZE_LIST:
        pattern = f"comp_*_{jobs}x{mach}_*.txt"
        files = glob.glob(os.path.join(DATA_DIR, pattern))
        files.sort()
        instance_dict[(jobs, mach)] = files
        print(f"{jobs}×{mach}: 找到 {len(files)} 个实例")

    # 计算全局最大值
    global_max_jobs, global_max_machines, global_max_proc_time, global_max_due_date = compute_global_maxima(instance_dict)
    print(f"\n全局最大值：max_jobs={global_max_jobs}, max_machines={global_max_machines}, max_proc_time={global_max_proc_time:.2f}, max_due_date={global_max_due_date:.2f}")

    # 定义方法列表及对应的运行函数和模型路径
    methods = [
        ("MOEA/D", run_moead, None),
        ("NSGA-II", run_nsga2, None),
        ("MLP+DQN", run_dqn_mlp, DQN_MODEL_PATH),
        ("MLP+PPO", run_ppo_mlp, PPO_MLP_MODEL_PATH),
        ("GAT+PPO", run_ppo_gat, PPO_GAT_MODEL_PATH),
        ("HeteroGAT+PPO", run_ppo_hgat, PPO_HGAT_MODEL_PATH),
    ]

    # 存储结果：results[scale][method] = {
    #    'obj_vals': list of best weighted objectives for each instance,
    #    'all_objectives': list of lists of objectives for each instance (for HV/GD/SP)
    #    'times': list of runtimes for each instance
    # }
    results = defaultdict(lambda: defaultdict(lambda: {'obj_vals': [], 'all_objectives': [], 'times': []}))

    # 运行所有方法
    for (jobs, mach), file_list in instance_dict.items():
        scale_key = f"{jobs}x{mach}"
        if not file_list:
            continue
        print(f"\n===== 处理规模 {scale_key} =====")

        for method_name, run_func, model_path in methods:
            # 检查该方法是否可用
            if run_func is None:
                print(f"  跳过 {method_name}（模块未导入）")
                continue
            print(f"  运行 {method_name} ...")
            total_time = 0
            obj_vals = []
            all_obj = []
            for fpath in file_list:
                try:
                    start = time.time()
                    if method_name in ("MOEA/D", "NSGA-II"):
                        objectives_list = run_func(fpath)
                    else:
                        # 强化学习方法需要传递全局最大值
                        objectives_list = run_func(fpath, model_path, RL_SEEDS,
                                                  global_max_jobs, global_max_machines,
                                                  global_max_proc_time, global_max_due_date)
                    elapsed = time.time() - start

                    if len(objectives_list) == 0:
                        continue
                    w_objs = [weighted_objective(obj) for obj in objectives_list]
                    best_idx = np.argmin(w_objs)
                    obj_vals.append(w_objs[best_idx])
                    all_obj.append(objectives_list)
                    total_time += elapsed
                except Exception as e:
                    print(f"      实例 {os.path.basename(fpath)} 失败: {e}")
                    continue

            if obj_vals:
                results[scale_key][method_name]['obj_vals'] = obj_vals
                results[scale_key][method_name]['all_objectives'] = all_obj
                results[scale_key][method_name]['times'].append(total_time)


    # 计算各方法在每类规模上的平均指标，并收集用于导出

    print("\n" + "="*120)
    print("对比实验2 结果汇总")
    print("="*120)

    # 用于存储导出数据的字典
    exp2_results = defaultdict(dict)  # exp2_results[scale][method] = {avg_F, GAP, HV, GD, SP, time}

    for scale in sorted(results.keys(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
        print(f"\n【规模 {scale}】")

        # 收集该规模下所有实例的所有解，确定全局参考点
        all_objs_scale = []
        for method_data in results[scale].values():
            for inst_objs in method_data['all_objectives']:
                all_objs_scale.extend(inst_objs)
        if all_objs_scale:
            all_objs_scale = np.array(all_objs_scale)
            ref_point = np.max(all_objs_scale, axis=0) * 1.1
        else:
            ref_point = np.array([1000, 1000, 1000])

        # 为每个实例计算真实前沿
        num_instances = len(next(iter(results[scale].values()))['obj_vals'])
        inst_all_objs = [[] for _ in range(num_instances)]
        for method_data in results[scale].values():
            for inst_idx, inst_objs in enumerate(method_data['all_objectives']):
                inst_all_objs[inst_idx].extend(inst_objs)

        true_frontiers = []
        for inst_idx in range(num_instances):
            if inst_all_objs[inst_idx]:
                true_idx = nondominated_sort(inst_all_objs[inst_idx])
                if true_idx:
                    true_objs = [inst_all_objs[inst_idx][i] for i in true_idx]
                else:
                    true_objs = []
                true_frontiers.append(true_objs)
            else:
                true_frontiers.append([])

        avg_F = {}
        avg_gap = {}
        avg_hv = {}
        avg_gd = {}
        avg_sp = {}
        total_time = {}

        for method_name, _, _ in methods:
            if method_name not in results[scale]:
                continue
            data = results[scale][method_name]
            obj_vals = data['obj_vals']
            all_objs = data['all_objectives']
            time_vals = data['times']
            if not obj_vals:
                continue

            avg_F[method_name] = np.mean(obj_vals)
            total_time[method_name] = np.sum(time_vals) if time_vals else 0

            hv_list = []
            gd_list = []
            sp_list = []

            for inst_idx, inst_objs in enumerate(all_objs):
                if len(inst_objs) == 0:
                    continue
                hv = compute_hypervolume(inst_objs, ref_point)
                hv_list.append(hv)
                sp = compute_sp(inst_objs)
                sp_list.append(sp)
                true_pf = true_frontiers[inst_idx]
                if len(true_pf) > 0:
                    gd = compute_gd(inst_objs, true_pf)
                    gd_list.append(gd)
                else:
                    gd_list.append(np.nan)

            avg_hv[method_name] = np.mean(hv_list) if hv_list else np.nan
            avg_gd[method_name] = np.mean(gd_list) if gd_list else np.nan
            avg_sp[method_name] = np.mean(sp_list) if sp_list else np.nan

        # 以HeteroGAT+PPO为基准计算GAP
        baseline = avg_F.get("HeteroGAT+PPO", np.nan)
        for method_name in avg_F.keys():
            if method_name == "HeteroGAT+PPO":
                avg_gap[method_name] = 0.0
            else:
                avg_gap[method_name] = (avg_F[method_name] - baseline) / baseline * 100 if baseline != 0 else np.nan

        # 存储到exp2_results，用于导出
        for method_name in avg_F.keys():
            exp2_results[scale][method_name] = {
                'avg_F': avg_F[method_name],
                'GAP': avg_gap[method_name],
                'HV': avg_hv[method_name],
                'GD': avg_gd[method_name],
                'SP': avg_sp[method_name],
                'time': total_time[method_name]
            }

        # 打印结果（控制台）
        header = (f"{'方法':<12} | {'平均F':>10} | {'GAP(%)':>7} | {'HV':>10} | {'GD':>10} | {'SP':>10} | {'总耗时(s)':>10}")
        print(header)
        print("-" * len(header))

        for method_name, _, _ in methods:
            if method_name not in avg_F:
                continue
            # 控制台简单格式化，保留两位小数（HV除外）
            hv_str = f"{avg_hv[method_name]:.4f}" if not np.isnan(avg_hv[method_name]) else "NaN"
            gd_str = f"{avg_gd[method_name]:.4f}" if not np.isnan(avg_gd[method_name]) else "NaN"
            sp_str = f"{avg_sp[method_name]:.4f}" if not np.isnan(avg_sp[method_name]) else "NaN"
            print(f"{method_name:<12} | {avg_F[method_name]:>10.2f} | {avg_gap.get(method_name, np.nan):>7.2f} | "
                  f"{hv_str:>10} | {gd_str:>10} | {sp_str:>10} | {total_time.get(method_name, 0):>10.2f}")

    print("="*120)

    # 导出结果到Excel
    export_experiment2_to_excel(exp2_results, filename=os.path.join(PROJECT_ROOT, "Experiment2_Result.xlsx"))

if __name__ == "__main__":
    main()