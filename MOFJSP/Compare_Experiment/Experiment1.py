"""
对比实验1：简单规则 vs PPO_HGAT（加载预训练模型）
对比的规模（jobs x machines）由固定列表定义，每个规模应有10个实例。
"""

import os
import time
import glob
import sys
import numpy as np
from collections import defaultdict

# 尝试导入pandas用于Excel输出
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


# 导入简单规则模块

try:
    from Heuristic.SPT import SPTScheduler, read_fjsp_instance as read_spt
except ImportError:
    print("警告：无法导入 SPT 模块，请检查 Heuristic/SPT.py 是否存在")
    SPTScheduler = None
    read_spt = None

try:
    from Heuristic.EDD import EDDScheduler, read_fjsp_instance as read_edd
except ImportError:
    print("警告：无法导入 EDD 模块")
    EDDScheduler = None
    read_edd = None

try:
    from Heuristic.LBD import LBDScheduler, read_fjsp_instance as read_lbd
except ImportError:
    print("警告：无法导入 LBD 模块")
    LBDScheduler = None
    read_lbd = None

try:
    from Heuristic.FIFO import FIFOScheduler, read_fjsp_instance as read_fifo
except ImportError:
    print("警告：无法导入 FIFO 模块")
    FIFOScheduler = None
    read_fifo = None

try:
    from Heuristic.MOPNR import MOPNRScheduler, read_fjsp_instance as read_mopnr
except ImportError:
    print("警告：无法导入 MOPNR 模块")
    MOPNRScheduler = None
    read_mopnr = None

try:
    from Heuristic.MWKR import MWScheduler, read_fjsp_instance as read_mwkr
except ImportError:
    print("警告：无法导入 MWKR 模块")
    MWScheduler = None
    read_mwkr = None


# 导入 PPO_HGAT 所需组件

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
        cfg,                     # 训练配置，用于环境初始化和网络维度
        EPS,
        read_fjsp_instance as read_ppo_instance,
    )
except ImportError as e:
    print(f"错误：无法导入 PPO_HGAT 模块：{e}")
    print("请确保 PPO_HGAT/PPO_HGAT.py 存在且包含所需类，且文件夹名为 PPO_HGAT（不含加号）。")
    sys.exit(1)


# 规则名称到调度器类和读取函数的映射

RULE_MAP = {
    'SPT': (SPTScheduler, read_spt),
    'EDD': (EDDScheduler, read_edd),
    'LBD': (LBDScheduler, read_lbd),
    'FIFO': (FIFOScheduler, read_fifo),
    'MOPNR': (MOPNRScheduler, read_mopnr),
    'MWKR': (MWScheduler, read_mwkr),
}

def run_simple_rule(rule_name, file_path):
    """运行指定的简单规则，返回 (makespan, load_balance, total_tardiness, time_cost)"""
    if rule_name not in RULE_MAP:
        raise ValueError(f"未知规则: {rule_name}")

    scheduler_class, read_func = RULE_MAP[rule_name]
    if scheduler_class is None or read_func is None:
        raise ImportError(f"规则 {rule_name} 的模块未正确导入")

    start_time = time.time()
    jobs, machines = read_func(file_path)   # 读取实例，返回 (jobs, machines)
    scheduler = scheduler_class(jobs, machines)
    scheduler.run_schedule()

    # 计算指标
    makespan = max(job.get_completion_time() for job in jobs)
    machine_loads = [m.get_total_load() for m in machines]
    load_balance = np.std(machine_loads)   # 标准差
    total_tardiness = sum(max(0, job.get_completion_time() - job.due_date) for job in jobs)

    elapsed = time.time() - start_time
    return makespan, load_balance, total_tardiness, elapsed

def run_ppo_hgat(file_path, model_path):
    """加载预训练的PPO+HGAT模型，运行调度，返回 (makespan, load_balance, total_tardiness, time_cost)"""
    start_time = time.time()

    # 读取实例，返回5个值
    jobs, machine_capabilities, due_dates, file_name, size = read_ppo_instance(file_path)

    # 从实例数据中计算全局最大值（用于归一化）
    max_proc_time = 0.0
    for job_ops in jobs:
        for op in job_ops:
            for t in op.machine_times.values():
                max_proc_time = max(max_proc_time, t)
    max_capability = max(machine_capabilities) if machine_capabilities else 0
    max_due_date = max(due_dates) if due_dates else 0
    total_ops = sum(len(job_ops) for job_ops in jobs)
    n_jobs = len(jobs)
    n_machines = len(machine_capabilities)

    # 创建实例对象，需要传入全局最大值
    inst = MOFJSPInstance(jobs, machine_capabilities, due_dates, file_name, size,
                          max_proc_time, max_capability)

    # 初始化网络，使用与训练时一致的超参数
    dim_op, dim_mac = 3, 3   # 操作节点特征维度3，机器节点特征维度3
    hidden = cfg.gat_hidden_dim          # 256
    out = cfg.gat_out_dim                # 256
    policy_hidden = cfg.policy_hidden_dim  # 512
    gat = HeteroGAT(dim_op, dim_mac, hidden, out).to(DEVICE)
    upper = UpperPolicy(out, policy_hidden).to(DEVICE)
    lower = LowerPolicy(out, policy_hidden).to(DEVICE)
    critic_u = Critic(out, policy_hidden).to(DEVICE)
    critic_l = Critic(out, policy_hidden).to(DEVICE)

    # 创建agent并加载模型（传入训练配置 cfg）
    agent = HierarchicalPPOAgent(inst, gat, upper, lower, critic_u, critic_l, cfg)
    agent.load(model_path)          # 加载最佳模型
    agent.gat.eval()
    agent.upper.eval()
    agent.lower.eval()

    # 创建环境（需要全局最大值和配置）
    env = HeteroGraphEnv(inst, n_jobs, total_ops, n_machines,
                         max_proc_time, max_due_date, cfg)
    state = env.reset()
    done = False
    while not done:
        # 使用批量方法处理单个状态
        actions, _, _ = agent.get_action_batch([state], [inst], deterministic=True)
        action_tuple = actions[0]
        next_state, _, done = env.step(action_tuple)
        state = next_state

    # 从环境属性中提取结果
    job_completion = env.job_completion_time.cpu().numpy()
    machine_loads = env.machine_load.cpu().numpy()

    makespan = float(job_completion.max())
    load_balance = float(np.std(machine_loads))
    total_tardiness = float(np.sum(np.maximum(0, job_completion - np.array(due_dates))))

    elapsed = time.time() - start_time
    return makespan, load_balance, total_tardiness, elapsed


# 定义实例文件收集函数

def collect_instances(data_dir, size_list):
    instance_dict = {}
    for jobs, machines in size_list:
        pattern = f"comp_*_{jobs}x{machines}_*.txt"
        files = glob.glob(os.path.join(data_dir, pattern))
        files.sort()
        instance_dict[(jobs, machines)] = files
    return instance_dict


# 导出结果到Excel

def export_to_excel(results, filename="Experiment1_Result.xlsx"):
    """
    将结果保存为Excel文件，包含三个工作表：小规模、中规模、大规模。
    数值按指定小数位四舍五入：C(平均最大完工时间)保留1位，LB(平均负载均衡度)保留2位，
    T(平均总拖期)保留1位，t(总耗时)保留3位。
    """
    if not HAS_PANDAS:
        print("pandas未安装，无法导出Excel。")
        return

    # 定义规模分类
    small_scales = ["10x5", "15x10", "20x10"]
    medium_scales = ["25x15", "30x20", "40x20"]
    large_scales = ["50x25", "75x25", "100x30"]
    # 所有方法（顺序与实验报告一致）
    methods = ['SPT', 'EDD', 'LBD', 'FIFO', 'MOPNR', 'MWKR', 'HeteroGAT+PPO']

    # 指标名称（与实验报告表头一致）
    metric_names = ['平均最大完工时间', '平均负载均衡度', '平均总拖期', '总耗时(秒)']

    # 创建一个ExcelWriter对象
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 处理每个规模类别
        for category, scales in [("小规模", small_scales), ("中规模", medium_scales), ("大规模", large_scales)]:
            # 构建多级列索引：第一级为规模，第二级为指标
            columns = pd.MultiIndex.from_product([scales, metric_names], names=['调度规模', '指标'])
            # 创建空DataFrame，行索引为方法名
            df = pd.DataFrame(index=methods, columns=columns)

            # 填充数据，并对数值进行四舍五入
            for scale in scales:
                if scale not in results:
                    continue
                scale_data = results[scale]
                for method in methods:
                    if method in scale_data:
                        avg_m, avg_lb, avg_tardy, total_t, cnt = scale_data[method]
                        # 按指定小数位四舍五入
                        df.loc[method, (scale, '平均最大完工时间')] = round(avg_m, 1)
                        df.loc[method, (scale, '平均负载均衡度')] = round(avg_lb, 2)
                        df.loc[method, (scale, '平均总拖期')] = round(avg_tardy, 1)
                        df.loc[method, (scale, '总耗时(秒)')] = round(total_t, 3)
                    # 如果方法没有该规模的数据，单元格将保持NaN，写入Excel时会变为空

            # 将DataFrame写入对应工作表
            df.to_excel(writer, sheet_name=category)

    print(f"\n结果已成功导出至Excel文件：{filename}")

# 主对比实验

def main():
    # 使用基于项目根目录的路径
    DATA_DIR = os.path.join(PROJECT_ROOT, "comparison_instances")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model/ppo_hgat/ppo_hgat_best.pth")

    # 定义要对比的固定尺寸
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

    # 检查实例目录是否存在
    if not os.path.isdir(DATA_DIR):
        print(f"错误：实例目录 '{DATA_DIR}' 不存在，请确保已生成实例并放置在正确位置。")
        return

    # 收集实例
    instance_dict = collect_instances(DATA_DIR, size_list)
    for (jobs, mach), files in instance_dict.items():
        print(f"{jobs}×{mach}: 找到 {len(files)} 个实例")
        if len(files) == 0:
            print(f"警告：规模 {jobs}×{mach} 没有实例文件，跳过")

    # 定义要比较的方法
    methods = ['SPT', 'EDD', 'LBD', 'FIFO', 'MOPNR', 'MWKR', 'PPO_HGAT']

    # 存储结果：results[scale_key][method] = (avg_makespan, avg_load_balance, avg_tardiness, total_time, count)
    results = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0]))

    # 对每个规模、每个方法、每个实例运行
    for (jobs, mach), file_list in instance_dict.items():
        if not file_list:
            continue
        scale_key = f"{jobs}x{mach}"
        print(f"\n===== 处理规模 {scale_key} =====")
        for method in methods:
            print(f"  运行 {method} ...")
            total_makespan = 0.0
            total_load = 0.0
            total_tardy = 0.0
            total_time = 0.0
            count = 0
            for fpath in file_list:
                try:
                    if method == 'PPO_HGAT':
                        if not os.path.isfile(MODEL_PATH):
                            raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")
                        m, lb, tardy, t = run_ppo_hgat(fpath, MODEL_PATH)
                    else:
                        m, lb, tardy, t = run_simple_rule(method, fpath)
                    total_makespan += m
                    total_load += lb
                    total_tardy += tardy
                    total_time += t
                    count += 1
                except Exception as e:
                    print(f"      实例 {os.path.basename(fpath)} 失败: {e}")
            if count > 0:
                avg_m = total_makespan / count
                avg_lb = total_load / count
                avg_tardy = total_tardy / count
                total_t = total_time
                results[scale_key][method] = (avg_m, avg_lb, avg_tardy, total_t, count)

    # 打印结果表格（控制台输出）
    print("\n" + "="*120)
    print("对比实验1 结果汇总")
    print("="*120)
    for scale_key in sorted(results.keys(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
        print(f"\n【规模 {scale_key}】")
        header = f"{'方法':<12} | {'平均最大完工时间':>12} | {'平均负载均衡度':>14} | {'平均总拖期':>12} | {'总耗时(秒)':>10}"
        print(header)
        print("-" * len(header))
        for method in methods:
            if method in results[scale_key]:
                avg_m, avg_lb, avg_tardy, total_t, cnt = results[scale_key][method]
                print(f"{method:<12} | {avg_m:>12.1f} | {avg_lb:>14.2f} | {avg_tardy:>12.1f} | {total_t:>10.3f}")
            else:
                print(f"{method:<12} | {'-':>12} | {'-':>14} | {'-':>12} | {'-':>10}")
    print("="*120)

    # 导出结果到Excel（与实验报告格式一致）
    export_results = {}
    for scale_key, method_dict in results.items():
        export_results[scale_key] = {}
        for method, vals in method_dict.items():
            if method == 'PPO_HGAT':
                export_results[scale_key]['HeteroGAT+PPO'] = vals
            else:
                export_results[scale_key][method] = vals

    export_to_excel(export_results, filename=os.path.join(PROJECT_ROOT, "Experiment1_Result.xlsx"))

if __name__ == "__main__":
    main()