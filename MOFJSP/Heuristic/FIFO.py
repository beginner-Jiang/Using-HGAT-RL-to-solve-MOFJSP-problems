"""
启发式规则实现：先来先服务-FIFO
"""
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# 数据结构定义

class Operation:
    """工序类"""

    def __init__(self, job_id: int, op_id: int, machine_times: Dict[int, float]):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_times = machine_times  # 机器ID -> 加工时间
        self.assigned_machine = None
        self.start_time = 0
        self.end_time = 0
        self.is_scheduled = False

    def get_min_processing_time(self) -> float:
        """返回最短加工时间"""
        return min(self.machine_times.values())

    def get_best_machine(self) -> int:
        """返回加工时间最短的机器ID"""
        return min(self.machine_times.items(), key=lambda x: x[1])[0]


class Job:
    """作业类"""

    def __init__(self, job_id: int, operations: List[Operation], due_date: float, release_time: float = 0):
        self.job_id = job_id
        self.operations = operations
        self.due_date = due_date          # 交货期（用于性能评估）
        self.release_time = release_time  # 作业到达时间（FIFO 依据，默认为0）
        self.current_op_index = 0         # 当前待调度的工序索引
        self.remaining_ops = len(operations)  # 剩余工序数
        self.remaining_processing_time = sum(min(op.machine_times.values()) for op in operations)  # 剩余最短加工时间总和

    def get_current_operation(self) -> Operation:
        """返回当前待调度的工序"""
        if self.current_op_index < len(self.operations):
            return self.operations[self.current_op_index]
        return None

    def complete_current_operation(self):
        """标记当前工序完成，准备下一个工序"""
        if self.current_op_index < len(self.operations):
            self.remaining_ops -= 1
            current_op = self.get_current_operation()
            if current_op:
                self.remaining_processing_time -= current_op.get_min_processing_time()
            self.current_op_index += 1

    def is_completed(self) -> bool:
        """检查作业是否全部完成"""
        return self.current_op_index >= len(self.operations)

    def get_completion_time(self) -> float:
        """返回作业完成时间（最后一道工序的完成时间）"""
        if len(self.operations) == 0:
            return 0
        return self.operations[-1].end_time

    def get_remaining_processing_time(self) -> float:
        """返回剩余最短加工时间总和"""
        return self.remaining_processing_time

    def get_urgency(self, current_time: float) -> float:
        """计算作业紧迫度（保留，但在FIFO中不使用）"""
        remaining_time_buffer = self.due_date - current_time - self.get_remaining_processing_time()
        return remaining_time_buffer


class Machine:
    """机器类"""

    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.schedule = []  # 已安排的工序列表 [(start_time, end_time, operation)]
        self.available_time = 0  # 机器最早可用时间

    def assign_operation(self, operation: Operation, start_time: float) -> float:
        """将工序分配到机器上，返回完成时间（仅允许分配到最优机器）"""
        best_machine_id = operation.get_best_machine()
        if best_machine_id != self.machine_id:
            return float('inf')  # 不是最优机器，不分配

        processing_time = operation.machine_times[self.machine_id]
        actual_start = max(start_time, self.available_time)
        end_time = actual_start + processing_time

        self.schedule.append((actual_start, end_time, operation))
        self.available_time = end_time

        operation.assigned_machine = self.machine_id
        operation.start_time = actual_start
        operation.end_time = end_time
        operation.is_scheduled = True

        return end_time

    def get_total_load(self) -> float:
        """返回机器总负载（加工时间总和）"""
        return sum(end - start for start, end, _ in self.schedule)

    def get_available_time(self) -> float:
        """返回机器最早可用时间"""
        return self.available_time


# FIFO调度算法实现

class FIFOScheduler:
    """基于FIFO（先进先出，即最早到达优先）规则的调度器"""

    def __init__(self, jobs: List[Job], machines: List[Machine]):
        self.jobs = jobs
        self.machines = machines
        self.current_time = 0
        self.completed_operations = 0
        self.total_operations = sum(len(job.operations) for job in jobs)

    def get_available_operations(self) -> List[Tuple[Operation, Job]]:
        """获取所有可调度的工序（各作业的当前未调度工序）"""
        available_ops = []
        for job in self.jobs:
            if not job.is_completed():
                current_op = job.get_current_operation()
                if current_op and not current_op.is_scheduled:
                    available_ops.append((current_op, job))
        return available_ops

    def get_current_makespan(self) -> float:
        """获取当前最大完工时间"""
        makespan = 0
        for machine in self.machines:
            if machine.schedule:
                last_end_time = max(end for _, end, _ in machine.schedule)
                makespan = max(makespan, last_end_time)
        return makespan

    def schedule_step(self) -> bool:
        """执行一步调度，返回是否还有工序需要调度"""
        available_ops_with_jobs = self.get_available_operations()

        if not available_ops_with_jobs:
            return False

        # FIFO规则：选择到达时间最早的作业的当前工序
        # 若到达时间相同，则选择作业ID较小的（即先进入系统）
        selected_op, selected_job = min(available_ops_with_jobs,
                                        key=lambda x: (x[1].release_time, x[1].job_id))

        # 为工序选择最优机器
        best_machine_id = selected_op.get_best_machine()
        best_machine = self.machines[best_machine_id]

        # 计算作业允许开始时间（前序工序完成时间）
        job_ready_time = 0
        if selected_op.op_id > 0:
            prev_op = selected_job.operations[selected_op.op_id - 1]
            job_ready_time = prev_op.end_time

        # 分配工序
        completion_time = best_machine.assign_operation(selected_op, job_ready_time)

        # 更新作业状态
        if completion_time != float('inf'):
            selected_job.complete_current_operation()
            self.completed_operations += 1

        # 更新当前时间（所有机器最早可用时间的最小值）
        machine_times = [machine.available_time for machine in self.machines]
        self.current_time = min(machine_times) if machine_times else self.current_time

        return True

    def run_schedule(self):
        """运行完整调度"""
        while self.schedule_step():
            pass

    def calculate_metrics(self) -> Dict[str, float]:
        """计算调度性能指标"""
        # 1. 最大完工时间
        completion_times = [job.get_completion_time() for job in self.jobs]
        makespan = max(completion_times)

        # 2. 机器负载均衡度（变异系数）
        machine_loads = [machine.get_total_load() for machine in self.machines]
        avg_load = np.mean(machine_loads)
        load_balance = np.sqrt(np.mean([(load - avg_load) ** 2 for load in machine_loads])) / avg_load if avg_load > 0 else 0

        # 3. 总拖期时间
        total_tardiness = 0
        tardy_jobs = []
        for job in self.jobs:
            completion_time = job.get_completion_time()
            tardiness = max(0, completion_time - job.due_date)
            total_tardiness += tardiness
            if tardiness > 0:
                tardy_jobs.append((job.job_id, completion_time, job.due_date, tardiness))

        # 4. 平均拖期时间
        avg_tardiness = total_tardiness / len(self.jobs) if self.jobs else 0

        # 5. 拖期作业比例
        tardy_ratio = len(tardy_jobs) / len(self.jobs) if self.jobs else 0

        return {
            "makespan": makespan,
            "load_balance": load_balance,
            "total_tardiness": total_tardiness,
            "avg_tardiness": avg_tardiness,
            "tardy_ratio": tardy_ratio,
            "tardy_jobs": tardy_jobs
        }

    def print_schedule(self):
        """打印调度结果"""
        print("=" * 80)
        print("FIFO调度结果（先进先出，最早到达优先）")
        print("=" * 80)

        # 按机器打印
        for machine in self.machines:
            print(f"\n机器 {machine.machine_id} (总负载: {machine.get_total_load():.1f}):")
            machine.schedule.sort(key=lambda x: x[0])
            for start, end, op in machine.schedule:
                job_release = self.jobs[op.job_id].release_time
                print(f"  作业{op.job_id}-工序{op.op_id}: [{start:.1f} - {end:.1f}], 作业到达时间: {job_release}")

        # 打印性能指标
        metrics = self.calculate_metrics()
        print(f"\n{'=' * 80}")
        print("性能指标:")
        print(f"  最大完工时间 (Makespan): {metrics['makespan']:.2f}")
        print(f"  机器负载均衡度: {metrics['load_balance']:.4f}")
        print(f"  总拖期时间: {metrics['total_tardiness']:.2f}")
        print(f"  平均拖期时间: {metrics['avg_tardiness']:.2f}")
        print(f"  拖期作业比例: {metrics['tardy_ratio']:.2%}")

        # 打印拖期作业详情
        if metrics['tardy_jobs']:
            print(f"\n{'=' * 80}")
            print("拖期作业详情:")
            for job_id, completion_time, due_date, tardiness in metrics['tardy_jobs']:
                print(f"  作业{job_id}: 完成时间={completion_time:.1f}, 交货期={due_date}, 拖期时间={tardiness:.1f}")

        # 打印作业完成情况
        print(f"\n{'=' * 80}")
        print("作业完成情况:")
        for job in self.jobs:
            completion_time = job.get_completion_time()
            tardiness = max(0, completion_time - job.due_date)
            status = "拖期" if tardiness > 0 else "准时"
            print(f"  作业{job.job_id}: 到达时间={job.release_time}, 交货期={job.due_date}, "
                  f"完成时间={completion_time:.1f}, 拖期时间={tardiness:.1f}, 状态={status}")

        return metrics


# 文件读取模块

def read_fjsp_instance(file_path: str) -> Tuple[List[Job], List[Machine]]:
    """
    读取标准 MOFJSP 数据文件（扩展 Brandimarte 格式）
    文件格式：
        第一行: 作业数  机器数
        接下来 N 行: 每个作业的工序信息
            - 第一个整数: 该作业的工序数量 O
            - 后续 O 组: 每组以 k 开头，后跟 k 对 (机器ID, 加工时间)
        然后一行: N 个整数，每个作业的交货期
        可能还有额外行（如柔性矩阵等），忽略
    注意：标准实例中不包含作业到达时间，本函数默认将所有作业的到达时间设为0，
         如需体现 FIFO 规则，可在此处修改或从外部指定。
    返回: (jobs列表, machines列表)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 过滤空行和注释行
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到！")
        exit(1)

    if len(lines) < 2:
        raise ValueError("文件格式错误：行数不足")

    # 第一行：作业数 机器数
    num_jobs, num_machines = map(int, lines[0].split())
    machines = [Machine(i) for i in range(num_machines)]

    # 接下来的 num_jobs 行是作业工序数据
    job_lines = lines[1:1 + num_jobs]
    if len(job_lines) != num_jobs:
        raise ValueError(f"文件格式错误：期望 {num_jobs} 个作业行，实际得到 {len(job_lines)} 行")

    # 再下一行是交货期（必须存在）
    due_date_line = lines[1 + num_jobs]
    due_dates = list(map(float, due_date_line.split()))
    if len(due_dates) != num_jobs:
        raise ValueError(f"文件格式错误：期望 {num_jobs} 个交货期，实际得到 {len(due_dates)} 个")

    jobs = []
    for job_idx in range(num_jobs):
        nums = list(map(int, job_lines[job_idx].split()))
        if len(nums) == 0:
            continue
        num_ops = nums[0]
        idx = 1
        operations = []
        for op_idx in range(num_ops):
            if idx >= len(nums):
                raise ValueError(f"作业 {job_idx} 的数据不完整")
            k = nums[idx]
            idx += 1
            machine_times = {}
            for _ in range(k):
                if idx + 1 >= len(nums):
                    raise ValueError(f"作业 {job_idx} 工序 {op_idx} 的机器数据不足")
                machine_id = nums[idx] - 1  # 转换为0索引
                processing_time = float(nums[idx + 1])
                if 0 <= machine_id < num_machines:
                    machine_times[machine_id] = processing_time
                idx += 2
            op = Operation(job_idx, op_idx, machine_times)
            operations.append(op)
        due_date = due_dates[job_idx]
        # 默认所有作业到达时间为0，如需模拟不同到达时间，可在此处修改
        release_time = 0.0
        job = Job(job_idx, operations, due_date, release_time)
        jobs.append(job)

    return jobs, machines


# 可视化函数

def visualize_gantt_chart(jobs: List[Job], machines: List[Machine],
                          title="FIFO Rule - Gantt Chart", save_path=None):
    """绘制甘特图并保存"""
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#EF476F', '#073B4C', '#118AB2']
    job_colors = {job.job_id: colors[i % len(colors)] for i, job in enumerate(jobs)}

    # 为每一台机器绘制调度
    for machine in machines:
        y_pos = machine.machine_id
        for start, end, op in machine.schedule:
            color = job_colors[op.job_id]
            ax.barh(y_pos, end - start, left=start, height=0.6,
                    color=color, edgecolor='black')
            # 工序标签移动

    # 增加释放时间线
    for job in jobs:
        release_time = job.release_time
        ax.axvline(x=release_time, color=job_colors[job.job_id], linestyle=':', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {i}' for i in range(len(machines))])
    ax.set_title(title, fontsize=14)

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=job_colors[job.job_id],
                            label=f'Job {job.job_id} (Release={job.release_time}, Due={job.due_date})')
                      for job in jobs]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.grid(axis='x', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to: {save_path}")

    plt.show()


# 主函数

def main():
    """主函数：读取文件，执行FIFO调度，展示并保存甘特图，导出指标到Excel"""
    print("=" * 80)
    print("基于FIFO（先进先出）规则的柔性作业车间调度（文件驱动）")
    print("=" * 80)

    # 指定数据文件路径
    file_path = "../mo_fjsp_instances/mo_fjsp_000_small_train.txt"  # 确保文件存在

    # 1. 从文件读取调度实例
    try:
        jobs, machines = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print("\n问题描述:")
    print(f"  作业数量: {len(jobs)}")
    print(f"  机器数量: {len(machines)}")
    total_ops = sum(len(job.operations) for job in jobs)
    print(f"  总工序数量: {total_ops}")

    print("\n作业到达时间与交货期:")
    for job in jobs:
        min_total_time = sum(min(op.machine_times.values()) for op in job.operations)
        print(f"  作业{job.job_id}: 到达时间={job.release_time:.1f}, "
              f"最短完成时间={min_total_time:.1f}, 交货期={job.due_date:.1f}")

    # 2. 执行FIFO调度
    scheduler = FIFOScheduler(jobs, machines)
    scheduler.run_schedule()

    # 3. 打印调度结果并获取指标
    fifo_metrics = scheduler.print_schedule()

    # 4. 绘制并保存甘特图
    print("\n正在生成甘特图...")
    visualize_gantt_chart(jobs, machines, save_path="../Figure_And_File/Heuristic/FIFO/gantt_chart_fifo.png")

    # 5. 将性能指标导出为Excel文件
    try:
        # 创建汇总指标DataFrame
        summary_data = {
            '指标': ['最大完工时间 (Makespan)', '机器负载均衡度', '总拖期时间', '平均拖期时间', '拖期作业比例'],
            '数值': [
                fifo_metrics['makespan'],
                fifo_metrics['load_balance'],
                fifo_metrics['total_tardiness'],
                fifo_metrics['avg_tardiness'],
                fifo_metrics['tardy_ratio']
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        # 创建拖期作业详情DataFrame（如果有）
        tardy_jobs_list = []
        for job_id, comp_time, due, tard in fifo_metrics['tardy_jobs']:
            tardy_jobs_list.append({
                '作业ID': job_id,
                '完成时间': comp_time,
                '交货期': due,
                '拖期时间': tard
            })
        df_tardy = pd.DataFrame(tardy_jobs_list) if tardy_jobs_list else pd.DataFrame({'提示': ['无拖期作业']})

        # 写入Excel文件，包含两个sheet
        excel_file = '../Figure_And_File/Heuristic/FIFO/FIFO_metrics.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_tardy.to_excel(writer, sheet_name='TardyJobs', index=False)

        print(f"\n性能指标已成功导出到 {excel_file}")
    except ImportError:
        print("\n警告：未安装 pandas 或 openpyxl，无法导出Excel。请安装：pip install pandas openpyxl")
    except Exception as e:
        print(f"\n导出Excel时出错：{e}")

    # 6. 调度可行性验证
    print("\n" + "=" * 80)
    print("调度可行性验证:")
    print("-" * 80)

    unscheduled_ops = []
    for job in jobs:
        for op in job.operations:
            if not op.is_scheduled:
                unscheduled_ops.append((job.job_id, op.op_id))

    if unscheduled_ops:
        print(f"  警告: {len(unscheduled_ops)} 个工序未被调度")
        for job_id, op_id in unscheduled_ops:
            print(f"    作业{job_id}-工序{op_id}")
    else:
        print("  所有工序都已成功调度")

    conflicts = []
    for machine in machines:
        schedule = sorted(machine.schedule, key=lambda x: x[0])
        for i in range(1, len(schedule)):
            prev_end = schedule[i - 1][1]
            curr_start = schedule[i][0]
            if curr_start < prev_end:
                conflicts.append((machine.machine_id, i - 1, i))

    if conflicts:
        print(f"  警告: 发现 {len(conflicts)} 处机器时间冲突")
    else:
        print("  无机器时间冲突")

    print("\nFIFO调度完成!")


if __name__ == "__main__":
    main()