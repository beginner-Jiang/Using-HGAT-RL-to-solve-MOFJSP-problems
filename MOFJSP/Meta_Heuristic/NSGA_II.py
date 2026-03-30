"""
元启发式算法实现：非支配排序遗传算法 II-NSGA-II
"""
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Set
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import copy
import math
import matplotlib


# 数据结构定义

@dataclass
class Operation:
    """工序类"""
    job_id: int
    op_id: int
    machine_times: Dict[int, float]  # 机器ID -> 加工时间

    def get_processing_time(self, machine_id: int) -> float:
        """返回指定机器上的加工时间"""
        return self.machine_times.get(machine_id, float('inf'))


@dataclass
class Job:
    """作业类"""
    job_id: int
    operations: List[Operation]
    due_date: float


@dataclass
class Chromosome:
    """染色体类，表示一个调度解"""
    operation_sequence: List[Tuple[int, int]]  # 工序序列 [(job_id, op_id), ...]
    machine_assignment: List[int]  # 机器分配序列，与工序序列一一对应
    objectives: List[float] = None  # 目标函数值 [makespan, load_balance, total_tardiness]
    rank: int = 0  # 非支配排序等级
    crowding_distance: float = 0.0  # 拥挤度距离

    def __post_init__(self):
        """验证染色体有效性"""
        if len(self.operation_sequence) != len(self.machine_assignment):
            raise ValueError("工序序列和机器分配序列长度必须相同")

    def copy(self):
        """创建染色体的深拷贝"""
        return Chromosome(
            operation_sequence=self.operation_sequence.copy(),
            machine_assignment=self.machine_assignment.copy(),
            objectives=self.objectives.copy() if self.objectives else None,
            rank=self.rank,
            crowding_distance=self.crowding_distance
        )

    def __len__(self):
        """返回染色体长度（工序数量）"""
        return len(self.operation_sequence)


class NSGA2Scheduler:
    """基于NSGA-II算法的调度器"""

    def __init__(self, jobs: List[Job], machines: List[int],
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1):
        """
        初始化NSGA-II调度器

        Args:
            jobs: 作业列表
            machines: 机器ID列表
            population_size: 种群大小
            max_generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
        """
        self.jobs = jobs
        self.machines = machines
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # 计算总工序数
        self.total_operations = sum(len(job.operations) for job in jobs)

        # 创建作业-工序映射
        self.job_ops_map = {}
        self.op_index_map = {}  # (job_id, op_id) -> 全局索引
        self.global_index_to_op = {}  # 全局索引 -> (job_id, op_id)

        idx = 0
        for job in jobs:
            self.job_ops_map[job.job_id] = []
            for op in job.operations:
                self.job_ops_map[job.job_id].append((job.job_id, op.op_id))
                self.op_index_map[(job.job_id, op.op_id)] = idx
                self.global_index_to_op[idx] = (job.job_id, op.op_id)
                idx += 1

        # 种群存储
        self.population: List[Chromosome] = []
        self.fronts: List[List[Chromosome]] = []  # 非支配前沿

        # 记录最佳解
        self.best_solutions: List[Chromosome] = []
        self.history: List[Dict] = []  # 历史记录

        # 可视化数据
        self.visualization_data = {
            'generations': [],
            'avg_makespan': [],
            'avg_load_balance': [],
            'avg_tardiness': [],
            'best_makespan': [],
            'best_load_balance': [],
            'best_tardiness': []
        }

    # 染色体初始化

    def initialize_population(self):
        """初始化种群"""
        self.population = []

        for _ in range(self.population_size):
            # 1. 生成工序序列（基于工序的编码）
            operation_sequence = []

            # 创建每个作业的工序列表副本
            job_ops_remaining = {job_id: ops.copy() for job_id, ops in self.job_ops_map.items()}

            # 随机生成工序序列
            while any(job_ops_remaining.values()):
                # 从还有剩余工序的作业中随机选择一个
                available_jobs = [job_id for job_id, ops in job_ops_remaining.items() if ops]
                if not available_jobs:
                    break

                selected_job = random.choice(available_jobs)
                # 选择该作业的下一个工序
                selected_op = job_ops_remaining[selected_job].pop(0)
                operation_sequence.append(selected_op)

            # 2. 生成机器分配序列
            machine_assignment = []
            for job_id, op_id in operation_sequence:
                # 获取工序对象
                job = self.jobs[job_id]
                op = job.operations[op_id]

                # 从可用的机器中随机选择一个
                available_machines = list(op.machine_times.keys())
                if available_machines:
                    selected_machine = random.choice(available_machines)
                else:
                    # 如果没有可用机器，随机选择一个（虽然这种情况不应该发生）
                    selected_machine = random.choice(self.machines)

                machine_assignment.append(selected_machine)

            # 创建染色体
            chromosome = Chromosome(operation_sequence, machine_assignment)
            self.population.append(chromosome)

    # 解码与目标函数计算

    def decode_chromosome(self, chromosome: Chromosome) -> Tuple[Dict, Dict, Dict]:
        """
        将染色体解码为调度方案

        Returns:
            Tuple[operation_schedule, machine_schedule, job_schedule]
        """
        # 初始化数据结构
        machine_schedule = {machine_id: [] for machine_id in self.machines}
        job_completion_times = {job.job_id: 0 for job in self.jobs}
        operation_completion_times = {}

        # 处理每个工序
        for i, ((job_id, op_id), machine_id) in enumerate(zip(
                chromosome.operation_sequence, chromosome.machine_assignment)):

            # 获取工序对象
            job = self.jobs[job_id]
            op = job.operations[op_id]

            # 获取加工时间
            processing_time = op.get_processing_time(machine_id)
            if processing_time == float('inf'):
                # 如果机器不可用，使用最小加工时间
                available_machines = list(op.machine_times.keys())
                if available_machines:
                    machine_id = available_machines[0]
                    processing_time = op.get_processing_time(machine_id)
                else:
                    processing_time = 100  # 默认值

            # 计算开始时间
            # 1. 作业内约束：前一道工序的完成时间
            job_ready_time = 0
            if op_id > 0:
                prev_op_key = (job_id, op_id - 1)
                if prev_op_key in operation_completion_times:
                    job_ready_time = operation_completion_times[prev_op_key]

            # 2. 机器约束：机器的可用时间
            machine_available_time = 0
            if machine_schedule[machine_id]:
                machine_available_time = max(end for _, end, _ in machine_schedule[machine_id])

            # 实际开始时间
            start_time = max(job_ready_time, machine_available_time)
            end_time = start_time + processing_time

            # 记录调度信息
            machine_schedule[machine_id].append((start_time, end_time, (job_id, op_id)))
            operation_completion_times[(job_id, op_id)] = end_time

            # 更新作业完成时间
            if op_id == len(job.operations) - 1:
                job_completion_times[job_id] = end_time

        return operation_completion_times, machine_schedule, job_completion_times

    def evaluate_chromosome(self, chromosome: Chromosome) -> List[float]:
        """评估染色体，计算三个目标函数值"""
        _, machine_schedule, job_completion_times = self.decode_chromosome(chromosome)

        # 1. 最大完工时间 (Makespan)
        all_completion_times = []
        for schedules in machine_schedule.values():
            for start, end, _ in schedules:
                all_completion_times.append(end)

        if not all_completion_times:
            makespan = 0
        else:
            makespan = max(all_completion_times)

        # 2. 机器负载均衡度
        machine_loads = []
        for machine_id, schedules in machine_schedule.items():
            total_load = sum(end - start for start, end, _ in schedules)
            machine_loads.append(total_load)

        if not machine_loads:
            load_balance = 0
        else:
            avg_load = np.mean(machine_loads)
            if avg_load == 0:
                load_balance = 0
            else:
                # 使用变异系数（标准差/均值）作为均衡度指标
                std_load = np.std(machine_loads)
                load_balance = std_load / avg_load

        # 3. 总拖期时间
        total_tardiness = 0
        for job in self.jobs:
            completion_time = job_completion_times.get(job.job_id, 0)
            tardiness = max(0, completion_time - job.due_date)
            total_tardiness += tardiness

        return [makespan, load_balance, total_tardiness]

    def evaluate_population(self):
        """评估整个种群"""
        for chromosome in self.population:
            if chromosome.objectives is None:
                chromosome.objectives = self.evaluate_chromosome(chromosome)

    # NSGA-II核心算法

    def fast_non_dominated_sort(self, population: List[Chromosome]) -> List[List[Chromosome]]:
        """快速非支配排序"""
        fronts = [[]]

        # 为每个个体计算支配关系
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []

            for q in population:
                if p is q:
                    continue

                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []

            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def dominates(self, a: Chromosome, b: Chromosome) -> bool:
        """判断个体a是否支配个体b（三个目标都最小化）"""
        if a.objectives is None or b.objectives is None:
            return False

        # a支配b的条件：a在所有目标上都不比b差，且至少在一个目标上严格优于b
        not_worse = all(a_obj <= b_obj for a_obj, b_obj in zip(a.objectives, b.objectives))
        better = any(a_obj < b_obj for a_obj, b_obj in zip(a.objectives, b.objectives))

        return not_worse and better

    def calculate_crowding_distance(self, front: List[Chromosome]):
        """计算拥挤度距离"""
        if not front:
            return

        # 初始化拥挤度距离
        for individual in front:
            individual.crowding_distance = 0.0

        # 对每个目标函数计算
        for obj_idx in range(len(front[0].objectives)):
            # 按当前目标函数值排序
            front.sort(key=lambda x: x.objectives[obj_idx])

            # 边界个体的拥挤度设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # 获取目标函数值的范围
            min_obj = front[0].objectives[obj_idx]
            max_obj = front[-1].objectives[obj_idx]
            obj_range = max_obj - min_obj

            if obj_range == 0:
                continue

            # 计算中间个体的拥挤度
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                                                      front[i + 1].objectives[obj_idx] - front[i - 1].objectives[
                                                  obj_idx]
                                              ) / obj_range

    def selection(self, population: List[Chromosome]) -> List[Chromosome]:
        """锦标赛选择"""
        selected = []

        while len(selected) < len(population):
            # 随机选择两个个体
            candidates = random.sample(population, 2)

            # 锦标赛选择：优先选择非支配等级低的，如果等级相同选择拥挤度大的
            if candidates[0].rank < candidates[1].rank:
                selected.append(candidates[0])
            elif candidates[0].rank > candidates[1].rank:
                selected.append(candidates[1])
            else:
                # 等级相同，选择拥挤度大的
                if candidates[0].crowding_distance > candidates[1].crowding_distance:
                    selected.append(candidates[0])
                else:
                    selected.append(candidates[1])

        return selected

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """交叉操作：基于工序序列的交叉和基于机器分配的交叉"""

        # 1. 工序序列交叉（POX：Precedence Preserving Order-based Crossover）
        def pox_crossover(p1_seq, p2_seq):
            # 随机选择一些作业
            job_ids = list(self.job_ops_map.keys())
            selected_jobs = random.sample(job_ids, random.randint(1, len(job_ids) // 2))

            # 创建子代序列
            child1_seq = []
            child2_seq = []

            # 从parent1复制选中的作业的工序到child1，保持顺序
            job_positions_p1 = defaultdict(list)
            for idx, (job_id, op_id) in enumerate(p1_seq):
                job_positions_p1[job_id].append((idx, (job_id, op_id)))

            job_positions_p2 = defaultdict(list)
            for idx, (job_id, op_id) in enumerate(p2_seq):
                job_positions_p2[job_id].append((idx, (job_id, op_id)))

            # 构建child1
            # 先复制选中的作业的工序（从parent1）
            for job_id in selected_jobs:
                for _, op in job_positions_p1[job_id]:
                    child1_seq.append(op)

            # 复制未选中的作业的工序（从parent2），保持顺序
            for op in p2_seq:  # 修复：直接遍历op，而不是解包
                job_id = op[0]
                if job_id not in selected_jobs:
                    child1_seq.append(op)

            # 构建child2
            # 先复制选中的作业的工序（从parent2）
            for job_id in selected_jobs:
                for _, op in job_positions_p2[job_id]:
                    child2_seq.append(op)

            # 复制未选中的作业的工序（从parent1），保持顺序
            for op in p1_seq:  # 修复：直接遍历op，而不是解包
                job_id = op[0]
                if job_id not in selected_jobs:
                    child2_seq.append(op)

            return child1_seq, child2_seq

        # 2. 机器分配交叉（均匀交叉）
        def uniform_crossover(p1_machines, p2_machines):
            child1_machines = []
            child2_machines = []

            for m1, m2 in zip(p1_machines, p2_machines):
                if random.random() < 0.5:
                    child1_machines.append(m1)
                    child2_machines.append(m2)
                else:
                    child1_machines.append(m2)
                    child2_machines.append(m1)

            return child1_machines, child2_machines

        # 执行交叉
        if random.random() < self.crossover_rate:
            child1_seq, child2_seq = pox_crossover(parent1.operation_sequence, parent2.operation_sequence)
            child1_machines, child2_machines = uniform_crossover(parent1.machine_assignment, parent2.machine_assignment)
        else:
            # 不交叉，直接复制
            child1_seq, child2_seq = parent1.operation_sequence.copy(), parent2.operation_sequence.copy()
            child1_machines, child2_machines = parent1.machine_assignment.copy(), parent2.machine_assignment.copy()

        # 创建子代染色体
        child1 = Chromosome(child1_seq, child1_machines)
        child2 = Chromosome(child2_seq, child2_machines)

        return child1, child2

    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """变异操作"""
        mutated = chromosome.copy()

        # 1. 工序序列变异：交换变异
        if random.random() < self.mutation_rate and len(mutated.operation_sequence) >= 2:
            # 随机选择两个不同位置
            idx1, idx2 = random.sample(range(len(mutated.operation_sequence)), 2)

            # 检查交换后是否满足工序约束
            op1 = mutated.operation_sequence[idx1]
            op2 = mutated.operation_sequence[idx2]

            # 如果两个工序属于不同作业，可以交换
            if op1[0] != op2[0]:
                mutated.operation_sequence[idx1], mutated.operation_sequence[idx2] = op2, op1

        # 2. 机器分配变异：随机重置
        if random.random() < self.mutation_rate:
            # 随机选择一个工序进行机器重置
            idx = random.randint(0, len(mutated.operation_sequence) - 1)
            job_id, op_id = mutated.operation_sequence[idx]

            # 获取工序对象
            job = self.jobs[job_id]
            op = job.operations[op_id]

            # 从可用的机器中随机选择一个
            available_machines = list(op.machine_times.keys())
            if available_machines:
                mutated.machine_assignment[idx] = random.choice(available_machines)

        return mutated

    def create_offspring(self, parents: List[Chromosome]) -> List[Chromosome]:
        """生成子代种群"""
        offspring = []

        # 确保父代数量为偶数
        if len(parents) % 2 != 0:
            parents = parents[:-1]

        # 交叉和变异
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # 交叉
            child1, child2 = self.crossover(parent1, parent2)

            # 变异
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            offspring.extend([child1, child2])

        return offspring

    def run_generation(self, generation: int):
        """运行一代进化"""
        # 评估种群
        self.evaluate_population()

        # 非支配排序
        self.fronts = self.fast_non_dominated_sort(self.population)

        # 计算拥挤度距离
        for front in self.fronts:
            self.calculate_crowding_distance(front)

        # 记录历史数据
        self.record_generation_data(generation)

        # 选择父代
        parents = self.selection(self.population)

        # 生成子代
        offspring = self.create_offspring(parents)

        # 合并父代和子代
        combined_population = self.population + offspring

        # 评估合并的种群
        for chromosome in combined_population:
            if chromosome.objectives is None:
                chromosome.objectives = self.evaluate_chromosome(chromosome)

        # 对新种群进行非支配排序
        new_fronts = self.fast_non_dominated_sort(combined_population)

        # 构建下一代种群
        next_population = []
        front_idx = 0

        while len(next_population) + len(new_fronts[front_idx]) <= self.population_size:
            # 计算当前前沿的拥挤度距离
            self.calculate_crowding_distance(new_fronts[front_idx])

            # 添加整个前沿
            next_population.extend(new_fronts[front_idx])
            front_idx += 1

        # 如果还需要更多个体，按拥挤度距离排序当前前沿
        if len(next_population) < self.population_size:
            remaining_count = self.population_size - len(next_population)
            self.calculate_crowding_distance(new_fronts[front_idx])

            # 按拥挤度距离降序排序
            sorted_front = sorted(new_fronts[front_idx],
                                  key=lambda x: x.crowding_distance,
                                  reverse=True)

            next_population.extend(sorted_front[:remaining_count])

        # 更新种群
        self.population = next_population

        # 更新最佳解
        self.update_best_solutions()

    def record_generation_data(self, generation: int):
        """记录一代的数据用于可视化"""
        # 计算平均目标值
        if self.population:
            makespans = [c.objectives[0] for c in self.population]
            load_balances = [c.objectives[1] for c in self.population]
            tardinesses = [c.objectives[2] for c in self.population]

            avg_makespan = np.mean(makespans)
            avg_load_balance = np.mean(load_balances)
            avg_tardiness = np.mean(tardinesses)

            # 找到Pareto前沿
            pareto_front = self.fronts[0] if self.fronts else []
            if pareto_front:
                best_makespan = min(c.objectives[0] for c in pareto_front)
                best_load_balance = min(c.objectives[1] for c in pareto_front)
                best_tardiness = min(c.objectives[2] for c in pareto_front)
            else:
                best_makespan = min(makespans)
                best_load_balance = min(load_balances)
                best_tardiness = min(tardinesses)

            self.visualization_data['generations'].append(generation)
            self.visualization_data['avg_makespan'].append(avg_makespan)
            self.visualization_data['avg_load_balance'].append(avg_load_balance)
            self.visualization_data['avg_tardiness'].append(avg_tardiness)
            self.visualization_data['best_makespan'].append(best_makespan)
            self.visualization_data['best_load_balance'].append(best_load_balance)
            self.visualization_data['best_tardiness'].append(best_tardiness)

    def update_best_solutions(self):
        """更新最佳解（Pareto前沿）"""
        if self.fronts and self.fronts[0]:
            # 获取第一前沿（非支配解）
            pareto_front = self.fronts[0]

            # 清空并更新最佳解
            self.best_solutions = [c.copy() for c in pareto_front]

    # 主算法流程

    def run(self):
        """运行NSGA-II算法"""
        print("=" * 80)
        print("NSGA-II算法启动")
        print("=" * 80)
        print(f"种群大小: {self.population_size}")
        print(f"最大迭代代数: {self.max_generations}")
        print(f"总工序数: {self.total_operations}")
        print(f"作业数: {len(self.jobs)}")
        print(f"机器数: {len(self.machines)}")

        # 初始化种群
        print("\n初始化种群...")
        self.initialize_population()

        # 主循环
        print("\n开始进化...")
        for generation in range(self.max_generations):
            self.run_generation(generation)

            # 每10代打印进度
            if generation % 10 == 0:
                pareto_count = len(self.fronts[0]) if self.fronts and self.fronts[0] else 0
                avg_makespan = self.visualization_data['avg_makespan'][-1]
                print(f"  第{generation:3d}代: Pareto前沿解数={pareto_count:3d}, "
                      f"平均完工时间={avg_makespan:.2f}")

        # 最终评估
        self.evaluate_population()
        self.fronts = self.fast_non_dominated_sort(self.population)
        self.update_best_solutions()

        print("\n进化完成!")
        print(f"最终Pareto前沿解数: {len(self.best_solutions)}")

        return self.best_solutions

    # 结果输出与可视化（修改后）

    def print_results(self):
        """打印调度结果（中文输出）"""
        if not self.best_solutions:
            print("没有找到最优解")
            return

        print("=" * 80)
        print("NSGA-II调度结果 - Pareto前沿解")
        print("=" * 80)

        # 对Pareto前沿解排序（按完工时间）
        sorted_solutions = sorted(self.best_solutions,
                                  key=lambda x: x.objectives[0])

        for i, solution in enumerate(sorted_solutions[:5]):  # 只显示前5个解
            print(f"\n解 #{i + 1}:")
            print(f"  最大完工时间: {solution.objectives[0]:.2f}")
            print(f"  机器负载均衡度: {solution.objectives[1]:.4f}")
            print(f"  总拖期时间: {solution.objectives[2]:.2f}")

            # 解码并打印调度详情
            op_completion, machine_schedule, job_completion = self.decode_chromosome(solution)

            print(f"\n  作业完成时间:")
            for job in self.jobs:
                completion = job_completion.get(job.job_id, 0)
                tardiness = max(0, completion - job.due_date)
                status = "拖期" if tardiness > 0 else "准时"
                print(f"    作业{job.job_id}: 完成时间={completion:.1f}, "
                      f"交货期={job.due_date}, 拖期时间={tardiness:.1f}, 状态={status}")

    def visualize_convergence(self, save_path=None):
        """Visualize convergence process"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 平均目标收敛曲线
        gens = self.visualization_data['generations']

        # Makespan
        ax1 = axes[0, 0]
        ax1.plot(gens, self.visualization_data['avg_makespan'], 'b-', label='Average', alpha=0.7)
        ax1.plot(gens, self.visualization_data['best_makespan'], 'r-', label='Best', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Makespan')
        ax1.set_title('Makespan Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Load balance
        ax2 = axes[0, 1]
        ax2.plot(gens, self.visualization_data['avg_load_balance'], 'b-', label='Average', alpha=0.7)
        ax2.plot(gens, self.visualization_data['best_load_balance'], 'r-', label='Best', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Load Balance')
        ax2.set_title('Load Balance Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Total tardiness
        ax3 = axes[1, 0]
        ax3.plot(gens, self.visualization_data['avg_tardiness'], 'b-', label='Average', alpha=0.7)
        ax3.plot(gens, self.visualization_data['best_tardiness'], 'r-', label='Best', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Tardiness')
        ax3.set_title('Total Tardiness Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 2. 帕累托2D前沿散点图
        ax4 = axes[1, 1]

        if self.best_solutions:
            makespans = [s.objectives[0] for s in self.best_solutions]
            load_balances = [s.objectives[1] for s in self.best_solutions]
            tardinesses = [s.objectives[2] for s in self.best_solutions]

            # 对颜色进行归一化
            norm_makespan = (makespans - np.min(makespans)) / (np.max(makespans) - np.min(makespans) + 1e-10)

            scatter = ax4.scatter(makespans, load_balances, c=norm_makespan,
                                  cmap='viridis', s=50, alpha=0.7, edgecolors='k')

            # 标记最佳解决方案
            for i, (x, y) in enumerate(zip(makespans[:3], load_balances[:3])):
                ax4.annotate(f'#{i+1}', (x, y), fontsize=10, fontweight='bold')

            ax4.set_xlabel('Makespan')
            ax4.set_ylabel('Load Balance')
            ax4.set_title('Pareto Front (2D Projection)')
            ax4.grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=ax4, label='Normalized Makespan')

        plt.suptitle('NSGA-II Convergence and Pareto Front', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线图已保存至: {save_path}")

        plt.show()

    def visualize_pareto_front_3d(self, save_path=None):
        """帕累托前沿3D可视化"""
        if not self.best_solutions:
            print("没有Pareto前沿解可可视化")
            return

        # 提取目标值
        makespans = [s.objectives[0] for s in self.best_solutions]
        load_balances = [s.objectives[1] for s in self.best_solutions]
        tardinesses = [s.objectives[2] for s in self.best_solutions]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 创建散点图
        scatter = ax.scatter(makespans, load_balances, tardinesses,
                             c=tardinesses, cmap='viridis', s=50,
                             alpha=0.7, edgecolors='k')

        # 标记最佳方案
        for i, (x, y, z) in enumerate(zip(makespans[:5], load_balances[:5], tardinesses[:5])):
            ax.text(x, y, z, f'#{i+1}', fontsize=10, fontweight='bold')

        ax.set_xlabel('Makespan', fontsize=12)
        ax.set_ylabel('Load Balance', fontsize=12)
        ax.set_zlabel('Total Tardiness', fontsize=12)
        ax.set_title('3D Pareto Front', fontsize=14)

        # 增加颜色条
        plt.colorbar(scatter, ax=ax, label='Total Tardiness', pad=0.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"三维Pareto前沿图已保存至: {save_path}")

        plt.show()

    def visualize_best_schedule_gantt(self, solution_idx: int = 0, save_path=None):
        """可视化最佳调度甘特图"""
        if not self.best_solutions or solution_idx >= len(self.best_solutions):
            print(f"没有可用的解 #{solution_idx}")
            return

        solution = self.best_solutions[solution_idx]
        _, machine_schedule, _ = self.decode_chromosome(solution)

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.jobs)))
        job_colors = {job.job_id: colors[i] for i, job in enumerate(self.jobs)}

        # 为每一台机器分配调度
        for machine_idx, machine_id in enumerate(sorted(machine_schedule.keys())):
            schedules = machine_schedule[machine_id]
            schedules.sort(key=lambda x: x[0])  # 按开始时间排序

            for start, end, (job_id, op_id) in schedules:
                color = job_colors[job_id]
                ax.barh(machine_idx, end - start, left=start, height=0.6,
                        color=color, edgecolor='black')
                # 工序标签移动

        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_yticks(range(len(machine_schedule)))
        ax.set_yticklabels([f'Machine {mid}' for mid in sorted(machine_schedule.keys())])
        ax.set_title(f'NSGA-II Best Schedule #{solution_idx+1} - Gantt Chart', fontsize=14)

        # 增加图例
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=job_colors[job.job_id], label=f'Job {job.job_id}')
                          for job in self.jobs]
        ax.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()
        plt.grid(axis='x', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"甘特图已保存至: {save_path}")

        plt.show()


# 文件读取模块

def read_fjsp_instance(file_path: str) -> Tuple[List[Job], List[int]]:
    """
    读取标准 MOFJSP 数据文件（扩展 Brandimarte 格式）
    文件格式：
        第一行: 作业数  机器数
        接下来 N 行: 每个作业的工序信息
            - 第一个整数: 该作业的工序数量 O
            - 后续 O 组: 每组以 k 开头，后跟 k 对 (机器ID, 加工时间)
        然后一行: N 个整数，每个作业的交货期
        可能还有额外行（如柔性矩阵等），忽略
    返回: (jobs列表, machine_ids列表)
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
    machine_ids = list(range(num_machines))

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
        job = Job(job_idx, operations, due_date)
        jobs.append(job)

    return jobs, machine_ids


# 主函数

def main():
    """主函数：读取文件，运行NSGA-II算法并展示结果"""
    print("=" * 80)
    print("NSGA-II算法：多目标柔性作业车间调度（文件驱动）")
    print("=" * 80)

    # 指定数据文件路径（可根据需要修改或通过命令行参数传入）
    file_path = "../mo_fjsp_instances/mo_fjsp_000_small_train.txt"  # 确保文件存在

    # 1. 从文件读取调度实例
    print("\n读取调度实例...")
    try:
        jobs, machine_ids = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print(f"作业数量: {len(jobs)}")
    print(f"机器数量: {len(machine_ids)}")

    total_ops = sum(len(job.operations) for job in jobs)
    print(f"总工序数量: {total_ops}")

    # 2. 创建NSGA-II调度器
    print("\n配置NSGA-II参数...")
    scheduler = NSGA2Scheduler(
        jobs=jobs,
        machines=machine_ids,
        population_size=50,  # 较小种群以加快运行速度
        max_generations=100,  # 较少代数用于演示
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    # 3. 运行NSGA-II算法
    print("\n运行NSGA-II算法...")
    best_solutions = scheduler.run()

    # 4. 打印结果
    print("\n" + "=" * 80)
    print("调度结果分析")
    print("=" * 80)
    scheduler.print_results()

    # 5. 可视化并保存图片到当前目录
    print("\n生成可视化图表...")
    scheduler.visualize_convergence(save_path="../Meta_Heuristic/convergence.png")
    scheduler.visualize_pareto_front_3d(save_path="../Meta_Heuristic/pareto_front_3d.png")
    scheduler.visualize_best_schedule_gantt(0, save_path="../Meta_Heuristic/gantt_chart.png")

    # 8. 性能统计
    if best_solutions:
        print("\n" + "=" * 80)
        print("Pareto前沿统计")
        print("=" * 80)

        makespans = [s.objectives[0] for s in best_solutions]
        load_balances = [s.objectives[1] for s in best_solutions]
        tardinesses = [s.objectives[2] for s in best_solutions]

        print(f"Pareto解数量: {len(best_solutions)}")
        print(f"最大完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
        print(f"机器负载均衡度范围: {min(load_balances):.4f} - {max(load_balances):.4f}")
        print(f"总拖期时间范围: {min(tardinesses):.2f} - {max(tardinesses):.2f}")

        # 找到各个目标上的最优解
        best_makespan_idx = np.argmin(makespans)
        best_balance_idx = np.argmin(load_balances)
        best_tardiness_idx = np.argmin(tardinesses)

        print(f"\n各目标最优解:")
        print(f"  最小完工时间解: 完工时间={makespans[best_makespan_idx]:.2f}, "
              f"均衡度={load_balances[best_makespan_idx]:.4f}, "
              f"拖期={tardinesses[best_makespan_idx]:.2f}")
        print(f"  最佳负载均衡解: 完工时间={makespans[best_balance_idx]:.2f}, "
              f"均衡度={load_balances[best_balance_idx]:.4f}, "
              f"拖期={tardinesses[best_balance_idx]:.2f}")
        print(f"  最小拖期时间解: 完工时间={makespans[best_tardiness_idx]:.2f}, "
              f"均衡度={load_balances[best_tardiness_idx]:.4f}, "
              f"拖期={tardinesses[best_tardiness_idx]:.2f}")

    print("\nNSGA-II调度完成!")


# 运行主函数

if __name__ == "__main__":
    main()