"""
元启发式算法实现：基于分解的多目标进化算法-MOEA/D
"""
import numpy as np
import random
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import copy
from scipy.spatial.distance import cdist
import itertools
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
class Individual:
    """个体类，表示一个调度解"""
    operation_sequence: List[Tuple[int, int]]  # 工序序列 [(job_id, op_id), ...]
    machine_assignment: List[int]  # 机器分配序列，与工序序列一一对应
    objectives: List[float] = None  # 目标函数值 [makespan, load_balance, total_tardiness]

    def copy(self):
        """创建个体的深拷贝"""
        return Individual(
            operation_sequence=self.operation_sequence.copy(),
            machine_assignment=self.machine_assignment.copy(),
            objectives=self.objectives.copy() if self.objectives else None
        )

    def __len__(self):
        """返回个体长度（工序数量）"""
        return len(self.operation_sequence)


class MOEADScheduler:
    """基于MOEA/D算法的调度器"""

    def __init__(self, jobs: List[Job], machines: List[int],
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 neighbor_size: int = 20,
                 decomposition_method: str = 'te'):
        """
        初始化MOEA/D调度器

        Args:
            jobs: 作业列表
            machines: 机器ID列表
            population_size: 种群大小
            max_generations: 最大迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            neighbor_size: 邻居大小
            decomposition_method: 分解方法 ('te': 切比雪夫, 'ws': 加权和, 'pbi': 惩罚边界交方法)
        """
        self.jobs = jobs
        self.machines = machines
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.neighbor_size = min(neighbor_size, population_size)
        self.decomposition_method = decomposition_method

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

        # MOEA/D特定数据结构
        self.population: List[Individual] = []  # 种群
        self.weight_vectors: List[List[float]] = []  # 权重向量
        self.neighbors: List[List[int]] = []  # 邻居索引
        self.ideal_point: List[float] = [float('inf'), float('inf'), float('inf')]  # 理想点
        self.nadir_point: List[float] = [-float('inf'), -float('inf'), -float('inf')]  # 最差点
        self.ep: List[Individual] = []  # 外部种群（存储找到的Pareto最优解）

        # 记录历史数据
        self.visualization_data = {
            'generations': [],
            'avg_makespan': [],
            'avg_load_balance': [],
            'avg_tardiness': [],
            'best_makespan': [],
            'best_load_balance': [],
            'best_tardiness': [],
            'ideal_point': [],
            'nadir_point': []
        }

        # PBI方法的惩罚参数
        self.theta = 5.0

        # 初始化权重向量
        self.initialize_weight_vectors()

    # 权重向量初始化

    def initialize_weight_vectors(self):
        """初始化权重向量（三个目标）"""
        # 生成均匀分布的权重向量
        if self.population_size < 10:
            # 如果种群大小很小，直接生成随机权重
            self.weight_vectors = []
            for _ in range(self.population_size):
                w = np.random.rand(3)
                w = w / np.sum(w)
                self.weight_vectors.append(w.tolist())
        else:
            # 使用系统方法生成均匀权重向量
            self.weight_vectors = self.generate_weight_vectors_systematic()

        # 确保权重向量数量与种群大小一致
        if len(self.weight_vectors) > self.population_size:
            self.weight_vectors = self.weight_vectors[:self.population_size]
        elif len(self.weight_vectors) < self.population_size:
            # 如果生成的权重不够，补充随机权重
            while len(self.weight_vectors) < self.population_size:
                w = np.random.rand(3)
                w = w / np.sum(w)
                self.weight_vectors.append(w.tolist())

        # 计算每个权重向量的邻居
        self.calculate_neighbors()

    def generate_weight_vectors_systematic(self):
        """系统生成均匀分布的权重向量"""
        weight_vectors = []

        # 对于三个目标，我们可以使用两层嵌套循环生成权重
        # H值决定了权重向量的密度
        H = int(np.cbrt(self.population_size * 6))  # 近似计算H值

        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                if k >= 0:
                    w1 = i / H
                    w2 = j / H
                    w3 = k / H

                    # 确保权重和为1
                    total = w1 + w2 + w3
                    if total > 0:
                        w1 /= total
                        w2 /= total
                        w3 /= total
                        weight_vectors.append([w1, w2, w3])

        return weight_vectors

    def calculate_neighbors(self):
        """计算每个权重向量的邻居（基于欧氏距离）"""
        # 将权重向量转换为numpy数组
        weights_array = np.array(self.weight_vectors)

        # 计算所有权重向量之间的距离
        distances = cdist(weights_array, weights_array, 'euclidean')

        # 对每个权重向量，找到最近的neighbor_size个邻居
        self.neighbors = []
        for i in range(len(self.weight_vectors)):
            # 获取距离排序后的索引（排除自己）
            sorted_indices = np.argsort(distances[i])
            # 排除自身（距离为0）
            neighbor_indices = sorted_indices[1:self.neighbor_size + 1].tolist()
            self.neighbors.append(neighbor_indices)

    # 个体初始化

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
                    # 如果没有可用机器，随机选择一个
                    selected_machine = random.choice(self.machines)

                machine_assignment.append(selected_machine)

            # 创建个体
            individual = Individual(operation_sequence, machine_assignment)
            self.population.append(individual)

    # 解码与目标函数计算

    def decode_individual(self, individual: Individual) -> Tuple[Dict, Dict, Dict]:
        """
        将个体解码为调度方案

        Returns:
            Tuple[operation_schedule, machine_schedule, job_schedule]
        """
        # 初始化数据结构
        machine_schedule = {machine_id: [] for machine_id in self.machines}
        job_completion_times = {job.job_id: 0 for job in self.jobs}
        operation_completion_times = {}

        # 处理每个工序
        for i, ((job_id, op_id), machine_id) in enumerate(zip(
                individual.operation_sequence, individual.machine_assignment)):

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

    def evaluate_individual(self, individual: Individual) -> List[float]:
        """评估个体，计算三个目标函数值"""
        _, machine_schedule, job_completion_times = self.decode_individual(individual)

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
        """评估整个种群，并更新理想点和最差点"""
        for i, individual in enumerate(self.population):
            if individual.objectives is None:
                individual.objectives = self.evaluate_individual(individual)

            # 更新理想点（最小值）
            for obj_idx in range(3):
                if individual.objectives[obj_idx] < self.ideal_point[obj_idx]:
                    self.ideal_point[obj_idx] = individual.objectives[obj_idx]

            # 更新最差点（最大值）
            for obj_idx in range(3):
                if individual.objectives[obj_idx] > self.nadir_point[obj_idx]:
                    self.nadir_point[obj_idx] = individual.objectives[obj_idx]

    # 分解方法

    def te_chebycheff(self, objectives: List[float], weight: List[float]) -> float:
        """切比雪夫分解方法"""
        # 归一化目标值
        normalized_obj = []
        for obj_idx in range(3):
            if self.nadir_point[obj_idx] - self.ideal_point[obj_idx] > 0:
                norm = (objectives[obj_idx] - self.ideal_point[obj_idx]) / (
                        self.nadir_point[obj_idx] - self.ideal_point[obj_idx])
            else:
                norm = 0
            normalized_obj.append(norm)

        # 计算切比雪夫值
        max_val = -float('inf')
        for obj_idx in range(3):
            val = weight[obj_idx] * abs(normalized_obj[obj_idx])
            if val > max_val:
                max_val = val

        return max_val

    def weighted_sum(self, objectives: List[float], weight: List[float]) -> float:
        """加权和分解方法"""
        # 归一化目标值
        normalized_obj = []
        for obj_idx in range(3):
            if self.nadir_point[obj_idx] - self.ideal_point[obj_idx] > 0:
                norm = (objectives[obj_idx] - self.ideal_point[obj_idx]) / (
                        self.nadir_point[obj_idx] - self.ideal_point[obj_idx])
            else:
                norm = 0
            normalized_obj.append(norm)

        # 计算加权和
        ws_val = 0
        for obj_idx in range(3):
            ws_val += weight[obj_idx] * normalized_obj[obj_idx]

        return ws_val

    def penalty_boundary_intersection(self, objectives: List[float], weight: List[float]) -> float:
        """惩罚边界交方法 (PBI)"""
        # 归一化目标值
        normalized_obj = []
        for obj_idx in range(3):
            if self.nadir_point[obj_idx] - self.ideal_point[obj_idx] > 0:
                norm = (objectives[obj_idx] - self.ideal_point[obj_idx]) / (
                        self.nadir_point[obj_idx] - self.ideal_point[obj_idx])
            else:
                norm = 0
            normalized_obj.append(norm)

        # 归一化权重向量
        norm_weight = np.linalg.norm(weight)
        if norm_weight > 0:
            normalized_weight = [w / norm_weight for w in weight]
        else:
            normalized_weight = weight

        # 计算d1和d2
        d1 = 0
        for obj_idx in range(3):
            d1 += normalized_obj[obj_idx] * normalized_weight[obj_idx]

        obj_array = np.array(normalized_obj)
        weight_array = np.array(normalized_weight)
        d2 = np.linalg.norm(obj_array - d1 * weight_array)

        # PBI值 = d1 + theta * d2
        pbi_val = d1 + self.theta * d2

        return pbi_val

    def decomposition_function(self, objectives: List[float], weight: List[float]) -> float:
        """根据选择的分解方法计算分解函数值"""
        if self.decomposition_method == 'te':
            return self.te_chebycheff(objectives, weight)
        elif self.decomposition_method == 'ws':
            return self.weighted_sum(objectives, weight)
        elif self.decomposition_method == 'pbi':
            return self.penalty_boundary_intersection(objectives, weight)
        else:
            raise ValueError(f"未知的分解方法: {self.decomposition_method}")

    # 遗传操作

    def selection(self, neighbor_indices: List[int]) -> Tuple[int, int]:
        """从邻居中选择两个父代索引"""
        # 从邻居中随机选择两个不同的索引
        if len(neighbor_indices) >= 2:
            selected = random.sample(neighbor_indices, 2)
            return selected[0], selected[1]
        else:
            # 如果邻居太少，从整个种群中随机选择
            return random.randint(0, len(self.population) - 1), random.randint(0, len(self.population) - 1)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
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
            for op in p2_seq:
                job_id = op[0]
                if job_id not in selected_jobs:
                    child1_seq.append(op)

            # 构建child2
            # 先复制选中的作业的工序（从parent2）
            for job_id in selected_jobs:
                for _, op in job_positions_p2[job_id]:
                    child2_seq.append(op)

            # 复制未选中的作业的工序（从parent1），保持顺序
            for op in p1_seq:
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

        # 创建子代个体
        child1 = Individual(child1_seq, child1_machines)
        child2 = Individual(child2_seq, child2_machines)

        return child1, child2

    def mutation(self, individual: Individual) -> Individual:
        """变异操作"""
        mutated = individual.copy()

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

    def generate_offspring(self, i: int) -> Individual:
        """为第i个子问题生成子代"""
        # 获取邻居索引
        neighbor_indices = self.neighbors[i]

        # 选择父代
        parent1_idx, parent2_idx = self.selection(neighbor_indices)
        parent1 = self.population[parent1_idx]
        parent2 = self.population[parent2_idx]

        # 交叉
        child1, child2 = self.crossover(parent1, parent2)

        # 随机选择一个子代进行变异
        if random.random() < 0.5:
            child = child1
        else:
            child = child2

        # 变异
        child = self.mutation(child)

        return child

    # 更新操作

    def update_ideal_point(self, objectives: List[float]):
        """用新解的目标值更新理想点"""
        for obj_idx in range(3):
            if objectives[obj_idx] < self.ideal_point[obj_idx]:
                self.ideal_point[obj_idx] = objectives[obj_idx]

    def update_nadir_point(self, objectives: List[float]):
        """用新解的目标值更新最差点"""
        for obj_idx in range(3):
            if objectives[obj_idx] > self.nadir_point[obj_idx]:
                self.nadir_point[obj_idx] = objectives[obj_idx]

    def update_neighbors(self, i: int, new_individual: Individual):
        """用新个体更新邻居解"""
        new_objectives = new_individual.objectives

        # 获取邻居索引
        neighbor_indices = self.neighbors[i]

        # 更新每个邻居
        for j in neighbor_indices:
            old_individual = self.population[j]
            old_objectives = old_individual.objectives

            # 计算新旧解在子问题j上的分解函数值
            weight_j = self.weight_vectors[j]
            old_value = self.decomposition_function(old_objectives, weight_j)
            new_value = self.decomposition_function(new_objectives, weight_j)

            # 如果新解更好，替换旧解
            if new_value < old_value:
                self.population[j] = new_individual.copy()

    def update_external_population(self, individual: Individual):
        """更新外部种群（存储Pareto最优解）"""
        # 检查是否支配外部种群中的解
        to_remove = []
        is_dominated = False

        for ep_idx, ep_individual in enumerate(self.ep):
            # 检查是否被支配
            if self.dominates(ep_individual.objectives, individual.objectives):
                is_dominated = True
                break

            # 检查是否支配外部种群中的解
            if self.dominates(individual.objectives, ep_individual.objectives):
                to_remove.append(ep_idx)

        # 如果新解不被任何外部种群中的解支配，则加入外部种群
        if not is_dominated:
            # 先移除被支配的解
            for idx in sorted(to_remove, reverse=True):
                self.ep.pop(idx)

            # 加入新解
            self.ep.append(individual.copy())

            # 限制外部种群大小
            if len(self.ep) > self.population_size:
                # 使用拥挤度距离进行筛选
                self.prune_external_population()

    def dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """判断目标值obj1是否支配obj2"""
        # obj1支配obj2的条件：obj1在所有目标上都不比obj2差，且至少在一个目标上严格优于obj2
        not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))

        return not_worse and better

    def prune_external_population(self):
        """修剪外部种群，保持多样性"""
        if len(self.ep) <= self.population_size:
            return

        # 计算拥挤度距离
        self.calculate_crowding_distance_ep()

        # 按拥挤度距离排序，保留距离大的解
        self.ep.sort(key=lambda x: x.crowding_distance, reverse=True)
        self.ep = self.ep[:self.population_size]

    def calculate_crowding_distance_ep(self):
        """计算外部种群中个体的拥挤度距离"""
        if not self.ep:
            return

        # 初始化拥挤度距离
        for individual in self.ep:
            individual.crowding_distance = 0.0

        # 对每个目标函数计算
        for obj_idx in range(3):
            # 按当前目标函数值排序
            self.ep.sort(key=lambda x: x.objectives[obj_idx])

            # 边界个体的拥挤度设为无穷大
            self.ep[0].crowding_distance = float('inf')
            self.ep[-1].crowding_distance = float('inf')

            # 获取目标函数值的范围
            min_obj = self.ep[0].objectives[obj_idx]
            max_obj = self.ep[-1].objectives[obj_idx]
            obj_range = max_obj - min_obj

            if obj_range == 0:
                continue

            # 计算中间个体的拥挤度
            for i in range(1, len(self.ep) - 1):
                self.ep[i].crowding_distance += (
                                                        self.ep[i + 1].objectives[obj_idx] - self.ep[i - 1].objectives[
                                                    obj_idx]
                                                ) / obj_range

    # 主算法流程

    def run_generation(self, generation: int):
        """运行一代进化"""
        # 评估种群并更新理想点和最差点
        self.evaluate_population()

        # 记录历史数据
        self.record_generation_data(generation)

        # 对每个子问题
        for i in range(self.population_size):
            # 生成子代
            child = self.generate_offspring(i)

            # 评估子代
            child.objectives = self.evaluate_individual(child)

            # 更新理想点和最差点
            self.update_ideal_point(child.objectives)
            self.update_nadir_point(child.objectives)

            # 更新外部种群
            self.update_external_population(child)

            # 更新邻居
            self.update_neighbors(i, child)

    def record_generation_data(self, generation: int):
        """记录一代的数据用于可视化"""
        # 计算平均目标值
        if self.population:
            makespans = [ind.objectives[0] for ind in self.population if ind.objectives is not None]
            load_balances = [ind.objectives[1] for ind in self.population if ind.objectives is not None]
            tardinesses = [ind.objectives[2] for ind in self.population if ind.objectives is not None]

            if makespans:
                avg_makespan = np.mean(makespans)
                avg_load_balance = np.mean(load_balances)
                avg_tardiness = np.mean(tardinesses)

                # 找到最佳解
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
                self.visualization_data['ideal_point'].append(self.ideal_point.copy())
                self.visualization_data['nadir_point'].append(self.nadir_point.copy())

    def run(self):
        """运行MOEA/D算法"""
        print("=" * 80)
        print("MOEA/D算法启动")
        print("=" * 80)
        print(f"种群大小: {self.population_size}")
        print(f"最大迭代代数: {self.max_generations}")
        print(f"分解方法: {self.decomposition_method}")
        print(f"邻居大小: {self.neighbor_size}")
        print(f"总工序数: {self.total_operations}")
        print(f"作业数: {len(self.jobs)}")
        print(f"机器数: {len(self.machines)}")

        # 初始化种群
        print("\n初始化种群...")
        self.initialize_population()

        # 初始评估
        self.evaluate_population()

        # 初始外部种群
        for individual in self.population:
            self.update_external_population(individual)

        # 主循环
        print("\n开始进化...")
        for generation in range(self.max_generations):
            self.run_generation(generation)

            # 每10代打印进度
            if generation % 10 == 0:
                ep_count = len(self.ep)
                avg_makespan = self.visualization_data['avg_makespan'][-1] if self.visualization_data[
                    'avg_makespan'] else 0
                print(f"  第{generation:3d}代: 外部种群解数={ep_count:3d}, "
                      f"平均完工时间={avg_makespan:.2f}")

        # 最终评估
        self.evaluate_population()

        print("\n进化完成!")
        print(f"最终外部种群解数: {len(self.ep)}")

        return self.ep

    # 结果输出与可视化

    def print_results(self):
        """打印调度结果（中文输出）"""
        if not self.ep:
            print("没有找到最优解")
            return

        print("=" * 80)
        print("MOEA/D调度结果 - 外部种群解")
        print("=" * 80)

        # 对解排序（按完工时间）
        sorted_solutions = sorted(self.ep,
                                  key=lambda x: x.objectives[0])

        for i, solution in enumerate(sorted_solutions[:5]):  # 只显示前5个解
            print(f"\n解 #{i + 1}:")
            print(f"  最大完工时间: {solution.objectives[0]:.2f}")
            print(f"  机器负载均衡度: {solution.objectives[1]:.4f}")
            print(f"  总拖期时间: {solution.objectives[2]:.2f}")

            # 解码并打印调度详情
            op_completion, machine_schedule, job_completion = self.decode_individual(solution)

            print(f"\n  作业完成时间:")
            for job in self.jobs:
                completion = job_completion.get(job.job_id, 0)
                tardiness = max(0, completion - job.due_date)
                status = "拖期" if tardiness > 0 else "准时"
                print(f"    作业{job.job_id}: 完成时间={completion:.1f}, "
                      f"交货期={job.due_date}, 拖期时间={tardiness:.1f}, 状态={status}")

    def visualize_convergence(self, save_path=None):
        """可视化收敛过程"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. 平均目标收敛曲线
        gens = self.visualization_data['generations']

        if not gens:
            print("No convergence data to visualize")
            return

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
        ax3 = axes[0, 2]
        ax3.plot(gens, self.visualization_data['avg_tardiness'], 'b-', label='Average', alpha=0.7)
        ax3.plot(gens, self.visualization_data['best_tardiness'], 'r-', label='Best', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Tardiness')
        ax3.set_title('Total Tardiness Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 2. 理想点收敛
        ax4 = axes[1, 0]
        ideal_makespan = [point[0] for point in self.visualization_data['ideal_point']]
        ax4.plot(gens, ideal_makespan, 'g-', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Ideal Point - Makespan')
        ax4.set_title('Ideal Point (Makespan) Convergence')
        ax4.grid(True, alpha=0.3)

        # 3. 天底点变化
        ax5 = axes[1, 1]
        nadir_makespan = [point[0] for point in self.visualization_data['nadir_point']]
        ax5.plot(gens, nadir_makespan, 'm-', linewidth=2)
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Nadir Point - Makespan')
        ax5.set_title('Nadir Point (Makespan) Change')
        ax5.grid(True, alpha=0.3)

        # 4. 帕累托前沿二维散点图
        ax6 = axes[1, 2]

        if self.ep:
            makespans = [s.objectives[0] for s in self.ep]
            load_balances = [s.objectives[1] for s in self.ep]
            tardinesses = [s.objectives[2] for s in self.ep]

            # 为着色而归一化
            norm_makespan = (makespans - np.min(makespans)) / (np.max(makespans) - np.min(makespans) + 1e-10)

            scatter = ax6.scatter(makespans, load_balances, c=norm_makespan,
                                  cmap='viridis', s=50, alpha=0.7, edgecolors='k')

            # 标记最佳解决方案
            for i, (x, y) in enumerate(zip(makespans[:3], load_balances[:3])):
                ax6.annotate(f'#{i+1}', (x, y), fontsize=10, fontweight='bold')

            ax6.set_xlabel('Makespan')
            ax6.set_ylabel('Load Balance')
            ax6.set_title(f'MOEA/D Pareto Front ({self.decomposition_method})')
            ax6.grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=ax6, label='Normalized Makespan')

        plt.suptitle(f'MOEA/D Convergence and Pareto Front (Decomposition: {self.decomposition_method})', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线图已保存至: {save_path}")

        plt.show()

    def visualize_weight_vectors(self, save_path=None):
        """可视化权重向量分布"""
        if not self.weight_vectors:
            return

        weights = np.array(self.weight_vectors)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2],
                   c='blue', s=50, alpha=0.7, edgecolors='k')

        ax.set_xlabel('Weight 1 (Makespan)')
        ax.set_ylabel('Weight 2 (Load Balance)')
        ax.set_zlabel('Weight 3 (Tardiness)')
        ax.set_title('MOEA/D Weight Vector Distribution', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"权重向量图已保存至: {save_path}")

        plt.show()

    def visualize_pareto_front_3d(self, save_path=None):
        """可视化三维帕累托前沿"""
        if not self.ep:
            print("No Pareto front solutions to visualize")
            return

        makespans = [s.objectives[0] for s in self.ep]
        load_balances = [s.objectives[1] for s in self.ep]
        tardinesses = [s.objectives[2] for s in self.ep]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(makespans, load_balances, tardinesses,
                             c=tardinesses, cmap='viridis', s=50,
                             alpha=0.7, edgecolors='k')

        for i, (x, y, z) in enumerate(zip(makespans[:5], load_balances[:5], tardinesses[:5])):
            ax.text(x, y, z, f'#{i+1}', fontsize=10, fontweight='bold')

        ax.set_xlabel('Makespan', fontsize=12)
        ax.set_ylabel('Load Balance', fontsize=12)
        ax.set_zlabel('Total Tardiness', fontsize=12)
        ax.set_title(f'MOEA/D 3D Pareto Front ({self.decomposition_method})', fontsize=14)

        plt.colorbar(scatter, ax=ax, label='Total Tardiness', pad=0.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"三维Pareto前沿图已保存至: {save_path}")

        plt.show()

    def visualize_best_schedule_gantt(self, solution_idx: int = 0, save_path=None):
        """最佳调度甘特图可视化"""
        if not self.ep or solution_idx >= len(self.ep):
            print(f"解 #{solution_idx} 不存在")
            return

        solution = self.ep[solution_idx]
        _, machine_schedule, _ = self.decode_individual(solution)

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.jobs)))
        job_colors = {job.job_id: colors[i] for i, job in enumerate(self.jobs)}

        # 为每一台机器分配调度
        for machine_idx, machine_id in enumerate(sorted(machine_schedule.keys())):
            schedules = machine_schedule[machine_id]
            schedules.sort(key=lambda x: x[0])  # Sort by start time

            for start, end, (job_id, op_id) in schedules:
                color = job_colors[job_id]
                ax.barh(machine_idx, end - start, left=start, height=0.6,
                        color=color, edgecolor='black')
                # 工序标签移动

        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_yticks(range(len(machine_schedule)))
        ax.set_yticklabels([f'Machine {mid}' for mid in sorted(machine_schedule.keys())])
        ax.set_title(f'MOEA/D Best Schedule #{solution_idx+1} ({self.decomposition_method}) - Gantt Chart', fontsize=14)

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


# 主函数（中文输出）

def main():
    """主函数：读取文件，运行MOEA/D算法并展示结果"""
    print("=" * 80)
    print("MOEA/D算法：多目标柔性作业车间调度（文件驱动）")
    print("=" * 80)

    # 指定数据文件路径
    file_path = "../mo_fjsp_instances/mo_fjsp_000_small_train.txt"  # 确保文件存在

    # 1. 从文件读取调度实例
    print("\n读取调度实例...")
    try:
        jobs, machines = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    print(f"作业数量: {len(jobs)}")
    print(f"机器数量: {len(machines)}")

    total_ops = sum(len(job.operations) for job in jobs)
    print(f"总工序数量: {total_ops}")

    # 2. 创建MOEA/D调度器
    print("\n配置MOEA/D参数...")
    scheduler = MOEADScheduler(
        jobs=jobs,
        machines=machines,
        population_size=50,  # 较小种群以加快运行速度
        max_generations=100,  # 较少代数用于演示
        crossover_rate=0.9,
        mutation_rate=0.1,
        neighbor_size=10,
        decomposition_method='ws'  # 使用加权和分解方法
    )

    # 3. 运行MOEA/D算法
    print("\n运行MOEA/D算法...")
    best_solutions = scheduler.run()

    # 4. 打印结果
    print("\n" + "=" * 80)
    print("调度结果分析")
    print("=" * 80)
    scheduler.print_results()

    # 5. 可视化并保存图片到当前目录
    print("\n生成可视化图表...")
    scheduler.visualize_convergence(save_path="../Meta_Heuristic/convergence.png")
    scheduler.visualize_weight_vectors(save_path="../Meta_Heuristic/weight_vectors.png")
    scheduler.visualize_pareto_front_3d(save_path="../Meta_Heuristic/pareto_front_3d.png")
    scheduler.visualize_best_schedule_gantt(0, save_path="../Meta_Heuristic/gantt_chart.png")

    # 6. 性能统计
    if best_solutions:
        print("\n" + "=" * 80)
        print("外部种群统计")
        print("=" * 80)

        makespans = [s.objectives[0] for s in best_solutions]
        load_balances = [s.objectives[1] for s in best_solutions]
        tardinesses = [s.objectives[2] for s in best_solutions]

        print(f"外部种群解数量: {len(best_solutions)}")
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

    print("\nMOEA/D调度完成!")


# 运行主函数

if __name__ == "__main__":
    main()