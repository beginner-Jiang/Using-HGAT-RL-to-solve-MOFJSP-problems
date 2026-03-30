"""
MOFJSP 对比数据集生成程序
配置文件固定为 config2.json
"""

import json
import random
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any

class MOFJSPComparisonGenerator:
    def __init__(self, config_file: str = "config2.json"):
        """初始化生成器"""
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，将创建默认对比数据集配置...")
            self.create_default_config(config_file)

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 创建输出目录
        output_dir = self.config['output_config']['output_directory']
        os.makedirs(output_dir, exist_ok=True)

        # 设置随机种子以保证可重复性
        random.seed(42)
        np.random.seed(42)

        print(f"MOFJSP 对比数据集生成器已初始化")
        print(f"配置文件: {config_file}")
        print(f"输出目录: {output_dir}")
        print(f"将生成 {self.config['comparison_config']['total_instances']} 个实例")
        print("多目标数据格式: 作业数据之后，倒数第2行交货期，倒数第1行机器能力等级")

    def create_default_config(self, config_file: str):
        """创建默认对比数据集配置文件 (config2.json)"""
        default_config = {
            "comparison_config": {
                "description": "对比实验专用数据集，固定规模每种10个",
                "fixed_sizes": [
                    {"name": "small", "jobs": 10, "machines": 5, "count": 10},
                    {"name": "small", "jobs": 15, "machines": 10, "count": 10},
                    {"name": "small", "jobs": 20, "machines": 10, "count": 10},
                    {"name": "medium", "jobs": 25, "machines": 15, "count": 10},
                    {"name": "medium", "jobs": 30, "machines": 20, "count": 10},
                    {"name": "medium", "jobs": 40, "machines": 20, "count": 10},
                    {"name": "large", "jobs": 50, "machines": 25, "count": 10},
                    {"name": "large", "jobs": 75, "machines": 25, "count": 10},
                    {"name": "large", "jobs": 100, "machines": 30, "count": 10}
                ],
                "total_instances": 90
            },
            "generation_parameters": {
                "operations_per_job_range": [3, 8],
                "processing_time_range": [10, 50],
                "due_date_factor_range": [1.2, 2.0],
                "machine_flexibility": 0.5,
                "setup_time_range": [0, 5],
                "energy_consumption_range": [0.5, 1.5],
                "machine_capability_levels": [1, 2, 3]
            },
            "output_config": {
                "output_directory": "./comparison_instances",
                "file_format": "txt",
                "include_due_dates_in_data": True,
                "include_machine_capabilities_in_data": True
            }
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        print(f"已创建默认对比数据集配置文件: {config_file}")

    def generate_instance(self, instance_id: int, n_jobs: int, n_machines: int) -> Dict[str, Any]:
        """生成单个MOFJSP实例 (复用原生成逻辑)"""
        # 生成每个作业的工序数
        ops_per_job = np.random.randint(
            self.config['generation_parameters']['operations_per_job_range'][0],
            self.config['generation_parameters']['operations_per_job_range'][1] + 1,
            size=n_jobs
        )
        total_operations = int(np.sum(ops_per_job))

        # 生成交货期（基于最短加工时间的倍数）
        due_dates = []
        for job_idx in range(n_jobs):
            min_processing = ops_per_job[job_idx] * self.config['generation_parameters']['processing_time_range'][0]
            due_factor = random.uniform(*self.config['generation_parameters']['due_date_factor_range'])
            due_date = int(min_processing * due_factor)
            due_dates.append(due_date)

        # 生成工序信息
        operations = []
        operation_id = 0
        for job_idx in range(n_jobs):
            job_operations = []
            for op_idx in range(ops_per_job[job_idx]):
                # 确定可用机器数量
                available_machines = self.select_available_machines(
                    n_machines,
                    self.config['generation_parameters']['machine_flexibility']
                )
                machine_times = []
                for machine_idx in available_machines:
                    processing_time = random.randint(
                        self.config['generation_parameters']['processing_time_range'][0],
                        self.config['generation_parameters']['processing_time_range'][1]
                    )
                    machine_times.append((machine_idx + 1, processing_time))  # 机器编号从1开始
                job_operations.append({
                    'id': operation_id,
                    'job_id': job_idx + 1,
                    'op_idx': op_idx + 1,
                    'available_machines': machine_times
                })
                operation_id += 1
            operations.append(job_operations)

        # 机器能力等级
        machine_capabilities = random.choices(
            self.config['generation_parameters']['machine_capability_levels'],
            k=n_machines
        )

        # 计算机器柔性度实际值
        total_possible_assignments = sum(
            len(op['available_machines'])
            for job in operations
            for op in job
        )
        actual_flexibility = total_possible_assignments / (total_operations * n_machines)

        return {
            'instance_id': instance_id,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'operations_per_job': ops_per_job.tolist(),
            'total_operations': total_operations,
            'due_dates': due_dates,
            'operations': operations,
            'machine_capabilities': machine_capabilities,
            'actual_flexibility': actual_flexibility,
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def select_available_machines(self, total_machines: int, flexibility: float) -> List[int]:
        """为工序选择可用机器"""
        min_available = max(1, int(total_machines * flexibility * 0.3))
        max_available = max(min_available, int(total_machines * flexibility * 1.2))
        n_available = random.randint(min_available, min(max_available, total_machines))
        available = random.sample(range(total_machines), n_available)
        return sorted(available)

    def format_instance_text(self, instance_info: Dict[str, Any]) -> str:
        """格式化为扩展 Brandimarte 格式"""
        lines = []
        # 第1行：作业数 机器数
        lines.append(f"{instance_info['n_jobs']} {instance_info['n_machines']}")

        # 作业信息行
        for job_idx, job_ops in enumerate(instance_info['operations']):
            job_line = [len(job_ops)]
            for op in job_ops:
                job_line.append(len(op['available_machines']))
                for machine_id, proc_time in op['available_machines']:
                    job_line.extend([machine_id, proc_time])
            lines.append(" ".join(map(str, job_line)))

        # 交货期行
        if self.config['output_config'].get('include_due_dates_in_data', True):
            lines.append(" ".join(map(str, instance_info['due_dates'])))

        # 机器能力等级行
        if self.config['output_config'].get('include_machine_capabilities_in_data', True):
            lines.append(" ".join(map(str, instance_info['machine_capabilities'])))

        # 元数据注释
        lines.append(f"# Instance ID: {instance_info['instance_id']}")
        lines.append(f"# Total operations: {instance_info['total_operations']}")
        lines.append(f"# Actual flexibility: {instance_info['actual_flexibility']:.3f}")
        lines.append(f"# Generated: {instance_info['generation_time']}")
        lines.append(f"# Format: Extended Brandimarte for MOFJSP (Comparison Dataset)")

        return "\n".join(lines)

    def save_instance(self, instance_info: Dict[str, Any], size_name: str):
        """保存实例到文件，文件名包含规模信息"""
        filename = f"comp_{size_name}_{instance_info['n_jobs']}x{instance_info['n_machines']}_{instance_info['instance_id']:03d}.txt"
        filepath = os.path.join(
            self.config['output_config']['output_directory'],
            filename
        )
        instance_text = self.format_instance_text(instance_info)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(instance_text)
        print(f"已保存: {filename}")
        return filepath

    def generate_comparison_dataset(self):
        """生成对比数据集：按固定规模列表逐个生成"""
        fixed_sizes = self.config['comparison_config']['fixed_sizes']
        instance_counter = 1  # 从1开始编号，便于文件名连续

        print("\n开始生成对比数据集...")
        for size_entry in fixed_sizes:
            size_name = size_entry['name']
            jobs = size_entry['jobs']
            machines = size_entry['machines']
            count = size_entry['count']

            print(f"\n生成 {size_name} 规模 {jobs}×{machines} 共 {count} 个实例...")
            for i in range(count):
                inst = self.generate_instance(instance_counter, jobs, machines)
                self.save_instance(inst, size_name)
                # 简单打印摘要
                print(f"  实例 {instance_counter:03d}: 工序数={inst['total_operations']}, "
                      f"交货期范围={min(inst['due_dates'])}-{max(inst['due_dates'])}")
                instance_counter += 1

        # 生成统计信息
        self.generate_statistics()
        print(f"\n对比数据集生成完成！共 {instance_counter-1} 个实例保存在 {self.config['output_config']['output_directory']}")

    def generate_statistics(self):
        """生成数据集统计文件"""
        stats_file = os.path.join(
            self.config['output_config']['output_directory'],
            "comparison_dataset_statistics.txt"
        )
        lines = [
            "MOFJSP 对比数据集统计信息",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总实例数: {self.config['comparison_config']['total_instances']}",
            "固定规模列表:",
        ]
        for sz in self.config['comparison_config']['fixed_sizes']:
            lines.append(f"  {sz['name']}: {sz['jobs']}×{sz['machines']} × {sz['count']}")

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"统计信息已保存至: {stats_file}")

def main():
    generator = MOFJSPComparisonGenerator("config2.json")
    generator.generate_comparison_dataset()

if __name__ == "__main__":
    main()