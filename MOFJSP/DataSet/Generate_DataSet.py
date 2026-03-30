"""
MOFJSP数据集生成程序
多目标数据（交货期、机器能力）放在作业数据之后，注释行之前
JSON配置文件名为config.json
"""

import json
import random
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any
import argparse

class MOFJSPInstanceGenerator:
    def __init__(self, config_file: str = "config.json"):
        """初始化生成器"""
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，将使用默认配置创建...")
            self.create_default_config(config_file)

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 创建输出目录
        output_dir = self.config['output_config']['output_directory']
        os.makedirs(output_dir, exist_ok=True)

        # 设置随机种子以保证可重复性
        random.seed(42)
        np.random.seed(42)

        print(f"MOFJSP数据集生成器已初始化")
        print(f"配置文件: {config_file}")
        print(f"输出目录: {output_dir}")
        print(f"多目标数据格式: 作业数据之后，注释行之前")

    def create_default_config(self, config_file: str):
        """创建默认配置文件"""
        default_config = {
            "dataset_config": {
                "num_instances": 100,
                "train_val_test_split": [60, 20, 20],
                "instance_sizes": [
                    {
                        "name": "small",
                        "jobs_range": [10, 20],
                        "machines_range": [5, 10],
                        "num_instances": 40
                    },
                    {
                        "name": "medium",
                        "jobs_range": [21, 40],
                        "machines_range": [11, 20],
                        "num_instances": 30
                    },
                    {
                        "name": "large",
                        "jobs_range": [41, 100],
                        "machines_range": [21, 30],
                        "num_instances": 30
                    }
                ]
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
                "output_directory": "./mo_fjsp_instances",
                "file_format": "txt",
                "data_format": "extended_brandimarte",
                "include_due_dates_in_data": True,
                "include_machine_capabilities_in_data": True
            }
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        print(f"已创建默认配置文件: {config_file}")

    def generate_instance(self, instance_id: int, n_jobs: int, n_machines: int) -> Dict[str, Any]:
        """生成单个MOFJSP实例"""
        # 生成每个作业的工序数
        ops_per_job = np.random.randint(
            self.config['generation_parameters']['operations_per_job_range'][0],
            self.config['generation_parameters']['operations_per_job_range'][1] + 1,
            size=n_jobs
        )

        total_operations = np.sum(ops_per_job)

        # 生成交货期（基于最短加工时间的倍数）
        due_dates = []
        for job_idx in range(n_jobs):
            # 估算最短加工时间
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
                # 确定该工序可用的机器数量（基于柔性度）
                available_machines = self.select_available_machines(
                    n_machines,
                    self.config['generation_parameters']['machine_flexibility']
                )

                # 生成每台可用机器的加工时间
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

        # 计算机器柔性度的实际值
        total_possible_assignments = sum(
            len(op['available_machines'])
            for job in operations
            for op in job
        )
        actual_flexibility = total_possible_assignments / (total_operations * n_machines)

        instance_info = {
            'instance_id': instance_id,
            'size_type': self.get_size_type(n_jobs, n_machines),
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'operations_per_job': ops_per_job.tolist(),
            'total_operations': int(total_operations),
            'due_dates': due_dates,
            'operations': operations,
            'machine_capabilities': machine_capabilities,
            'actual_flexibility': actual_flexibility,
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return instance_info

    def get_size_type(self, n_jobs: int, n_machines: int) -> str:
        """根据作业数和机器数确定实例规模类型"""
        for size_config in self.config['dataset_config']['instance_sizes']:
            if (size_config['jobs_range'][0] <= n_jobs <= size_config['jobs_range'][1] and
                size_config['machines_range'][0] <= n_machines <= size_config['machines_range'][1]):
                return size_config['name']
        return "custom"

    def select_available_machines(self, total_machines: int, flexibility: float) -> List[int]:
        """为工序选择可用的机器"""
        # 确保至少有一台机器可用
        min_available = max(1, int(total_machines * flexibility * 0.3))
        max_available = max(min_available, int(total_machines * flexibility * 1.2))

        n_available = random.randint(min_available, min(max_available, total_machines))
        available = random.sample(range(total_machines), n_available)
        return sorted(available)

    def format_instance_text(self, instance_info: Dict[str, Any]) -> str:
        """
        将实例信息格式化为扩展Brandimarte格式
        格式：
          第1行：作业数 机器数
          接下来n_jobs行：每个作业的数据
          倒数第2行：交货期（正式数据行）
          倒数第1行：机器能力等级（正式数据行）
          最后几行：元数据（注释行）
        """
        lines = []

        # 第1行：作业数 机器数
        lines.append(f"{instance_info['n_jobs']} {instance_info['n_machines']}")

        # 接下来n_jobs行：作业信息
        for job_idx, job_ops in enumerate(instance_info['operations']):
            job_line = [len(job_ops)]  # 该作业的工序数

            for op in job_ops:
                job_line.append(len(op['available_machines']))  # 该工序的可选机器数
                for machine_id, proc_time in op['available_machines']:
                    job_line.extend([machine_id, proc_time])

            lines.append(" ".join(map(str, job_line)))

        # 倒数第2行：交货期（多目标数据）- 正式数据行
        if self.config['output_config'].get('include_due_dates_in_data', True):
            lines.append(" ".join(map(str, instance_info['due_dates'])))

        # 倒数第1行：机器能力等级（多目标数据）- 正式数据行
        if self.config['output_config'].get('include_machine_capabilities_in_data', True):
            lines.append(" ".join(map(str, instance_info['machine_capabilities'])))

        # 元数据作为注释（不影响程序读取）
        lines.append(f"# Instance ID: {instance_info['instance_id']}")
        lines.append(f"# Size type: {instance_info['size_type']}")
        lines.append(f"# Total operations: {instance_info['total_operations']}")
        lines.append(f"# Actual flexibility: {instance_info['actual_flexibility']:.3f}")
        lines.append(f"# Generated: {instance_info['generation_time']}")
        lines.append(f"# Format: Extended Brandimarte format for MOFJSP")

        return "\n".join(lines)

    def save_instance(self, instance_info: Dict[str, Any], split: str = "train"):
        """保存实例到文件"""
        filename = f"mo_fjsp_{instance_info['instance_id']:03d}_{instance_info['size_type']}_{split}.txt"
        filepath = os.path.join(
            self.config['output_config']['output_directory'],
            filename
        )

        instance_text = self.format_instance_text(instance_info)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(instance_text)

        print(f"已保存实例: {filename}")
        return filepath

    def read_instance(self, filepath: str) -> Dict[str, Any]:
        """
        读取实例文件
        格式说明：
          第1行：作业数 机器数
          接下来n_jobs行：每个作业的数据
          倒数第2行：交货期
          倒数第1行：机器能力等级
          最后几行：元数据（注释）
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # 读取第1行：作业数 机器数
        n_jobs, n_machines = map(int, lines[0].split())

        # 找到第一个注释行的位置
        comment_start = -1
        for i, line in enumerate(lines):
            if line.startswith('#'):
                comment_start = i
                break

        if comment_start == -1:
            comment_start = len(lines)

        # 计算数据行数
        # 总数据行数 = 注释行开始位置
        # 作业数据行数 = n_jobs
        # 交货期行 = 注释行开始位置 - 2
        # 机器能力行 = 注释行开始位置 - 1

        # 读取交货期行（倒数第2个数据行）
        due_dates_line = comment_start - 2
        due_dates = list(map(int, lines[due_dates_line].split()))

        # 读取机器能力等级行（倒数第1个数据行）
        capabilities_line = comment_start - 1
        machine_capabilities = list(map(int, lines[capabilities_line].split()))

        # 读取作业数据（第2行到第1+n_jobs行）
        operations_data = []
        for i in range(1, 1 + n_jobs):
            if i >= len(lines) or lines[i].startswith('#'):
                break

            job_data = list(map(int, lines[i].split()))
            idx = 0
            n_ops = job_data[idx]; idx += 1

            job_operations = []
            for _ in range(n_ops):
                n_options = job_data[idx]; idx += 1
                op_machines = []
                for __ in range(n_options):
                    machine = job_data[idx]; idx += 1
                    time = job_data[idx]; idx += 1
                    op_machines.append((machine, time))
                job_operations.append(op_machines)

            operations_data.append(job_operations)

        print(f"\n读取实例成功:")
        print(f"  作业数: {n_jobs}")
        print(f"  机器数: {n_machines}")
        print(f"  交货期: {due_dates}")
        print(f"  机器能力: {machine_capabilities}")
        print(f"  作业数据行数: {len(operations_data)}")

        return {
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'due_dates': due_dates,
            'machine_capabilities': machine_capabilities,
            'operations_data': operations_data
        }

    def print_instance_summary(self, instance_info: Dict[str, Any]):
        """在终端打印实例摘要"""
        print("\n" + "="*60)
        print(f"实例 {instance_info['instance_id']:03d} 摘要 ({instance_info['size_type']}规模)")
        print("="*60)
        print(f"作业数: {instance_info['n_jobs']}")
        print(f"机器数: {instance_info['n_machines']}")
        print(f"总工序数: {instance_info['total_operations']}")
        print(f"平均每作业工序数: {np.mean(instance_info['operations_per_job']):.2f}")
        print(f"交货期范围: {min(instance_info['due_dates'])} - {max(instance_info['due_dates'])}")
        print(f"机器能力: {instance_info['machine_capabilities']}")
        print(f"平均机器柔性度: {instance_info['actual_flexibility']:.3f}")

        # 加工时间统计
        all_times = []
        for job in instance_info['operations']:
            for op in job:
                for _, time in op['available_machines']:
                    all_times.append(time)

        print(f"加工时间范围: {min(all_times)} - {max(all_times)}")
        print(f"平均加工时间: {np.mean(all_times):.2f}")
        print("="*60)

    def generate_dataset(self):
        """生成完整的数据集"""
        print("开始生成MOFJSP数据集...")
        print("数据格式:")
        print("  第1行: 作业数 机器数")
        print("  接下来n_jobs行: 每个作业的数据")
        print("  倒数第2行: 交货期 (正式数据)")
        print("  倒数第1行: 机器能力等级 (正式数据)")
        print("  最后几行: 元数据 (注释)")

        # 确定各个规模的实例数量
        size_configs = self.config['dataset_config']['instance_sizes']
        splits = self.config['dataset_config']['train_val_test_split']

        instance_counter = 0
        all_instances = []

        # 生成每个规模的实例
        for size_config in size_configs:
            print(f"\n生成 {size_config['name']} 规模实例...")
            print(f"目标: {size_config['num_instances']}个实例")
            print(f"作业数范围: {size_config['jobs_range'][0]}-{size_config['jobs_range'][1]}")
            print(f"机器数范围: {size_config['machines_range'][0]}-{size_config['machines_range'][1]}")

            # 为当前规模生成指定数量的实例
            for i in range(size_config['num_instances']):
                # 在范围内随机选择作业数和机器数
                n_jobs = random.randint(size_config['jobs_range'][0], size_config['jobs_range'][1])
                n_machines = random.randint(size_config['machines_range'][0], size_config['machines_range'][1])

                # 生成实例
                instance_info = self.generate_instance(instance_counter, n_jobs, n_machines)
                all_instances.append(instance_info)

                # 确定划分（按比例）
                if i < size_config['num_instances'] * splits[0] / 100:
                    split = "train"
                elif i < size_config['num_instances'] * (splits[0] + splits[1]) / 100:
                    split = "val"
                else:
                    split = "test"

                # 保存实例
                filepath = self.save_instance(instance_info, split)
                self.print_instance_summary(instance_info)

                # 测试读取第一个实例
                if i == 0 and instance_counter == 0:
                    print("\n测试读取第一个实例...")
                    self.read_instance(filepath)

                instance_counter += 1

        # 生成数据集统计信息
        self.generate_statistics(all_instances)

        print(f"\n数据集生成完成！共生成 {len(all_instances)} 个实例。")
        print(f"实例保存在: {self.config['output_config']['output_directory']}")

    def generate_statistics(self, all_instances: List[Dict[str, Any]]):
        """生成数据集统计信息"""
        stats_file = os.path.join(
            self.config['output_config']['output_directory'],
            "dataset_statistics.txt"
        )

        stats_lines = [
            "MOFJSP数据集统计信息",
            "="*60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总实例数: {len(all_instances)}",
            f"训练/验证/测试划分: {self.config['dataset_config']['train_val_test_split']}",
            f"数据格式: 作业数据之后，倒数第2行交货期，倒数第1行机器能力等级",
            ""
        ]

        # 按规模统计
        for size_config in self.config['dataset_config']['instance_sizes']:
            size_instances = [inst for inst in all_instances if inst['size_type'] == size_config['name']]

            if size_instances:
                stats_lines.append(f"{size_config['name']}规模 ({len(size_instances)}个实例):")
                stats_lines.append(f"  作业数范围: {min(inst['n_jobs'] for inst in size_instances)}-{max(inst['n_jobs'] for inst in size_instances)}")
                stats_lines.append(f"  机器数范围: {min(inst['n_machines'] for inst in size_instances)}-{max(inst['n_machines'] for inst in size_instances)}")
                stats_lines.append(f"  平均工序数: {np.mean([inst['total_operations'] for inst in size_instances]):.1f}")
                stats_lines.append(f"  平均柔性度: {np.mean([inst['actual_flexibility'] for inst in size_instances]):.3f}")
                stats_lines.append("")

        # 整体统计
        stats_lines.append("整体统计:")
        stats_lines.append(f"  总作业数: {sum(inst['n_jobs'] for inst in all_instances)}")
        stats_lines.append(f"  总机器数: {sum(inst['n_machines'] for inst in all_instances)}")
        stats_lines.append(f"  总工序数: {sum(inst['total_operations'] for inst in all_instances)}")
        stats_lines.append(f"  平均机器柔性度: {np.mean([inst['actual_flexibility'] for inst in all_instances]):.3f}")

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(stats_lines))

        print(f"数据集统计信息已保存到: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="生成MOFJSP数据集（最终版）")
    parser.add_argument("--single", action="store_true", help="仅生成单个实例")
    parser.add_argument("--jobs", type=int, default=15, help="单个实例的作业数")
    parser.add_argument("--machines", type=int, default=8, help="单个实例的机器数")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], help="指定规模生成单个实例")

    args = parser.parse_args()

    # 配置文件固定为config.json
    config_file = "config.json"
    generator = MOFJSPInstanceGenerator(config_file)

    if args.single:
        if args.size:
            # 根据指定规模生成
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            size_config = None
            for sc in config['dataset_config']['instance_sizes']:
                if sc['name'] == args.size:
                    size_config = sc
                    break

            if size_config:
                n_jobs = random.randint(size_config['jobs_range'][0], size_config['jobs_range'][1])
                n_machines = random.randint(size_config['machines_range'][0], size_config['machines_range'][1])
                print(f"生成{args.size}规模实例: {n_jobs}作业, {n_machines}机器")
            else:
                n_jobs = args.jobs
                n_machines = args.machines
        else:
            n_jobs = args.jobs
            n_machines = args.machines

        # 生成单个实例
        instance_info = generator.generate_instance(0, n_jobs, n_machines)
        filepath = generator.save_instance(instance_info, "single")
        generator.print_instance_summary(instance_info)

        # 测试读取
        print("\n测试读取生成的实例:")
        generator.read_instance(filepath)
        print(f"\n单个实例已保存到: {filepath}")
    else:
        # 生成完整数据集
        generator.generate_dataset()

if __name__ == "__main__":
    main()