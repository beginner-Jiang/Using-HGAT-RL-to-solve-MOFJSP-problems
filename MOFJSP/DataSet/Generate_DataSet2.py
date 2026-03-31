"""
MOFJSP Comparison Dataset Generation Program
The configuration file is fixed as config2.json.
"""

import json
import random
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any

class MOFJSPComparisonGenerator:
    def __init__(self, config_file: str = "config2.json"):
        """Initializes the generator"""
        if not os.path.exists(config_file):
            print(f"Configuration file {config_file} does not exist, creating default comparison dataset configuration...")
            self.create_default_config(config_file)

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Create output directory
        output_dir = self.config['output_config']['output_directory']
        os.makedirs(output_dir, exist_ok=True)

        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

        print(f"MOFJSP Comparison Dataset Generator initialized.")
        print(f"Configuration file: {config_file}")
        print(f"Output directory: {output_dir}")
        print(f"Total instances to generate: {self.config['comparison_config']['total_instances']}")
        print("Multi-objective data format: After job data, second to last line for due dates, last line for machine capability levels.")

    def create_default_config(self, config_file: str):
        """Creates the default comparison dataset configuration file (config2.json)"""
        default_config = {
            "comparison_config": {
                "description": "Dedicated dataset for comparison experiments, 10 instances for each fixed size.",
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

        print(f"Default comparison dataset configuration file created: {config_file}")

    def generate_instance(self, instance_id: int, n_jobs: int, n_machines: int) -> Dict[str, Any]:
        """Generates a single MOFJSP instance (reuses the original generation logic)"""
        # Generate number of operations for each job
        ops_per_job = np.random.randint(
            self.config['generation_parameters']['operations_per_job_range'][0],
            self.config['generation_parameters']['operations_per_job_range'][1] + 1,
            size=n_jobs
        )
        total_operations = int(np.sum(ops_per_job))

        # Generate due dates (multiple of the minimum processing time)
        due_dates = []
        for job_idx in range(n_jobs):
            min_processing = ops_per_job[job_idx] * self.config['generation_parameters']['processing_time_range'][0]
            due_factor = random.uniform(*self.config['generation_parameters']['due_date_factor_range'])
            due_date = int(min_processing * due_factor)
            due_dates.append(due_date)

        # Generate operation information
        operations = []
        operation_id = 0
        for job_idx in range(n_jobs):
            job_operations = []
            for op_idx in range(ops_per_job[job_idx]):
                # Determine the number of available machines
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
                    machine_times.append((machine_idx + 1, processing_time))  # Machine index starts from 1
                job_operations.append({
                    'id': operation_id,
                    'job_id': job_idx + 1,
                    'op_idx': op_idx + 1,
                    'available_machines': machine_times
                })
                operation_id += 1
            operations.append(job_operations)

        # Machine capability levels
        machine_capabilities = random.choices(
            self.config['generation_parameters']['machine_capability_levels'],
            k=n_machines
        )

        # Calculate actual machine flexibility
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
        """Selects available machines for an operation"""
        min_available = max(1, int(total_machines * flexibility * 0.3))
        max_available = max(min_available, int(total_machines * flexibility * 1.2))
        n_available = random.randint(min_available, min(max_available, total_machines))
        available = random.sample(range(total_machines), n_available)
        return sorted(available)

    def format_instance_text(self, instance_info: Dict[str, Any]) -> str:
        """Formats into extended Brandimarte format"""
        lines = []
        # Line 1: Number of jobs, Number of machines
        lines.append(f"{instance_info['n_jobs']} {instance_info['n_machines']}")

        # Job information lines
        for job_idx, job_ops in enumerate(instance_info['operations']):
            job_line = [len(job_ops)]
            for op in job_ops:
                job_line.append(len(op['available_machines']))
                for machine_id, proc_time in op['available_machines']:
                    job_line.extend([machine_id, proc_time])
            lines.append(" ".join(map(str, job_line)))

        # Due date line
        if self.config['output_config'].get('include_due_dates_in_data', True):
            lines.append(" ".join(map(str, instance_info['due_dates'])))

        # Machine capability level line
        if self.config['output_config'].get('include_machine_capabilities_in_data', True):
            lines.append(" ".join(map(str, instance_info['machine_capabilities'])))

        # Metadata comments
        lines.append(f"# Instance ID: {instance_info['instance_id']}")
        lines.append(f"# Total operations: {instance_info['total_operations']}")
        lines.append(f"# Actual flexibility: {instance_info['actual_flexibility']:.3f}")
        lines.append(f"# Generated: {instance_info['generation_time']}")
        lines.append(f"# Format: Extended Brandimarte for MOFJSP (Comparison Dataset)")

        return "\n".join(lines)

    def save_instance(self, instance_info: Dict[str, Any], size_name: str):
        """Saves the instance to a file, with size information in the filename."""
        filename = f"comp_{size_name}_{instance_info['n_jobs']}x{instance_info['n_machines']}_{instance_info['instance_id']:03d}.txt"
        filepath = os.path.join(
            self.config['output_config']['output_directory'],
            filename
        )
        instance_text = self.format_instance_text(instance_info)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(instance_text)
        print(f"Saved: {filename}")
        return filepath

    def generate_comparison_dataset(self):
        """Generates the comparison dataset: generates instances one by one according to the fixed size list."""
        fixed_sizes = self.config['comparison_config']['fixed_sizes']
        instance_counter = 1  # Start numbering from 1 for easier continuous filenames

        print("\nStarting comparison dataset generation...")
        for size_entry in fixed_sizes:
            size_name = size_entry['name']
            jobs = size_entry['jobs']
            machines = size_entry['machines']
            count = size_entry['count']

            print(f"\nGenerating {size_name} size {jobs}×{machines}, {count} instances in total...")
            for i in range(count):
                inst = self.generate_instance(instance_counter, jobs, machines)
                self.save_instance(inst, size_name)
                # Print a simple summary
                print(f"  Instance {instance_counter:03d}: Operations={inst['total_operations']}, "
                      f"Due date range={min(inst['due_dates'])}-{max(inst['due_dates'])}")
                instance_counter += 1

        # Generate statistics
        self.generate_statistics()
        print(f"\nComparison dataset generation completed! Total {instance_counter-1} instances saved in {self.config['output_config']['output_directory']}")

    def generate_statistics(self):
        """Generates the dataset statistics file."""
        stats_file = os.path.join(
            self.config['output_config']['output_directory'],
            "comparison_dataset_statistics.txt"
        )
        lines = [
            "MOFJSP Comparison Dataset Statistics",
            "=" * 60,
            f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total instances: {self.config['comparison_config']['total_instances']}",
            "Fixed size list:",
        ]
        for sz in self.config['comparison_config']['fixed_sizes']:
            lines.append(f"  {sz['name']}: {sz['jobs']}×{sz['machines']} × {sz['count']}")

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"Statistics saved to: {stats_file}")

def main():
    generator = MOFJSPComparisonGenerator("config2.json")
    generator.generate_comparison_dataset()

if __name__ == "__main__":
    main()