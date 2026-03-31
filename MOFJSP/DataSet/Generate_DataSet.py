"""
MOFJSP Dataset Generation Program
Multi-objective data (due dates, machine capabilities) is placed after job data, before comment lines.
The JSON configuration file is named config.json.
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
        """Initializes the generator"""
        if not os.path.exists(config_file):
            print(f"Configuration file {config_file} does not exist, creating with default settings...")
            self.create_default_config(config_file)

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Create output directory
        output_dir = self.config['output_config']['output_directory']
        os.makedirs(output_dir, exist_ok=True)

        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

        print(f"MOFJSP Dataset Generator initialized.")
        print(f"Configuration file: {config_file}")
        print(f"Output directory: {output_dir}")
        print(f"Multi-objective data format: After job data, before comment lines.")

    def create_default_config(self, config_file: str):
        """Creates the default configuration file"""
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

        print(f"Default configuration file created: {config_file}")

    def generate_instance(self, instance_id: int, n_jobs: int, n_machines: int) -> Dict[str, Any]:
        """Generates a single MOFJSP instance"""
        # Generate number of operations for each job
        ops_per_job = np.random.randint(
            self.config['generation_parameters']['operations_per_job_range'][0],
            self.config['generation_parameters']['operations_per_job_range'][1] + 1,
            size=n_jobs
        )

        total_operations = np.sum(ops_per_job)

        # Generate due dates (multiple of the minimum processing time)
        due_dates = []
        for job_idx in range(n_jobs):
            # Estimate minimum processing time
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
                # Determine the number of available machines for this operation (based on flexibility)
                available_machines = self.select_available_machines(
                    n_machines,
                    self.config['generation_parameters']['machine_flexibility']
                )

                # Generate processing time for each available machine
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
        """Determines the instance size type based on job and machine counts"""
        for size_config in self.config['dataset_config']['instance_sizes']:
            if (size_config['jobs_range'][0] <= n_jobs <= size_config['jobs_range'][1] and
                size_config['machines_range'][0] <= n_machines <= size_config['machines_range'][1]):
                return size_config['name']
        return "custom"

    def select_available_machines(self, total_machines: int, flexibility: float) -> List[int]:
        """Selects available machines for an operation"""
        # Ensure at least one machine is available
        min_available = max(1, int(total_machines * flexibility * 0.3))
        max_available = max(min_available, int(total_machines * flexibility * 1.2))

        n_available = random.randint(min_available, min(max_available, total_machines))
        available = random.sample(range(total_machines), n_available)
        return sorted(available)

    def format_instance_text(self, instance_info: Dict[str, Any]) -> str:
        """
        Formats the instance information into extended Brandimarte format.
        Format:
          Line 1: Number of jobs, Number of machines
          Next n_jobs lines: Data for each job
          Second to last line: Due dates (data line)
          Last line: Machine capability levels (data line)
          Final lines: Metadata (comment lines)
        """
        lines = []

        # Line 1: Number of jobs, Number of machines
        lines.append(f"{instance_info['n_jobs']} {instance_info['n_machines']}")

        # Next n_jobs lines: Job information
        for job_idx, job_ops in enumerate(instance_info['operations']):
            job_line = [len(job_ops)]  # Number of operations for this job

            for op in job_ops:
                job_line.append(len(op['available_machines']))  # Number of optional machines for this operation
                for machine_id, proc_time in op['available_machines']:
                    job_line.extend([machine_id, proc_time])

            lines.append(" ".join(map(str, job_line)))

        # Second to last line: Due dates (multi-objective data) - data line
        if self.config['output_config'].get('include_due_dates_in_data', True):
            lines.append(" ".join(map(str, instance_info['due_dates'])))

        # Last line: Machine capability levels (multi-objective data) - data line
        if self.config['output_config'].get('include_machine_capabilities_in_data', True):
            lines.append(" ".join(map(str, instance_info['machine_capabilities'])))

        # Metadata as comments (does not affect program reading)
        lines.append(f"# Instance ID: {instance_info['instance_id']}")
        lines.append(f"# Size type: {instance_info['size_type']}")
        lines.append(f"# Total operations: {instance_info['total_operations']}")
        lines.append(f"# Actual flexibility: {instance_info['actual_flexibility']:.3f}")
        lines.append(f"# Generated: {instance_info['generation_time']}")
        lines.append(f"# Format: Extended Brandimarte format for MOFJSP")

        return "\n".join(lines)

    def save_instance(self, instance_info: Dict[str, Any], split: str = "train"):
        """Saves the instance to a file"""
        filename = f"mo_fjsp_{instance_info['instance_id']:03d}_{instance_info['size_type']}_{split}.txt"
        filepath = os.path.join(
            self.config['output_config']['output_directory'],
            filename
        )

        instance_text = self.format_instance_text(instance_info)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(instance_text)

        print(f"Instance saved: {filename}")
        return filepath

    def read_instance(self, filepath: str) -> Dict[str, Any]:
        """
        Reads an instance file.
        Format description:
          Line 1: Number of jobs, Number of machines
          Next n_jobs lines: Data for each job
          Second to last line: Due dates
          Last line: Machine capability levels
          Final lines: Metadata (comments)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Read line 1: Number of jobs, Number of machines
        n_jobs, n_machines = map(int, lines[0].split())

        # Find the position of the first comment line
        comment_start = -1
        for i, line in enumerate(lines):
            if line.startswith('#'):
                comment_start = i
                break

        if comment_start == -1:
            comment_start = len(lines)

        # Calculate the number of data lines
        # Total data lines = comment start position
        # Job data lines = n_jobs
        # Due date line = comment start position - 2
        # Machine capability line = comment start position - 1

        # Read the due date line (second to last data line)
        due_dates_line = comment_start - 2
        due_dates = list(map(int, lines[due_dates_line].split()))

        # Read the machine capability level line (last data line)
        capabilities_line = comment_start - 1
        machine_capabilities = list(map(int, lines[capabilities_line].split()))

        # Read job data (lines 2 to 1+n_jobs)
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

        print(f"\nInstance read successfully:")
        print(f"  Number of jobs: {n_jobs}")
        print(f"  Number of machines: {n_machines}")
        print(f"  Due dates: {due_dates}")
        print(f"  Machine capabilities: {machine_capabilities}")
        print(f"  Job data lines: {len(operations_data)}")

        return {
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'due_dates': due_dates,
            'machine_capabilities': machine_capabilities,
            'operations_data': operations_data
        }

    def print_instance_summary(self, instance_info: Dict[str, Any]):
        """Prints an instance summary to the terminal"""
        print("\n" + "="*60)
        print(f"Instance {instance_info['instance_id']:03d} Summary ({instance_info['size_type']} size)")
        print("="*60)
        print(f"Number of jobs: {instance_info['n_jobs']}")
        print(f"Number of machines: {instance_info['n_machines']}")
        print(f"Total operations: {instance_info['total_operations']}")
        print(f"Average operations per job: {np.mean(instance_info['operations_per_job']):.2f}")
        print(f"Due date range: {min(instance_info['due_dates'])} - {max(instance_info['due_dates'])}")
        print(f"Machine capabilities: {instance_info['machine_capabilities']}")
        print(f"Average machine flexibility: {instance_info['actual_flexibility']:.3f}")

        # Processing time statistics
        all_times = []
        for job in instance_info['operations']:
            for op in job:
                for _, time in op['available_machines']:
                    all_times.append(time)

        print(f"Processing time range: {min(all_times)} - {max(all_times)}")
        print(f"Average processing time: {np.mean(all_times):.2f}")
        print("="*60)

    def generate_dataset(self):
        """Generates a complete dataset"""
        print("Starting MOFJSP dataset generation...")
        print("Data format:")
        print("  Line 1: Number of jobs, Number of machines")
        print("  Next n_jobs lines: Data for each job")
        print("  Second to last line: Due dates (data)")
        print("  Last line: Machine capability levels (data)")
        print("  Final lines: Metadata (comments)")

        # Determine the number of instances for each size
        size_configs = self.config['dataset_config']['instance_sizes']
        splits = self.config['dataset_config']['train_val_test_split']

        instance_counter = 0
        all_instances = []

        # Generate instances for each size
        for size_config in size_configs:
            print(f"\nGenerating {size_config['name']} size instances...")
            print(f"Target: {size_config['num_instances']} instances")
            print(f"Job count range: {size_config['jobs_range'][0]}-{size_config['jobs_range'][1]}")
            print(f"Machine count range: {size_config['machines_range'][0]}-{size_config['machines_range'][1]}")

            # Generate the specified number of instances for the current size
            for i in range(size_config['num_instances']):
                # Randomly select job and machine counts within the range
                n_jobs = random.randint(size_config['jobs_range'][0], size_config['jobs_range'][1])
                n_machines = random.randint(size_config['machines_range'][0], size_config['machines_range'][1])

                # Generate the instance
                instance_info = self.generate_instance(instance_counter, n_jobs, n_machines)
                all_instances.append(instance_info)

                # Determine the split (by proportion)
                if i < size_config['num_instances'] * splits[0] / 100:
                    split = "train"
                elif i < size_config['num_instances'] * (splits[0] + splits[1]) / 100:
                    split = "val"
                else:
                    split = "test"

                # Save the instance
                filepath = self.save_instance(instance_info, split)
                self.print_instance_summary(instance_info)

                # Test reading the first instance
                if i == 0 and instance_counter == 0:
                    print("\nTesting reading the first instance...")
                    self.read_instance(filepath)

                instance_counter += 1

        # Generate dataset statistics
        self.generate_statistics(all_instances)

        print(f"\nDataset generation completed! Total {len(all_instances)} instances generated.")
        print(f"Instances saved in: {self.config['output_config']['output_directory']}")

    def generate_statistics(self, all_instances: List[Dict[str, Any]]):
        """Generates dataset statistics"""
        stats_file = os.path.join(
            self.config['output_config']['output_directory'],
            "dataset_statistics.txt"
        )

        stats_lines = [
            "MOFJSP Dataset Statistics",
            "="*60,
            f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total instances: {len(all_instances)}",
            f"Train/Validation/Test split: {self.config['dataset_config']['train_val_test_split']}",
            f"Data format: Job data, then due dates (second to last), then machine capabilities (last)",
            ""
        ]

        # Statistics by size
        for size_config in self.config['dataset_config']['instance_sizes']:
            size_instances = [inst for inst in all_instances if inst['size_type'] == size_config['name']]

            if size_instances:
                stats_lines.append(f"{size_config['name']} size ({len(size_instances)} instances):")
                stats_lines.append(f"  Job count range: {min(inst['n_jobs'] for inst in size_instances)}-{max(inst['n_jobs'] for inst in size_instances)}")
                stats_lines.append(f"  Machine count range: {min(inst['n_machines'] for inst in size_instances)}-{max(inst['n_machines'] for inst in size_instances)}")
                stats_lines.append(f"  Average operations: {np.mean([inst['total_operations'] for inst in size_instances]):.1f}")
                stats_lines.append(f"  Average flexibility: {np.mean([inst['actual_flexibility'] for inst in size_instances]):.3f}")
                stats_lines.append("")

        # Overall statistics
        stats_lines.append("Overall statistics:")
        stats_lines.append(f"  Total jobs: {sum(inst['n_jobs'] for inst in all_instances)}")
        stats_lines.append(f"  Total machines: {sum(inst['n_machines'] for inst in all_instances)}")
        stats_lines.append(f"  Total operations: {sum(inst['total_operations'] for inst in all_instances)}")
        stats_lines.append(f"  Average machine flexibility: {np.mean([inst['actual_flexibility'] for inst in all_instances]):.3f}")

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(stats_lines))

        print(f"Dataset statistics saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Generates MOFJSP dataset (final version).")
    parser.add_argument("--single", action="store_true", help="Generate only a single instance.")
    parser.add_argument("--jobs", type=int, default=15, help="Number of jobs for a single instance.")
    parser.add_argument("--machines", type=int, default=8, help="Number of machines for a single instance.")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], help="Specify the size to generate a single instance.")

    args = parser.parse_args()

    # The configuration file is fixed as config.json
    config_file = "config.json"
    generator = MOFJSPInstanceGenerator(config_file)

    if args.single:
        if args.size:
            # Generate based on the specified size
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
                print(f"Generating a {args.size} size instance: {n_jobs} jobs, {n_machines} machines.")
            else:
                n_jobs = args.jobs
                n_machines = args.machines
        else:
            n_jobs = args.jobs
            n_machines = args.machines

        # Generate a single instance
        instance_info = generator.generate_instance(0, n_jobs, n_machines)
        filepath = generator.save_instance(instance_info, "single")
        generator.print_instance_summary(instance_info)

        # Test reading
        print("\nTesting reading the generated instance:")
        generator.read_instance(filepath)
        print(f"\nSingle instance saved to: {filepath}")
    else:
        # Generate a complete dataset
        generator.generate_dataset()

if __name__ == "__main__":
    main()