"""
Heuristic rule implementation: Least Load First - LBD
"""
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# Data structure definition

class Operation:
    """Operation class"""

    def __init__(self, job_id: int, op_id: int, machine_times: Dict[int, float]):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_times = machine_times  # Machine ID -> Processing time
        self.assigned_machine = None
        self.start_time = 0
        self.end_time = 0
        self.is_scheduled = False

    def get_available_machines(self) -> List[int]:
        """Returns the list of available machines"""
        return list(self.machine_times.keys())

    def get_processing_time(self, machine_id: int) -> float:
        """Returns processing time on the specified machine"""
        return self.machine_times.get(machine_id, float('inf'))


class Job:
    """Job class"""

    def __init__(self, job_id: int, operations: List[Operation], due_date: float):
        self.job_id = job_id
        self.operations = operations
        self.due_date = due_date
        self.current_op_index = 0  # Index of the current operation to be scheduled

    def get_current_operation(self) -> Operation:
        """Returns the current operation to be scheduled"""
        if self.current_op_index < len(self.operations):
            return self.operations[self.current_op_index]
        return None

    def complete_current_operation(self):
        """Marks the current operation as completed and prepares for the next one"""
        if self.current_op_index < len(self.operations):
            self.current_op_index += 1

    def is_completed(self) -> bool:
        """Checks if the job is fully completed"""
        return self.current_op_index >= len(self.operations)

    def get_completion_time(self) -> float:
        """Returns the job completion time (end time of the last operation)"""
        if len(self.operations) == 0:
            return 0
        return self.operations[-1].end_time


class Machine:
    """Machine class"""

    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.schedule = []  # List of scheduled operations [(start_time, end_time, operation)]
        self.available_time = 0  # Earliest available time of the machine
        self.current_load = 0  # Current load (sum of processing time of scheduled operations)

    def assign_operation(self, operation: Operation, start_time: float) -> float:
        """Assigns an operation to the machine, returns the completion time"""
        # Check if the machine is available
        processing_time = operation.get_processing_time(self.machine_id)
        if processing_time == float('inf'):
            return float('inf')

        # Ensure start time is not earlier than machine available time
        actual_start = max(start_time, self.available_time)
        end_time = actual_start + processing_time

        # Update machine status
        self.schedule.append((actual_start, end_time, operation))
        self.available_time = end_time
        self.current_load += processing_time

        # Update operation status
        operation.assigned_machine = self.machine_id
        operation.start_time = actual_start
        operation.end_time = end_time
        operation.is_scheduled = True

        return end_time

    def get_total_load(self) -> float:
        """Returns the total load of the machine (sum of processing times)"""
        return self.current_load

    def get_earliest_available_time(self) -> float:
        """Returns the machine's earliest available time"""
        return self.available_time


# LBD scheduling algorithm implementation

class LBDScheduler:
    """Scheduler based on LBD (Least Load First) rule"""

    def __init__(self, jobs: List[Job], machines: List[Machine]):
        self.jobs = jobs
        self.machines = machines
        self.current_time = 0
        self.completed_operations = 0
        self.total_operations = sum(len(job.operations) for job in jobs)

    def get_available_operations(self) -> List[Operation]:
        """Gets all schedulable operations (first unscheduled operation of each job)"""
        available_ops = []
        for job in self.jobs:
            if not job.is_completed():
                current_op = job.get_current_operation()
                if current_op and not current_op.is_scheduled:
                    available_ops.append(current_op)
        return available_ops

    def find_best_machine_for_operation(self, operation: Operation, job_ready_time: float) -> Tuple[int, float, float]:
        """Finds the best machine for the operation: the available machine with the smallest current load"""
        best_machine_id = -1
        best_completion_time = float('inf')
        best_machine_load = float('inf')

        # Iterate over all machines to find the available machine with the smallest load
        for machine in self.machines:
            # Check if the machine is available
            processing_time = operation.get_processing_time(machine.machine_id)
            if processing_time == float('inf'):
                continue

            # LBD rule: prioritize the machine with the smallest current load
            machine_load = machine.get_total_load()

            # Calculate completion time on this machine
            start_time = max(job_ready_time, machine.get_earliest_available_time())
            completion_time = start_time + processing_time

            # LBD rule: choose the machine with the smallest load
            # If loads are equal, choose the one with earlier completion time
            if machine_load < best_machine_load or (
                    machine_load == best_machine_load and completion_time < best_completion_time):
                best_machine_id = machine.machine_id
                best_completion_time = completion_time
                best_machine_load = machine_load

        return best_machine_id, best_completion_time, best_machine_load

    def select_operation_for_scheduling(self, available_ops: List[Operation]) -> Operation:
        """Selects an operation for scheduling from the available ones"""
        # LBD rule needs to consider both machine load and operation characteristics
        # Here we adopt: For each operation, find the machine with the smallest load,
        # then select the operation with the shortest "processing time on the machine with the smallest load"
        best_operation = None
        best_machine_load = float('inf')
        best_processing_time = float('inf')

        for operation in available_ops:
            # Get job ready time
            job = self.jobs[operation.job_id]
            job_ready_time = 0
            if operation.op_id > 0:
                prev_op = job.operations[operation.op_id - 1]
                job_ready_time = prev_op.end_time

            # Find the machine with the smallest load
            machine_id, _, machine_load = self.find_best_machine_for_operation(operation, job_ready_time)

            if machine_id == -1:
                continue

            # Get processing time on that machine
            processing_time = operation.get_processing_time(machine_id)

            # Selection criteria: prioritize operations on machines with smaller loads
            # If loads are equal, choose the one with shorter processing time
            if (machine_load < best_machine_load or
                    (machine_load == best_machine_load and processing_time < best_processing_time)):
                best_operation = operation
                best_machine_load = machine_load
                best_processing_time = processing_time

        return best_operation

    def schedule_step(self) -> bool:
        """Executes one scheduling step, returns whether there are still operations to be scheduled"""
        available_ops = self.get_available_operations()

        if not available_ops:
            return False

        # 1. LBD rule: Select the operation on the machine with the smallest load
        selected_op = self.select_operation_for_scheduling(available_ops)

        if selected_op is None:
            # If no suitable operation is found, try selecting the first available operation
            selected_op = available_ops[0]

        # 2. Find the machine with the smallest load for the selected operation
        job = self.jobs[selected_op.job_id]
        job_ready_time = 0
        if selected_op.op_id > 0:
            prev_op = job.operations[selected_op.op_id - 1]
            job_ready_time = prev_op.end_time

        best_machine_id, completion_time, _ = self.find_best_machine_for_operation(selected_op, job_ready_time)

        if best_machine_id == -1:
            # If no available machine, skip this operation
            # Mark as scheduled but actually not assigned (this should not happen as we checked available machines)
            selected_op.is_scheduled = True
            job.complete_current_operation()
            self.completed_operations += 1
            return True

        # 3. Assign the operation to the machine
        best_machine = self.machines[best_machine_id]
        actual_completion_time = best_machine.assign_operation(selected_op, job_ready_time)

        # 4. Update job status
        if actual_completion_time != float('inf'):
            job.complete_current_operation()
            self.completed_operations += 1

        # 5. Update current time (minimum of earliest available times of all machines)
        machine_times = [machine.available_time for machine in self.machines]
        self.current_time = min(machine_times) if machine_times else self.current_time

        return True

    def run_schedule(self):
        """Runs the complete schedule"""
        while self.schedule_step():
            pass

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculates scheduling performance metrics"""
        # 1. Makespan
        completion_times = []
        for job in self.jobs:
            completion_times.append(job.get_completion_time())
        makespan = max(completion_times)

        # 2. Machine load balance (Standard Deviation)
        machine_loads = [machine.get_total_load() for machine in self.machines]
        avg_load = np.mean(machine_loads)
        if avg_load == 0:
            load_balance = 0
        else:
            load_balance = np.sqrt(np.mean([(load - avg_load) ** 2 for load in machine_loads])) / avg_load

        # 3. Total tardiness
        total_tardiness = 0
        for job in self.jobs:
            completion_time = job.get_completion_time()
            tardiness = max(0, completion_time - job.due_date)
            total_tardiness += tardiness

        return {
            "makespan": makespan,
            "load_balance": load_balance,
            "total_tardiness": total_tardiness
        }

    def print_schedule(self):
        """Prints the schedule result"""
        print("=" * 80)
        print("LBD Schedule Result (Least Load First)")
        print("=" * 80)

        # Print by machine
        for machine in self.machines:
            print(f"\nMachine {machine.machine_id} (Total Load: {machine.get_total_load():.1f}):")
            machine.schedule.sort(key=lambda x: x[0])  # Sort by start time
            for start, end, op in machine.schedule:
                print(f"  Job{op.job_id}-Operation{op.op_id}: [{start:.1f} - {end:.1f}]")

        # Print performance metrics
        metrics = self.calculate_metrics()
        print(f"\n{'=' * 80}")
        print("Performance Metrics:")
        print(f"  Makespan: {metrics['makespan']:.2f}")
        print(f"  Machine Load Balance: {metrics['load_balance']:.4f}")
        print(f"  Total Tardiness: {metrics['total_tardiness']:.2f}")

        # Print machine load distribution
        print(f"\n{'=' * 80}")
        print("Machine Load Distribution:")
        total_load = 0
        for machine in self.machines:
            load = machine.get_total_load()
            total_load += load
            print(f"  Machine {machine.machine_id}: {load:.1f}")
        print(f"  Total Load: {total_load:.1f}")
        print(f"  Average Load: {total_load / len(self.machines):.1f}")

        return metrics


# File reading module

def read_fjsp_instance(file_path: str) -> Tuple[List[Job], List[Machine]]:
    """
    Reads a standard MOFJSP data file (extended Brandimarte format)
    File format:
        First line: Number of jobs   Number of machines
        Next N lines: Process information for each job
            - First integer: Number of operations O for this job
            - Following O groups: Each group starts with k, followed by k pairs (MachineID, Processing Time)
        Then one line: N integers, due date for each job
        There may be additional lines (e.g., flexibility matrix), ignore them.
    Returns: (list of jobs, list of machines)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Filter empty lines and comment lines
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        exit(1)

    if len(lines) < 2:
        raise ValueError("File format error: insufficient lines")

    # First line: Number of jobs, Number of machines
    num_jobs, num_machines = map(int, lines[0].split())
    machines = [Machine(i) for i in range(num_machines)]

    # The next num_jobs lines are job operation data
    job_lines = lines[1:1 + num_jobs]
    if len(job_lines) != num_jobs:
        raise ValueError(f"File format error: expected {num_jobs} job lines, got {len(job_lines)} lines")

    # The next line is due dates (must exist)
    due_date_line = lines[1 + num_jobs]
    due_dates = list(map(float, due_date_line.split()))
    if len(due_dates) != num_jobs:
        raise ValueError(f"File format error: expected {num_jobs} due dates, got {len(due_dates)}")

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
                raise ValueError(f"Job {job_idx} data incomplete")
            k = nums[idx]
            idx += 1
            machine_times = {}
            for _ in range(k):
                if idx + 1 >= len(nums):
                    raise ValueError(f"Insufficient machine data for Job {job_idx} operation {op_idx}")
                machine_id = nums[idx] - 1  # Convert to 0-index
                processing_time = float(nums[idx + 1])
                if 0 <= machine_id < num_machines:
                    machine_times[machine_id] = processing_time
                idx += 2
            op = Operation(job_idx, op_idx, machine_times)
            operations.append(op)
        due_date = due_dates[job_idx]
        job = Job(job_idx, operations, due_date)
        jobs.append(job)

    return jobs, machines


# Visualization function

def visualize_gantt_chart(jobs: List[Job], machines: List[Machine],
                          title="LBD Rule - Gantt Chart", save_path=None):
    """Draws and saves the Gantt chart"""
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#EF476F', '#073B4C', '#118AB2']

    # Assign colors to jobs
    job_colors = {}
    for i, job in enumerate(jobs):
        job_colors[job.job_id] = colors[i % len(colors)]

    # Draw schedule for each machine
    for machine in machines:
        y_pos = machine.machine_id
        for start, end, op in machine.schedule:
            color = job_colors[op.job_id]
            ax.barh(y_pos, end - start, left=start, height=0.6,
                    color=color, edgecolor='black')
            # Operation label moved

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {i}' for i in range(len(machines))])
    ax.set_title(title, fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=job_colors[job.job_id], label=f'Job {job.job_id}')
                      for job in jobs]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.grid(axis='x', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to: {save_path}")

    plt.show()


# Main function

def main():
    """Main function: reads file, executes LBD scheduling, displays and saves Gantt chart, exports metrics to Excel"""
    print("=" * 80)
    print("Flexible Job Shop Scheduling based on LBD (Least Load First) Rule (File-driven)")
    print("=" * 80)

    # Specify the data file path
    file_path = "../mo_fjsp_instances/mo_fjsp_000_small_train.txt"  # Ensure the file exists

    # 1. Read the scheduling instance from the file
    try:
        jobs, machines = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print("\nProblem Description:")
    print(f"  Number of Jobs: {len(jobs)}")
    print(f"  Number of Machines: {len(machines)}")

    total_ops = sum(len(job.operations) for job in jobs)
    print(f"  Total Number of Operations: {total_ops}")

    # 2. Execute LBD scheduling
    scheduler = LBDScheduler(jobs, machines)
    scheduler.run_schedule()

    # 3. Print scheduling results and obtain metrics
    lbd_metrics = scheduler.print_schedule()

    # 4. Draw and save Gantt chart
    print("\nGenerating Gantt Chart...")
    visualize_gantt_chart(jobs, machines, save_path="../Figure_And_File/Heuristic/LBD/gantt_chart_lbd.png")

    # 5. Export performance metrics to an Excel file
    try:
        # Create summary metrics DataFrame
        summary_data = {
            'Metric': ['Makespan', 'Machine Load Balance', 'Total Tardiness'],
            'Value': [
                lbd_metrics['makespan'],
                lbd_metrics['load_balance'],
                lbd_metrics['total_tardiness']
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        # Create tardy job details DataFrame (recalculated from jobs)
        tardy_jobs_list = []
        for job in jobs:
            completion_time = job.get_completion_time()
            tardiness = max(0, completion_time - job.due_date)
            if tardiness > 0:
                tardy_jobs_list.append({
                    'Job ID': job.job_id,
                    'Completion Time': completion_time,
                    'Due Date': job.due_date,
                    'Tardiness': tardiness
                })
        df_tardy = pd.DataFrame(tardy_jobs_list) if tardy_jobs_list else pd.DataFrame({'Note': ['No tardy jobs']})

        # Write to Excel file, containing two sheets
        excel_file = '../Figure_And_File/Heuristic/LBD/LBD_metrics.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_tardy.to_excel(writer, sheet_name='TardyJobs', index=False)

        print(f"\nPerformance metrics have been successfully exported to {excel_file}")
    except ImportError:
        print("\nWarning: pandas or openpyxl not installed, cannot export to Excel. Please install: pip install pandas openpyxl")
    except Exception as e:
        print(f"\nError exporting to Excel: {e}")

    # 6. Verify schedule feasibility
    print("\n" + "=" * 80)
    print("Schedule Feasibility Verification:")
    print("-" * 80)

    # Check if all operations are scheduled
    unscheduled_ops = []
    for job in jobs:
        for op in job.operations:
            if not op.is_scheduled:
                unscheduled_ops.append((job.job_id, op.op_id))

    if unscheduled_ops:
        print(f"  Warning: {len(unscheduled_ops)} operations are not scheduled")
        for job_id, op_id in unscheduled_ops:
            print(f"    Job{job_id}-Operation{op_id}")
    else:
        print("  All operations have been successfully scheduled")

    # Check for machine conflicts
    conflicts = []
    for machine in machines:
        schedule = sorted(machine.schedule, key=lambda x: x[0])  # Sort by start time
        for i in range(1, len(schedule)):
            prev_end = schedule[i - 1][1]
            curr_start = schedule[i][0]
            if curr_start < prev_end:
                conflicts.append((machine.machine_id, i - 1, i))

    if conflicts:
        print(f"  Warning: Found {len(conflicts)} machine time conflicts")
    else:
        print("  No machine time conflicts")

    print("\nLBD scheduling completed!")


if __name__ == "__main__":
    main()