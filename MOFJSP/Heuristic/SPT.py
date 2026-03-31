"""
Heuristic rule implementation: Shortest Processing Time - SPT
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

    def get_min_processing_time(self) -> float:
        """Returns the minimum processing time"""
        return min(self.machine_times.values())

    def get_best_machine(self) -> int:
        """Returns the machine ID with the shortest processing time"""
        return min(self.machine_times.items(), key=lambda x: x[1])[0]


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

    def assign_operation(self, operation: Operation, start_time: float) -> float:
        """Assigns an operation to the machine, returns the completion time"""
        # Find the machine with the shortest processing time
        best_machine_id = operation.get_best_machine()
        if best_machine_id != self.machine_id:
            # If this machine is not the optimal choice, return a large number indicating unsuitable
            return float('inf')

        processing_time = operation.machine_times[self.machine_id]
        # Ensure start time is not earlier than machine available time
        actual_start = max(start_time, self.available_time)
        end_time = actual_start + processing_time

        # Update machine status
        self.schedule.append((actual_start, end_time, operation))
        self.available_time = end_time

        # Update operation status
        operation.assigned_machine = self.machine_id
        operation.start_time = actual_start
        operation.end_time = end_time
        operation.is_scheduled = True

        return end_time

    def get_total_load(self) -> float:
        """Returns the total load of the machine (sum of processing times)"""
        return sum(end - start for start, end, _ in self.schedule)


# SPT scheduling algorithm implementation

class SPTScheduler:
    """Scheduler based on SPT (Shortest Processing Time) rule"""

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

    def schedule_step(self) -> bool:
        """Executes one scheduling step, returns whether there are still operations to be scheduled"""
        available_ops = self.get_available_operations()

        if not available_ops:
            return False

        # 1. SPT rule: Select the operation with the shortest processing time
        selected_op = min(available_ops, key=lambda op: op.get_min_processing_time())

        # 2. Select the machine for the operation (choose the machine with the shortest processing time)
        best_machine_id = selected_op.get_best_machine()
        best_machine = self.machines[best_machine_id]

        # 3. Calculate start time
        job = self.jobs[selected_op.job_id]
        job_ready_time = 0
        if selected_op.op_id > 0:
            prev_op = job.operations[selected_op.op_id - 1]
            job_ready_time = prev_op.end_time

        # 4. Assign operation to machine
        completion_time = best_machine.assign_operation(selected_op, job_ready_time)

        # 5. Update job status
        if completion_time != float('inf'):
            job.complete_current_operation()
            self.completed_operations += 1

        # 6. Update current time
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
        completion_times = [job.get_completion_time() for job in self.jobs]
        makespan = max(completion_times)

        # 2. Machine load balance (Standard Deviation)
        machine_loads = [machine.get_total_load() for machine in self.machines]
        avg_load = np.mean(machine_loads)
        load_balance = np.sqrt(np.mean([(load - avg_load) ** 2 for load in machine_loads]))

        # 3. Total tardiness & tardy job information
        total_tardiness = 0
        tardy_jobs = []
        for job in self.jobs:
            completion_time = job.get_completion_time()
            tardiness = max(0, completion_time - job.due_date)
            total_tardiness += tardiness
            if tardiness > 0:
                tardy_jobs.append((job.job_id, completion_time, job.due_date, tardiness))

        avg_tardiness = total_tardiness / len(self.jobs) if self.jobs else 0
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
        """Prints the schedule result"""
        print("=" * 80)
        print("SPT Schedule Result (Shortest Processing Time First)")
        print("=" * 80)

        # Print by machine
        for machine in self.machines:
            print(f"\nMachine {machine.machine_id} (Total Load: {machine.get_total_load():.1f}):")
            machine.schedule.sort(key=lambda x: x[0])
            for start, end, op in machine.schedule:
                job_due_date = self.jobs[op.job_id].due_date
                print(f"  Job{op.job_id}-Operation{op.op_id}: [{start:.1f} - {end:.1f}], Job Due Date: {job_due_date}")

        # Print performance metrics
        metrics = self.calculate_metrics()
        print(f"\n{'=' * 80}")
        print("Performance Metrics:")
        print(f"  Makespan: {metrics['makespan']:.2f}")
        print(f"  Machine Load Balance: {metrics['load_balance']:.4f}")
        print(f"  Total Tardiness: {metrics['total_tardiness']:.2f}")
        print(f"  Average Tardiness: {metrics['avg_tardiness']:.2f}")
        print(f"  Tardy Job Ratio: {metrics['tardy_ratio']:.2%}")

        # Print tardy job details
        if metrics['tardy_jobs']:
            print(f"\n{'=' * 80}")
            print("Tardy Job Details:")
            for job_id, completion_time, due_date, tardiness in metrics['tardy_jobs']:
                print(f"  Job{job_id}: Completion Time={completion_time:.1f}, Due Date={due_date}, Tardiness={tardiness:.1f}")

        # Print job completion status
        print(f"\n{'=' * 80}")
        print("Job Completion Status:")
        for job in self.jobs:
            completion_time = job.get_completion_time()
            tardiness = max(0, completion_time - job.due_date)
            status = "Tardy" if tardiness > 0 else "On Time"
            print(f"  Job{job.job_id}: Due Date={job.due_date}, Completion Time={completion_time:.1f}, "
                  f"Tardiness={tardiness:.1f}, Status={status}")

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
                          title="SPT Rule - Gantt Chart", save_path=None):
    """Draws and saves the Gantt chart"""
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#EF476F', '#073B4C', '#118AB2']
    job_colors = {job.job_id: colors[i % len(colors)] for i, job in enumerate(jobs)}

    # Draw schedule for each machine
    for machine in machines:
        y_pos = machine.machine_id
        for start, end, op in machine.schedule:
            color = job_colors[op.job_id]
            ax.barh(y_pos, end - start, left=start, height=0.6,
                    color=color, edgecolor='black')
            # Operation label moved

    # Add due date lines
    for job in jobs:
        due_date = job.due_date
        ax.axvline(x=due_date, color=job_colors[job.job_id], linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {i}' for i in range(len(machines))])
    ax.set_title(title, fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=job_colors[job.job_id], label=f'Job {job.job_id} (Due={job.due_date})')
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
    """Main function: reads file, executes SPT scheduling, displays and saves Gantt chart, exports metrics to Excel"""
    print("=" * 80)
    print("Flexible Job Shop Scheduling based on SPT (Shortest Processing Time) Rule (File-driven)")
    print("=" * 80)

    # Specify the data file path
    file_path = "../mo_fjsp_instances/mo_fjsp_000_small_train.txt"

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

    print("\nJob Due Dates:")
    for job in jobs:
        min_total_time = sum(min(op.machine_times.values()) for op in job.operations)
        print(f"  Job{job.job_id}: Minimum Completion Time={min_total_time:.1f}, Due Date={job.due_date:.1f}, "
              f"Slack Time={job.due_date - min_total_time:.1f}")

    # 2. Execute SPT scheduling
    scheduler = SPTScheduler(jobs, machines)
    scheduler.run_schedule()

    # 3. Print scheduling results and obtain metrics
    spt_metrics = scheduler.print_schedule()

    # 4. Draw and save Gantt chart
    print("\nGenerating Gantt Chart...")
    visualize_gantt_chart(jobs, machines, save_path="../Figure_And_File/Heuristic/SPT/gantt_chart_spt.png")

    # 5. Export performance metrics to an Excel file
    try:
        # Create summary metrics DataFrame
        summary_data = {
            'Metric': ['Makespan', 'Machine Load Balance', 'Total Tardiness', 'Average Tardiness', 'Tardy Job Ratio'],
            'Value': [
                spt_metrics['makespan'],
                spt_metrics['load_balance'],
                spt_metrics['total_tardiness'],
                spt_metrics['avg_tardiness'],
                spt_metrics['tardy_ratio']
            ]
        }
        df_summary = pd.DataFrame(summary_data)

        # Create tardy job details DataFrame (if any)
        tardy_jobs_list = []
        for job_id, comp_time, due, tard in spt_metrics['tardy_jobs']:
            tardy_jobs_list.append({
                'Job ID': job_id,
                'Completion Time': comp_time,
                'Due Date': due,
                'Tardiness': tard
            })
        df_tardy = pd.DataFrame(tardy_jobs_list) if tardy_jobs_list else pd.DataFrame({'Note': ['No tardy jobs']})

        # Write to Excel file, containing two sheets
        excel_file = '../Figure_And_File/Heuristic/SPT/SPT_metrics.xlsx'
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
        schedule = sorted(machine.schedule, key=lambda x: x[0])
        for i in range(1, len(schedule)):
            prev_end = schedule[i - 1][1]
            curr_start = schedule[i][0]
            if curr_start < prev_end:
                conflicts.append((machine.machine_id, i - 1, i))

    if conflicts:
        print(f"  Warning: Found {len(conflicts)} machine time conflicts")
    else:
        print("  No machine time conflicts")

    print("\nSPT scheduling completed!")

if __name__ == "__main__":
    main()