"""
Metaheuristic Algorithm Implementation: Non-dominated Sorting Genetic Algorithm II (NSGA-II)
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


# Data structure definitions

@dataclass
class Operation:
    """Operation class"""
    job_id: int
    op_id: int
    machine_times: Dict[int, float]  # machine ID -> processing time

    def get_processing_time(self, machine_id: int) -> float:
        """Returns the processing time on the specified machine"""
        return self.machine_times.get(machine_id, float('inf'))


@dataclass
class Job:
    """Job class"""
    job_id: int
    operations: List[Operation]
    due_date: float


@dataclass
class Chromosome:
    """Chromosome class, representing a scheduling solution"""
    operation_sequence: List[Tuple[int, int]]  # operation sequence [(job_id, op_id), ...]
    machine_assignment: List[int]  # machine assignment sequence, corresponding one-to-one with operation sequence
    objectives: List[float] = None  # objective values [makespan, load_balance, total_tardiness]
    rank: int = 0  # non-dominated sorting rank
    crowding_distance: float = 0.0  # crowding distance

    def __post_init__(self):
        """Validate chromosome validity"""
        if len(self.operation_sequence) != len(self.machine_assignment):
            raise ValueError("Operation sequence and machine assignment sequence must have the same length")

    def copy(self):
        """Create a deep copy of the chromosome"""
        return Chromosome(
            operation_sequence=self.operation_sequence.copy(),
            machine_assignment=self.machine_assignment.copy(),
            objectives=self.objectives.copy() if self.objectives else None,
            rank=self.rank,
            crowding_distance=self.crowding_distance
        )

    def __len__(self):
        """Return the chromosome length (number of operations)"""
        return len(self.operation_sequence)


class NSGA2Scheduler:
    """NSGA-II based scheduler"""

    def __init__(self, jobs: List[Job], machines: List[int],
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1):
        """
        Initialize NSGA-II scheduler

        Args:
            jobs: list of jobs
            machines: list of machine IDs
            population_size: population size
            max_generations: maximum number of generations
            crossover_rate: crossover probability
            mutation_rate: mutation probability
        """
        self.jobs = jobs
        self.machines = machines
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Total number of operations
        self.total_operations = sum(len(job.operations) for job in jobs)

        # Create job-operation mapping
        self.job_ops_map = {}
        self.op_index_map = {}  # (job_id, op_id) -> global index
        self.global_index_to_op = {}  # global index -> (job_id, op_id)

        idx = 0
        for job in jobs:
            self.job_ops_map[job.job_id] = []
            for op in job.operations:
                self.job_ops_map[job.job_id].append((job.job_id, op.op_id))
                self.op_index_map[(job.job_id, op.op_id)] = idx
                self.global_index_to_op[idx] = (job.job_id, op.op_id)
                idx += 1

        # Population storage
        self.population: List[Chromosome] = []
        self.fronts: List[List[Chromosome]] = []  # non-dominated fronts

        # Record best solutions
        self.best_solutions: List[Chromosome] = []
        self.history: List[Dict] = []  # history records

        # Visualization data
        self.visualization_data = {
            'generations': [],
            'avg_makespan': [],
            'avg_load_balance': [],
            'avg_tardiness': [],
            'best_makespan': [],
            'best_load_balance': [],
            'best_tardiness': []
        }

    # Chromosome initialization

    def initialize_population(self):
        """Initialize population"""
        self.population = []

        for _ in range(self.population_size):
            # 1. Generate operation sequence (operation-based encoding)
            operation_sequence = []

            # Create a copy of each job's operation list
            job_ops_remaining = {job_id: ops.copy() for job_id, ops in self.job_ops_map.items()}

            # Randomly generate operation sequence
            while any(job_ops_remaining.values()):
                # Randomly select a job that still has remaining operations
                available_jobs = [job_id for job_id, ops in job_ops_remaining.items() if ops]
                if not available_jobs:
                    break

                selected_job = random.choice(available_jobs)
                # Select the next operation of the selected job
                selected_op = job_ops_remaining[selected_job].pop(0)
                operation_sequence.append(selected_op)

            # 2. Generate machine assignment sequence
            machine_assignment = []
            for job_id, op_id in operation_sequence:
                # Get operation object
                job = self.jobs[job_id]
                op = job.operations[op_id]

                # Randomly select a machine from the available machines
                available_machines = list(op.machine_times.keys())
                if available_machines:
                    selected_machine = random.choice(available_machines)
                else:
                    # If no machine available, randomly select a machine (should not happen)
                    selected_machine = random.choice(self.machines)

                machine_assignment.append(selected_machine)

            # Create chromosome
            chromosome = Chromosome(operation_sequence, machine_assignment)
            self.population.append(chromosome)

    # Decoding and objective function calculation

    def decode_chromosome(self, chromosome: Chromosome) -> Tuple[Dict, Dict, Dict]:
        """
        Decode chromosome into a schedule

        Returns:
            Tuple[operation_schedule, machine_schedule, job_schedule]
        """
        # Initialize data structures
        machine_schedule = {machine_id: [] for machine_id in self.machines}
        job_completion_times = {job.job_id: 0 for job in self.jobs}
        operation_completion_times = {}

        # Process each operation
        for i, ((job_id, op_id), machine_id) in enumerate(zip(
                chromosome.operation_sequence, chromosome.machine_assignment)):

            # Get operation object
            job = self.jobs[job_id]
            op = job.operations[op_id]

            # Get processing time
            processing_time = op.get_processing_time(machine_id)
            if processing_time == float('inf'):
                # If machine is unavailable, use the minimum processing time
                available_machines = list(op.machine_times.keys())
                if available_machines:
                    machine_id = available_machines[0]
                    processing_time = op.get_processing_time(machine_id)
                else:
                    processing_time = 100  # default value

            # Calculate start time
            # 1. Job constraint: completion time of previous operation
            job_ready_time = 0
            if op_id > 0:
                prev_op_key = (job_id, op_id - 1)
                if prev_op_key in operation_completion_times:
                    job_ready_time = operation_completion_times[prev_op_key]

            # 2. Machine constraint: machine available time
            machine_available_time = 0
            if machine_schedule[machine_id]:
                machine_available_time = max(end for _, end, _ in machine_schedule[machine_id])

            # Actual start time
            start_time = max(job_ready_time, machine_available_time)
            end_time = start_time + processing_time

            # Record scheduling information
            machine_schedule[machine_id].append((start_time, end_time, (job_id, op_id)))
            operation_completion_times[(job_id, op_id)] = end_time

            # Update job completion time
            if op_id == len(job.operations) - 1:
                job_completion_times[job_id] = end_time

        return operation_completion_times, machine_schedule, job_completion_times

    def evaluate_chromosome(self, chromosome: Chromosome) -> List[float]:
        """Evaluate chromosome, compute three objective function values"""
        _, machine_schedule, job_completion_times = self.decode_chromosome(chromosome)

        # 1. Makespan
        all_completion_times = []
        for schedules in machine_schedule.values():
            for start, end, _ in schedules:
                all_completion_times.append(end)

        if not all_completion_times:
            makespan = 0
        else:
            makespan = max(all_completion_times)

        # 2. Load balance
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
                # Use coefficient of variation (std/mean) as balance metric
                std_load = np.std(machine_loads)
                load_balance = std_load / avg_load

        # 3. Total tardiness
        total_tardiness = 0
        for job in self.jobs:
            completion_time = job_completion_times.get(job.job_id, 0)
            tardiness = max(0, completion_time - job.due_date)
            total_tardiness += tardiness

        return [makespan, load_balance, total_tardiness]

    def evaluate_population(self):
        """Evaluate the entire population"""
        for chromosome in self.population:
            if chromosome.objectives is None:
                chromosome.objectives = self.evaluate_chromosome(chromosome)

    # NSGA-II core algorithms

    def fast_non_dominated_sort(self, population: List[Chromosome]) -> List[List[Chromosome]]:
        """Fast non-dominated sorting"""
        fronts = [[]]

        # Compute domination relationships for each individual
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
        """Determine if individual a dominates individual b (all objectives minimized)"""
        if a.objectives is None or b.objectives is None:
            return False

        # a dominates b if a is not worse in all objectives and strictly better in at least one
        not_worse = all(a_obj <= b_obj for a_obj, b_obj in zip(a.objectives, b.objectives))
        better = any(a_obj < b_obj for a_obj, b_obj in zip(a.objectives, b.objectives))

        return not_worse and better

    def calculate_crowding_distance(self, front: List[Chromosome]):
        """Calculate crowding distance"""
        if not front:
            return

        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0.0

        # Calculate for each objective
        for obj_idx in range(len(front[0].objectives)):
            # Sort by current objective value
            front.sort(key=lambda x: x.objectives[obj_idx])

            # Boundary individuals have infinite crowding distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Get range of objective values
            min_obj = front[0].objectives[obj_idx]
            max_obj = front[-1].objectives[obj_idx]
            obj_range = max_obj - min_obj

            if obj_range == 0:
                continue

            # Calculate crowding distance for interior individuals
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                                                      front[i + 1].objectives[obj_idx] - front[i - 1].objectives[
                                                  obj_idx]
                                              ) / obj_range

    def selection(self, population: List[Chromosome]) -> List[Chromosome]:
        """Tournament selection"""
        selected = []

        while len(selected) < len(population):
            # Randomly select two individuals
            candidates = random.sample(population, 2)

            # Tournament: prefer lower rank, if tie then larger crowding distance
            if candidates[0].rank < candidates[1].rank:
                selected.append(candidates[0])
            elif candidates[0].rank > candidates[1].rank:
                selected.append(candidates[1])
            else:
                # Same rank, select the one with larger crowding distance
                if candidates[0].crowding_distance > candidates[1].crowding_distance:
                    selected.append(candidates[0])
                else:
                    selected.append(candidates[1])

        return selected

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Crossover operations: crossover for operation sequence and machine assignment"""

        # 1. Operation sequence crossover (POX: Precedence Preserving Order-based Crossover)
        def pox_crossover(p1_seq, p2_seq):
            # Randomly select some jobs
            job_ids = list(self.job_ops_map.keys())
            selected_jobs = random.sample(job_ids, random.randint(1, len(job_ids) // 2))

            # Create child sequences
            child1_seq = []
            child2_seq = []

            # Record positions of each job in parents
            job_positions_p1 = defaultdict(list)
            for idx, (job_id, op_id) in enumerate(p1_seq):
                job_positions_p1[job_id].append((idx, (job_id, op_id)))

            job_positions_p2 = defaultdict(list)
            for idx, (job_id, op_id) in enumerate(p2_seq):
                job_positions_p2[job_id].append((idx, (job_id, op_id)))

            # Build child1
            # First copy selected jobs' operations from parent1
            for job_id in selected_jobs:
                for _, op in job_positions_p1[job_id]:
                    child1_seq.append(op)

            # Then copy unselected jobs' operations from parent2, preserving order
            for op in p2_seq:  # iterate directly over operations
                job_id = op[0]
                if job_id not in selected_jobs:
                    child1_seq.append(op)

            # Build child2
            # First copy selected jobs' operations from parent2
            for job_id in selected_jobs:
                for _, op in job_positions_p2[job_id]:
                    child2_seq.append(op)

            # Then copy unselected jobs' operations from parent1, preserving order
            for op in p1_seq:  # iterate directly over operations
                job_id = op[0]
                if job_id not in selected_jobs:
                    child2_seq.append(op)

            return child1_seq, child2_seq

        # 2. Machine assignment crossover (uniform crossover)
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

        # Perform crossover
        if random.random() < self.crossover_rate:
            child1_seq, child2_seq = pox_crossover(parent1.operation_sequence, parent2.operation_sequence)
            child1_machines, child2_machines = uniform_crossover(parent1.machine_assignment, parent2.machine_assignment)
        else:
            # No crossover, copy directly
            child1_seq, child2_seq = parent1.operation_sequence.copy(), parent2.operation_sequence.copy()
            child1_machines, child2_machines = parent1.machine_assignment.copy(), parent2.machine_assignment.copy()

        # Create child chromosomes
        child1 = Chromosome(child1_seq, child1_machines)
        child2 = Chromosome(child2_seq, child2_machines)

        return child1, child2

    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """Mutation operations"""
        mutated = chromosome.copy()

        # 1. Operation sequence mutation: swap mutation
        if random.random() < self.mutation_rate and len(mutated.operation_sequence) >= 2:
            # Randomly select two different positions
            idx1, idx2 = random.sample(range(len(mutated.operation_sequence)), 2)

            # Check if swapping maintains precedence constraints
            op1 = mutated.operation_sequence[idx1]
            op2 = mutated.operation_sequence[idx2]

            # If the two operations belong to different jobs, they can be swapped
            if op1[0] != op2[0]:
                mutated.operation_sequence[idx1], mutated.operation_sequence[idx2] = op2, op1

        # 2. Machine assignment mutation: randomly reset
        if random.random() < self.mutation_rate:
            # Randomly select an operation for machine reset
            idx = random.randint(0, len(mutated.operation_sequence) - 1)
            job_id, op_id = mutated.operation_sequence[idx]

            # Get operation object
            job = self.jobs[job_id]
            op = job.operations[op_id]

            # Randomly select a machine from available machines
            available_machines = list(op.machine_times.keys())
            if available_machines:
                mutated.machine_assignment[idx] = random.choice(available_machines)

        return mutated

    def create_offspring(self, parents: List[Chromosome]) -> List[Chromosome]:
        """Generate offspring population"""
        offspring = []

        # Ensure even number of parents
        if len(parents) % 2 != 0:
            parents = parents[:-1]

        # Crossover and mutation
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            offspring.extend([child1, child2])

        return offspring

    def run_generation(self, generation: int):
        """Run one generation of evolution"""
        # Evaluate population
        self.evaluate_population()

        # Non-dominated sorting
        self.fronts = self.fast_non_dominated_sort(self.population)

        # Calculate crowding distance
        for front in self.fronts:
            self.calculate_crowding_distance(front)

        # Record generation data
        self.record_generation_data(generation)

        # Select parents
        parents = self.selection(self.population)

        # Generate offspring
        offspring = self.create_offspring(parents)

        # Combine parent and offspring populations
        combined_population = self.population + offspring

        # Evaluate combined population
        for chromosome in combined_population:
            if chromosome.objectives is None:
                chromosome.objectives = self.evaluate_chromosome(chromosome)

        # Non-dominated sorting on combined population
        new_fronts = self.fast_non_dominated_sort(combined_population)

        # Build next generation population
        next_population = []
        front_idx = 0

        while len(next_population) + len(new_fronts[front_idx]) <= self.population_size:
            # Calculate crowding distance for current front
            self.calculate_crowding_distance(new_fronts[front_idx])

            # Add entire front
            next_population.extend(new_fronts[front_idx])
            front_idx += 1

        # If more individuals are needed, sort the current front by crowding distance
        if len(next_population) < self.population_size:
            remaining_count = self.population_size - len(next_population)
            self.calculate_crowding_distance(new_fronts[front_idx])

            # Sort by crowding distance descending
            sorted_front = sorted(new_fronts[front_idx],
                                  key=lambda x: x.crowding_distance,
                                  reverse=True)

            next_population.extend(sorted_front[:remaining_count])

        # Update population
        self.population = next_population

        # Update best solutions
        self.update_best_solutions()

    def record_generation_data(self, generation: int):
        """Record data for one generation for visualization"""
        # Calculate average objective values
        if self.population:
            makespans = [c.objectives[0] for c in self.population]
            load_balances = [c.objectives[1] for c in self.population]
            tardinesses = [c.objectives[2] for c in self.population]

            avg_makespan = np.mean(makespans)
            avg_load_balance = np.mean(load_balances)
            avg_tardiness = np.mean(tardinesses)

            # Find Pareto front
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
        """Update best solutions (Pareto front)"""
        if self.fronts and self.fronts[0]:
            # Get first front (non-dominated solutions)
            pareto_front = self.fronts[0]

            # Clear and update best solutions
            self.best_solutions = [c.copy() for c in pareto_front]

    # Main algorithm flow

    def run(self):
        """Run NSGA-II algorithm"""
        print("=" * 80)
        print("NSGA-II Algorithm Started")
        print("=" * 80)
        print(f"Population size: {self.population_size}")
        print(f"Maximum generations: {self.max_generations}")
        print(f"Total number of operations: {self.total_operations}")
        print(f"Number of jobs: {len(self.jobs)}")
        print(f"Number of machines: {len(self.machines)}")

        # Initialize population
        print("\nInitializing population...")
        self.initialize_population()

        # Main loop
        print("\nStarting evolution...")
        for generation in range(self.max_generations):
            self.run_generation(generation)

            # Print progress every 10 generations
            if generation % 10 == 0:
                pareto_count = len(self.fronts[0]) if self.fronts and self.fronts[0] else 0
                avg_makespan = self.visualization_data['avg_makespan'][-1]
                print(f"  Generation {generation:3d}: Pareto front size={pareto_count:3d}, "
                      f"Avg makespan={avg_makespan:.2f}")

        # Final evaluation
        self.evaluate_population()
        self.fronts = self.fast_non_dominated_sort(self.population)
        self.update_best_solutions()

        print("\nEvolution completed!")
        print(f"Final Pareto front size: {len(self.best_solutions)}")

        return self.best_solutions

    # Result output and visualization

    def print_results(self):
        """Print scheduling results"""
        if not self.best_solutions:
            print("No optimal solutions found")
            return

        print("=" * 80)
        print("NSGA-II Scheduling Results - Pareto Front Solutions")
        print("=" * 80)

        # Sort Pareto front solutions by makespan
        sorted_solutions = sorted(self.best_solutions,
                                  key=lambda x: x.objectives[0])

        for i, solution in enumerate(sorted_solutions[:5]):  # Show only top 5 solutions
            print(f"\nSolution #{i + 1}:")
            print(f"  Makespan: {solution.objectives[0]:.2f}")
            print(f"  Load balance: {solution.objectives[1]:.4f}")
            print(f"  Total tardiness: {solution.objectives[2]:.2f}")

            # Decode and print schedule details
            op_completion, machine_schedule, job_completion = self.decode_chromosome(solution)

            print(f"\n  Job completion times:")
            for job in self.jobs:
                completion = job_completion.get(job.job_id, 0)
                tardiness = max(0, completion - job.due_date)
                status = "Tardy" if tardiness > 0 else "On time"
                print(f"    Job{job.job_id}: completion={completion:.1f}, "
                      f"due date={job.due_date}, tardiness={tardiness:.1f}, status={status}")

    def visualize_convergence(self, save_path=None):
        """Visualize convergence process"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Average objective convergence curves
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

        # 2. Pareto front 2D scatter plot
        ax4 = axes[1, 1]

        if self.best_solutions:
            makespans = [s.objectives[0] for s in self.best_solutions]
            load_balances = [s.objectives[1] for s in self.best_solutions]
            tardinesses = [s.objectives[2] for s in self.best_solutions]

            # Normalize for color mapping
            norm_makespan = (makespans - np.min(makespans)) / (np.max(makespans) - np.min(makespans) + 1e-10)

            scatter = ax4.scatter(makespans, load_balances, c=norm_makespan,
                                  cmap='viridis', s=50, alpha=0.7, edgecolors='k')

            # Mark best solutions
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
            print(f"Convergence curve saved to: {save_path}")

        plt.show()

    def visualize_pareto_front_3d(self, save_path=None):
        """3D visualization of Pareto front"""
        if not self.best_solutions:
            print("No Pareto front solutions to visualize")
            return

        # Extract objective values
        makespans = [s.objectives[0] for s in self.best_solutions]
        load_balances = [s.objectives[1] for s in self.best_solutions]
        tardinesses = [s.objectives[2] for s in self.best_solutions]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create scatter plot
        scatter = ax.scatter(makespans, load_balances, tardinesses,
                             c=tardinesses, cmap='viridis', s=50,
                             alpha=0.7, edgecolors='k')

        # Mark best solutions
        for i, (x, y, z) in enumerate(zip(makespans[:5], load_balances[:5], tardinesses[:5])):
            ax.text(x, y, z, f'#{i+1}', fontsize=10, fontweight='bold')

        ax.set_xlabel('Makespan', fontsize=12)
        ax.set_ylabel('Load Balance', fontsize=12)
        ax.set_zlabel('Total Tardiness', fontsize=12)
        ax.set_title('3D Pareto Front', fontsize=14)

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Total Tardiness', pad=0.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D Pareto front saved to: {save_path}")

        plt.show()

    def visualize_best_schedule_gantt(self, solution_idx: int = 0, save_path=None):
        """Visualize Gantt chart of the best schedule"""
        if not self.best_solutions or solution_idx >= len(self.best_solutions):
            print(f"Solution #{solution_idx} not available")
            return

        solution = self.best_solutions[solution_idx]
        _, machine_schedule, _ = self.decode_chromosome(solution)

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.jobs)))
        job_colors = {job.job_id: colors[i] for i, job in enumerate(self.jobs)}

        # Assign schedule for each machine
        for machine_idx, machine_id in enumerate(sorted(machine_schedule.keys())):
            schedules = machine_schedule[machine_id]
            schedules.sort(key=lambda x: x[0])  # sort by start time

            for start, end, (job_id, op_id) in schedules:
                color = job_colors[job_id]
                ax.barh(machine_idx, end - start, left=start, height=0.6,
                        color=color, edgecolor='black')

        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_yticks(range(len(machine_schedule)))
        ax.set_yticklabels([f'Machine {mid}' for mid in sorted(machine_schedule.keys())])
        ax.set_title(f'NSGA-II Best Schedule #{solution_idx+1} - Gantt Chart', fontsize=14)

        # Add legend
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=job_colors[job.job_id], label=f'Job {job.job_id}')
                          for job in self.jobs]
        ax.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()
        plt.grid(axis='x', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved to: {save_path}")

        plt.show()


# File reading module

def read_fjsp_instance(file_path: str) -> Tuple[List[Job], List[int]]:
    """
    Read standard MOFJSP data file (extended Brandimarte format)
    File format:
        First line: number of jobs   number of machines
        Next N lines: operation information for each job
            - first integer: number of operations O for this job
            - subsequent O groups: each starts with k, followed by k pairs (machine ID, processing time)
        Then one line: N integers, due date for each job
        Additional lines (such as flexibility matrix, etc.) are ignored
    Returns: (list of jobs, list of machine IDs)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Filter out empty lines and comment lines
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        exit(1)

    if len(lines) < 2:
        raise ValueError("File format error: insufficient lines")

    # First line: number of jobs, number of machines
    num_jobs, num_machines = map(int, lines[0].split())
    machine_ids = list(range(num_machines))

    # Next num_jobs lines are job operation data
    job_lines = lines[1:1 + num_jobs]
    if len(job_lines) != num_jobs:
        raise ValueError(f"File format error: expected {num_jobs} job lines, got {len(job_lines)}")

    # Next line is due dates (must exist)
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
                raise ValueError(f"Incomplete data for job {job_idx}")
            k = nums[idx]
            idx += 1
            machine_times = {}
            for _ in range(k):
                if idx + 1 >= len(nums):
                    raise ValueError(f"Insufficient machine data for job {job_idx}, operation {op_idx}")
                machine_id = nums[idx] - 1  # convert to 0-index
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


# Main function

def main():
    """Main function: read file, run NSGA-II algorithm and display results"""
    print("=" * 80)
    print("NSGA-II Algorithm: Multi-objective Flexible Job Shop Scheduling (File-driven)")
    print("=" * 80)

    # Specify data file path (can be modified or passed via command line)
    file_path = "../mo_fjsp_instances/mo_fjsp_000_small_train.txt"  # ensure file exists

    # 1. Read scheduling instance from file
    print("\nReading scheduling instance...")
    try:
        jobs, machine_ids = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Number of jobs: {len(jobs)}")
    print(f"Number of machines: {len(machine_ids)}")

    total_ops = sum(len(job.operations) for job in jobs)
    print(f"Total number of operations: {total_ops}")

    # 2. Create NSGA-II scheduler
    print("\nConfiguring NSGA-II parameters...")
    scheduler = NSGA2Scheduler(
        jobs=jobs,
        machines=machine_ids,
        population_size=50,  # smaller population for faster execution
        max_generations=100,  # fewer generations for demonstration
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    # 3. Run NSGA-II algorithm
    print("\nRunning NSGA-II algorithm...")
    best_solutions = scheduler.run()

    # 4. Print results
    print("\n" + "=" * 80)
    print("Scheduling Results Analysis")
    print("=" * 80)
    scheduler.print_results()

    # 5. Generate visualizations and save to current directory
    print("\nGenerating visualizations...")
    scheduler.visualize_convergence(save_path="../Meta_Heuristic/convergence.png")
    scheduler.visualize_pareto_front_3d(save_path="../Meta_Heuristic/pareto_front_3d.png")
    scheduler.visualize_best_schedule_gantt(0, save_path="../Meta_Heuristic/gantt_chart.png")

    # 6. Performance statistics
    if best_solutions:
        print("\n" + "=" * 80)
        print("Pareto Front Statistics")
        print("=" * 80)

        makespans = [s.objectives[0] for s in best_solutions]
        load_balances = [s.objectives[1] for s in best_solutions]
        tardinesses = [s.objectives[2] for s in best_solutions]

        print(f"Number of Pareto solutions: {len(best_solutions)}")
        print(f"Makespan range: {min(makespans):.2f} - {max(makespans):.2f}")
        print(f"Load balance range: {min(load_balances):.4f} - {max(load_balances):.4f}")
        print(f"Total tardiness range: {min(tardinesses):.2f} - {max(tardinesses):.2f}")

        # Find best solution for each objective
        best_makespan_idx = np.argmin(makespans)
        best_balance_idx = np.argmin(load_balances)
        best_tardiness_idx = np.argmin(tardinesses)

        print(f"\nBest solutions per objective:")
        print(f"  Best makespan solution: makespan={makespans[best_makespan_idx]:.2f}, "
              f"balance={load_balances[best_makespan_idx]:.4f}, "
              f"tardiness={tardinesses[best_makespan_idx]:.2f}")
        print(f"  Best load balance solution: makespan={makespans[best_balance_idx]:.2f}, "
              f"balance={load_balances[best_balance_idx]:.4f}, "
              f"tardiness={tardinesses[best_balance_idx]:.2f}")
        print(f"  Best tardiness solution: makespan={makespans[best_tardiness_idx]:.2f}, "
              f"balance={load_balances[best_tardiness_idx]:.4f}, "
              f"tardiness={tardinesses[best_tardiness_idx]:.2f}")

    print("\nNSGA-II scheduling completed!")


# Run main function

if __name__ == "__main__":
    main()