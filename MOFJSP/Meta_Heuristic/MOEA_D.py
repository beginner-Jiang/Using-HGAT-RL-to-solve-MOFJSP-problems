"""
Metaheuristic Algorithm Implementation: Multi-objective Evolutionary Algorithm based on
Decomposition (MOEA/D)

The encoding, decoding, objective evaluation and the genetic operators (POX crossover,
uniform machine crossover and mutation) are inherited from the NSGA-II implementation in
``NSGA_II.py``; MOEA/D replaces the non-dominated-sorting based environmental selection by a
neighbourhood based replacement driven by uniformly distributed weight vectors and a
scalarizing decomposition function.

Reference:
    Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
    Decomposition," IEEE Transactions on Evolutionary Computation, vol. 11, no. 6,
    pp. 712-731, 2007.
"""
import os
import sys
import math
import random
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
# Without a display session (background run, stdout redirected to a file, headless server)
# the Qt backend cannot be initialised and the process would crash; fall back to the file-only
# Agg backend and keep the default backend for interactive runs.
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The representation and the evolutionary operators are shared with NSGA-II so that the two
# metaheuristics cannot drift apart. The import supports both the project-package layout
try:
    from Meta_Heuristic.NSGA_II import (
        Operation,
        Job,
        Chromosome,
        NSGA2Scheduler,
        read_fjsp_instance,
    )
except ImportError:
    from NSGA_II import (
        Operation,
        Job,
        Chromosome,
        NSGA2Scheduler,
        read_fjsp_instance,
    )


class MOEADScheduler(NSGA2Scheduler):
    """
    MOEA/D based scheduler.

    Every individual is attached to one subproblem, i.e. one weight vector. Offspring are
    generated from the neighbourhood of a subproblem and replace the neighbours whose
    scalarized objective is not better, which keeps the population spread along the whole
    Pareto front.
    """

    # Name used by the inherited visualisation helpers of NSGA2Scheduler
    algorithm_name = "MOEA/D"

    def __init__(self, jobs: List[Job], machines: List[int],
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 neighbor_size: int = 10,
                 decomposition_method: str = 'tchebycheff',
                 max_replacements: int = 2):
        """
        Initialize the MOEA/D scheduler.

        Args:
            jobs: list of jobs
            machines: list of machine IDs
            population_size: number of subproblems (weight vectors)
            max_generations: maximum number of generations
            crossover_rate: crossover probability
            mutation_rate: mutation probability
            neighbor_size: number of neighbours T of each weight vector
            decomposition_method: 'tchebycheff' (default) or 'ws' (weighted sum)
            max_replacements: maximal number of neighbours replaced by one offspring
        """
        super().__init__(jobs, machines, population_size, max_generations,
                         crossover_rate, mutation_rate)

        self.neighbor_size = int(max(2, min(neighbor_size, population_size)))
        self.decomposition_method = str(decomposition_method).lower()
        self.max_replacements = int(max(1, max_replacements))

        # Uniformly distributed weight vectors, one per subproblem
        self.weight_vectors = self._generate_weight_vectors(population_size, 3)
        self.population_size = len(self.weight_vectors)

        # Neighbourhood structure: for each subproblem the indices of the T closest ones
        self.neighbors = self._compute_neighborhoods(self.weight_vectors, self.neighbor_size)

        # Ideal point z* and nadir point (element-wise minimum / maximum of all evaluated
        # objectives), required by the scalarizing function
        self.ideal_point = np.full(3, np.inf, dtype=float)
        self.nadir_point = np.full(3, -np.inf, dtype=float)

        # External population (archive of non-dominated solutions). It is exposed as ``ep``
        # to keep the interface expected by the comparative experiment.
        self.ep: List[Chromosome] = []

    # Weight vectors and neighbourhood

    @staticmethod
    def _generate_weight_vectors(n_points: int, n_obj: int) -> np.ndarray:
        """
        Generate ``n_points`` uniformly distributed weight vectors on the unit simplex with
        the systematic (Das-Dennis) lattice method. The lattice is enlarged until it holds at
        least ``n_points`` vectors, from which the best spread subset is selected by farthest
        point sampling so that exactly ``n_points`` subproblems are created.
        """
        if n_obj != 3:
            raise ValueError("Only three-objective problems are supported.")

        # Smallest number of divisions H such that the lattice holds at least n_points vectors
        divisions = 1
        while math.comb(divisions + n_obj - 1, n_obj - 1) < n_points:
            divisions += 1

        lattice = []
        for i in range(divisions + 1):
            for j in range(divisions + 1 - i):
                k = divisions - i - j
                lattice.append([i / divisions, j / divisions, k / divisions])
        lattice = np.array(lattice, dtype=float)

        if len(lattice) == n_points:
            return lattice

        # Farthest point sampling, started from the vector closest to the centroid, keeps
        # the selected subset as evenly spread as possible.
        centroid = np.full(n_obj, 1.0 / n_obj)
        start = int(np.argmin(np.linalg.norm(lattice - centroid, axis=1)))
        selected = [start]
        dist = np.linalg.norm(lattice - lattice[start], axis=1)
        while len(selected) < n_points:
            idx = int(np.argmax(dist))
            selected.append(idx)
            dist = np.minimum(dist, np.linalg.norm(lattice - lattice[idx], axis=1))
        return lattice[sorted(selected)]

    @staticmethod
    def _compute_neighborhoods(weights: np.ndarray, t: int) -> np.ndarray:
        """Return, for every weight vector, the indices of its ``t`` closest neighbours."""
        dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
        return np.argsort(dist, axis=1)[:, :t]

    # Decomposition and archive maintenance

    def _decomposition_value(self, objectives, weight) -> float:
        """
        Scalarizing function g(x | w, z*) used by the replacement rule.

        'tchebycheff': max_j w_j * |f_j(x) - z*_j|  (default, handles non-convex fronts)
        'ws'         : sum_j w_j * f_j(x)           (weighted sum)
        """
        obj = np.asarray(objectives, dtype=float)

        if self.decomposition_method == 'ws':
            return float(np.dot(weight, obj))

        # Before the first evaluation the ideal point is still infinite. In that case only
        # the (finite) objective values themselves can be used for the comparison.
        ideal = np.where(np.isfinite(self.ideal_point), self.ideal_point, 0.0)
        # Normalise by the ideal/nadir span: the three objectives differ by two orders of
        # magnitude (tardiness against load balance), and an unnormalised Tchebycheff scalar
        # is decided by the largest objective alone.
        nadir = np.where(np.isfinite(self.nadir_point), self.nadir_point, ideal)
        span = np.maximum(nadir - ideal, 1e-12)
        normalized = (obj - ideal) / span
        return float(np.max(np.asarray(weight, dtype=float) * np.abs(normalized)))

    def _update_ideal_point(self, objectives):
        """Update the ideal (and nadir) point with a newly evaluated solution."""
        obj = np.asarray(objectives, dtype=float)
        self.ideal_point = np.minimum(self.ideal_point, obj)
        self.nadir_point = np.maximum(self.nadir_point, obj)

    def _update_external_population(self):
        """Keep the non-dominated solutions of the union of the archive and the population."""
        candidates = self.ep + list(self.population)
        fronts = self.fast_non_dominated_sort(candidates)
        first_front = fronts[0] if fronts else []

        # Remove duplicates (identical objective vectors) to keep the archive compact
        archive, seen = [], set()
        for chromosome in first_front:
            key = tuple(round(float(v), 6) for v in chromosome.objectives)
            if key not in seen:
                seen.add(key)
                archive.append(chromosome.copy())
        self.ep = archive

    # Main algorithm flow

    def run_generation(self, generation: int):
        """Run one generation of MOEA/D."""
        # Evaluate the current population
        self.evaluate_population()

        # Non-dominated sorting is only performed to keep the reporting/visualisation
        # interface of the base class (``record_generation_data`` expects ``self.fronts``).
        self.fronts = self.fast_non_dominated_sort(self.population)
        self.record_generation_data(generation)

        for i in range(self.population_size):
            # Two parents are drawn from the neighbourhood of subproblem i
            neighbor_idx = self.neighbors[i]
            if len(neighbor_idx) < 2:
                neighbor_idx = np.arange(self.population_size)
            k, l = np.random.choice(neighbor_idx, size=2, replace=False)

            # Offspring generation with the shared NSGA-II operators
            child = self.crossover(self.population[k], self.population[l])[0]
            child = self.mutation(child)
            child.objectives = self.evaluate_chromosome(child)

            self._update_ideal_point(child.objectives)

            # Replacement: the offspring replaces those neighbours for which it solves the
            # corresponding scalarizing subproblem at least as well
            replaced = 0
            for j in np.random.permutation(neighbor_idx):
                if self._decomposition_value(child.objectives, self.weight_vectors[j]) <= \
                        self._decomposition_value(self.population[j].objectives, self.weight_vectors[j]):
                    self.population[j] = child.copy()
                    replaced += 1
                    if replaced >= self.max_replacements:
                        break

        self._update_external_population()
        self.update_best_solutions()

    def update_best_solutions(self):
        """Expose the external population (archive) as the set of best solutions."""
        if self.ep:
            self.best_solutions = [c.copy() for c in self.ep]

    def run(self):
        """Run the MOEA/D algorithm"""
        print("=" * 80)
        print("MOEA/D Algorithm Started")
        print("=" * 80)
        print(f"Population size (subproblems): {self.population_size}")
        print(f"Maximum generations: {self.max_generations}")
        print(f"Neighbourhood size T: {self.neighbor_size}")
        print(f"Decomposition method: {self.decomposition_method}")
        print(f"Total number of operations: {self.total_operations}")
        print(f"Number of jobs: {len(self.jobs)}")
        print(f"Number of machines: {len(self.machines)}")

        # Initialize population
        print("\nInitializing population...")
        self.initialize_population()
        self.evaluate_population()
        for chromosome in self.population:
            self._update_ideal_point(chromosome.objectives)
        self._update_external_population()
        self.update_best_solutions()

        # Main loop
        print("\nStarting evolution...")
        for generation in range(self.max_generations):
            self.run_generation(generation)

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"  Generation {generation:3d}: Archive size={len(self.ep):3d}, "
                      f"Avg makespan={self.visualization_data['avg_makespan'][-1]:.2f}")

        # Final evaluation
        self.evaluate_population()
        self._update_external_population()
        self.update_best_solutions()

        print("\nEvolution completed!")
        print(f"Final archive (Pareto front) size: {len(self.best_solutions)}")

        return self.best_solutions

    # Result output

    def print_results(self):
        """Print scheduling results"""
        if not self.best_solutions:
            print("No optimal solutions found")
            return

        print("=" * 80)
        print("MOEA/D Scheduling Results - Pareto Front Solutions")
        print("=" * 80)

        makespans = [s.objectives[0] for s in self.best_solutions]
        load_balances = [s.objectives[1] for s in self.best_solutions]
        tardinesses = [s.objectives[2] for s in self.best_solutions]

        print(f"Number of Pareto solutions: {len(self.best_solutions)}")
        print(f"Makespan range: {min(makespans):.2f} - {max(makespans):.2f}")
        print(f"Load balance range: {min(load_balances):.4f} - {max(load_balances):.4f}")
        print(f"Total tardiness range: {min(tardinesses):.2f} - {max(tardinesses):.2f}")

        best_makespan_idx = int(np.argmin(makespans))
        best_balance_idx = int(np.argmin(load_balances))
        best_tardiness_idx = int(np.argmin(tardinesses))

        print("\nBest solutions per objective:")
        print(f"  Best makespan solution: makespan={makespans[best_makespan_idx]:.2f}, "
              f"balance={load_balances[best_makespan_idx]:.4f}, "
              f"tardiness={tardinesses[best_makespan_idx]:.2f}")
        print(f"  Best load balance solution: makespan={makespans[best_balance_idx]:.2f}, "
              f"balance={load_balances[best_balance_idx]:.4f}, "
              f"tardiness={tardinesses[best_balance_idx]:.2f}")
        print(f"  Best tardiness solution: makespan={makespans[best_tardiness_idx]:.2f}, "
              f"balance={load_balances[best_tardiness_idx]:.4f}, "
              f"tardiness={tardinesses[best_tardiness_idx]:.2f}")


# Main function

def main():
    """Main function: read file, run MOEA/D algorithm and display results"""
    print("=" * 80)
    print("MOEA/D Algorithm: Multi-objective Flexible Job Shop Scheduling (File-driven)")
    print("=" * 80)

    # Specify data file path (can be modified or passed via command line)
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "DataSet", "mo_fjsp_instances", "mo_fjsp_000_small_train.txt")  # ensure file exists

    # 1. Read scheduling instance from file
    print("\nReading scheduling instance...")
    try:
        jobs, machine_ids = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Number of jobs: {len(jobs)}")
    print(f"Number of machines: {len(machine_ids)}")
    print(f"Total number of operations: {sum(len(job.operations) for job in jobs)}")

    # 2. Create MOEA/D scheduler
    print("\nConfiguring MOEA/D parameters...")
    scheduler = MOEADScheduler(
        jobs=jobs,
        machines=machine_ids,
        population_size=50,
        max_generations=100,
        crossover_rate=0.9,
        mutation_rate=0.1,
        neighbor_size=10,
        decomposition_method='tchebycheff'
    )

    # 3. Run MOEA/D algorithm
    print("\nRunning MOEA/D algorithm...")
    scheduler.run()

    # 4. Print results
    print("\n" + "=" * 80)
    print("Scheduling Results Analysis")
    print("=" * 80)
    scheduler.print_results()

    # 5. Generate visualizations and save them next to the other experiment outputs
    print("\nGenerating visualizations...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "Figure_And_File", "Meta_Heuristic", "MOEA_D")
    os.makedirs(output_dir, exist_ok=True)
    scheduler.visualize_convergence(save_path=os.path.join(output_dir, "convergence.png"))
    scheduler.visualize_pareto_front_3d(save_path=os.path.join(output_dir, "pareto_front_3d.png"))
    scheduler.visualize_best_schedule_gantt(0, save_path=os.path.join(output_dir, "gantt_chart.png"))

    print("\nMOEA/D scheduling completed!")


# Run main function

if __name__ == "__main__":
    main()

