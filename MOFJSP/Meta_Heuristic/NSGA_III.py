"""
Metaheuristic Algorithm Implementation: Non-dominated Sorting Genetic Algorithm III
(NSGA-III)

NSGA-III extends NSGA-II to many-objective optimisation problems by replacing the crowding
distance by a reference-point based niche preservation operator. The encoding, decoding,
objective evaluation and the genetic operators are inherited from ``NSGA_II.py``; only the
environmental selection is re-implemented.

Reference:
    K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
    Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box
    Constraints," IEEE Transactions on Evolutionary Computation, vol. 18, no. 4,
    pp. 577-601, 2014.
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


class NSGA3Scheduler(NSGA2Scheduler):
    """
    NSGA-III based scheduler.

    The population is filled front by front as in NSGA-II. The front that does not fit
    completely is truncated with the niche preservation operator: every solution is
    associated with the closest reference line and reference points with fewer associated
    solutions are preferred, which preserves diversity also with three (or more) objectives.
    """

    # Name used by the inherited visualisation helpers of NSGA2Scheduler
    algorithm_name = "NSGA-III"

    def __init__(self, jobs: List[Job], machines: List[int],
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 num_divisions: int = None):
        """
        Initialize the NSGA-III scheduler.

        Args:
            jobs: list of jobs
            machines: list of machine IDs
            population_size: requested number of reference points / population size
            max_generations: maximum number of generations
            crossover_rate: crossover probability
            mutation_rate: mutation probability
            num_divisions: number of divisions p of the reference-point lattice; if it is
                None the smallest lattice containing at least ``population_size`` points is
                used (the effective population size is the lattice size, as in the original
                NSGA-III definition).
        """
        super().__init__(jobs, machines, population_size, max_generations,
                         crossover_rate, mutation_rate)

        # Reference points on the unit simplex, one per population slot
        self.ref_points = self._generate_reference_points(population_size, 3, num_divisions)
        self.population_size = len(self.ref_points)

        # Normalisation data of the current generation
        self.ideal_point = np.zeros(3, dtype=float)
        self.intercepts = np.ones(3, dtype=float)

    # Reference points and normalisation

    @staticmethod
    def _generate_reference_points(n_points: int, n_obj: int, num_divisions: int = None) -> np.ndarray:
        """
        Generate the Das-Dennis reference points on the unit simplex.

        For a three-objective problem the lattice with p divisions holds
        (p+1)(p+2)/2 points; p is increased until the lattice is large enough.
        """
        if n_obj != 3:
            raise ValueError("Only three-objective problems are supported.")

        if num_divisions is None:
            divisions = 1
            while math.comb(divisions + n_obj - 1, n_obj - 1) < n_points:
                divisions += 1
        else:
            divisions = int(num_divisions)

        points = []
        for i in range(divisions + 1):
            for j in range(divisions + 1 - i):
                k = divisions - i - j
                points.append([i / divisions, j / divisions, k / divisions])
        return np.array(points, dtype=float)

    def _find_extreme_points(self, objectives: np.ndarray) -> List[int]:
        """
        Return the indices of the M extreme solutions, obtained by minimising the
        achievement scalarizing function with the corresponding axis weight vector.
        """
        n_obj = objectives.shape[1]
        translated = objectives - objectives.min(axis=0)
        extreme = []
        for j in range(n_obj):
            weight = np.full(n_obj, 1e-6)
            weight[j] = 1.0
            asf = np.max(translated / weight, axis=1)
            extreme.append(int(np.argmin(asf)))
        return extreme

    def _compute_intercepts(self, objectives: np.ndarray, extreme_idx: List[int]) -> np.ndarray:
        """
        Compute the intercepts a_j of the hyperplane spanned by the M extreme points with the
        objective axes; they are used to normalise the objective values.
        """
        n_obj = objectives.shape[1]
        translated = objectives - objectives.min(axis=0)
        extreme_points = translated[extreme_idx]

        try:
            # The hyperplane sum_j (f_j - z_j^min) / a_j = 1 passes through the M extreme
            # points. With u_j = 1 / a_j the system is extreme_points @ u = 1, where the
            with np.errstate(divide='ignore', invalid='ignore'):
                intercepts = 1.0 / np.linalg.solve(extreme_points, np.ones(n_obj))
        except np.linalg.LinAlgError:
            intercepts = translated.max(axis=0)

        if not np.all(np.isfinite(intercepts)) or np.any(intercepts < 1e-12):
            intercepts = translated.max(axis=0)
        intercepts = np.where(intercepts < 1e-12, 1.0, intercepts)
        return intercepts

    def _normalize(self, objectives: np.ndarray) -> np.ndarray:
        """Normalise objectives with respect to the ideal point and the intercepts."""
        return (objectives - self.ideal_point) / self.intercepts

    def _associate(self, normalized: np.ndarray, ref_points: np.ndarray):
        """
        Associate every normalised solution with the closest reference line (the line through
        the origin with the direction of the reference point).

        Returns:
            niche: index of the associated reference point for every solution
            dist: perpendicular distance to the associated reference line
        """
        ref_norm_sq = np.sum(ref_points ** 2, axis=1)
        projection = normalized @ ref_points.T
        projection = projection / np.maximum(ref_norm_sq[None, :], 1e-12)
        closest = projection[:, :, None] * ref_points[None, :, :]
        dist_matrix = np.linalg.norm(normalized[:, None, :] - closest, axis=2)
        niche = np.argmin(dist_matrix, axis=1)
        return niche, dist_matrix

    # Environmental selection

    def _niching_select(self, last_front: List[Chromosome], selected: List[Chromosome],
                        num_required: int) -> List[Chromosome]:
        """
        Select ``num_required`` solutions from ``last_front`` with the niche preservation
        operator of NSGA-III.
        """
        if num_required <= 0:
            return []

        front_objs = np.array([c.objectives for c in last_front], dtype=float)
        front_norm = self._normalize(front_objs)
        niche, dist_matrix = self._associate(front_norm, self.ref_points)

        # Number of already selected solutions associated with every reference point
        rho = np.zeros(len(self.ref_points), dtype=int)
        if selected:
            sel_objs = np.array([c.objectives for c in selected], dtype=float)
            sel_norm = self._normalize(sel_objs)
            sel_niche, _ = self._associate(sel_norm, self.ref_points)
            for n in sel_niche:
                rho[n] += 1

        # Solutions of the last front associated with every reference point, ordered by
        # ascending perpendicular distance
        candidates = [[] for _ in range(len(self.ref_points))]
        for i in range(len(last_front)):
            candidates[niche[i]].append(i)
        for j in range(len(self.ref_points)):
            candidates[j].sort(key=lambda i: dist_matrix[i, j])

        chosen = []

        # First pass: reference points that are not yet represented in the new population
        for j in range(len(self.ref_points)):
            if len(chosen) >= num_required:
                break
            if rho[j] == 0 and candidates[j]:
                chosen.append(candidates[j].pop(0))
                rho[j] += 1

        # Second pass: pick the least populated niches until enough solutions are selected
        while len(chosen) < num_required:
            available = [j for j in range(len(self.ref_points)) if candidates[j]]
            if not available:
                break
            j = min(available, key=lambda x: (rho[x], dist_matrix[candidates[x][0], x]))
            chosen.append(candidates[j].pop(0))
            rho[j] += 1

        return [last_front[i] for i in chosen]

    def _environmental_selection(self, combined: List[Chromosome]) -> List[Chromosome]:
        """Build the next population from the union of parents and offspring."""
        # Normalisation data are derived from the union of both populations
        all_objs = np.array([c.objectives for c in combined], dtype=float)
        self.ideal_point = all_objs.min(axis=0)
        self.intercepts = self._compute_intercepts(all_objs, self._find_extreme_points(all_objs))

        fronts = self.fast_non_dominated_sort(combined)

        next_population: List[Chromosome] = []
        for front in fronts:
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
            else:
                num_required = self.population_size - len(next_population)
                next_population.extend(self._niching_select(front, next_population, num_required))
                break
        return next_population

    # Main algorithm flow

    def run_generation(self, generation: int):
        """Run one generation of NSGA-III."""
        # Evaluate population and set the ranks required by the mating selection
        self.evaluate_population()
        self.fronts = self.fast_non_dominated_sort(self.population)
        for front in self.fronts:
            self.calculate_crowding_distance(front)
        self.record_generation_data(generation)

        # Mating selection and offspring generation (shared operators)
        parents = self.selection(self.population)
        offspring = self.create_offspring(parents)

        # Environmental selection on the union of parents and offspring
        combined = self.population + offspring
        for chromosome in combined:
            if chromosome.objectives is None:
                chromosome.objectives = self.evaluate_chromosome(chromosome)

        self.population = self._environmental_selection(combined)

        # Refresh the fronts of the surviving population (used for reporting)
        self.fronts = self.fast_non_dominated_sort(self.population)
        self.update_best_solutions()

    def run(self):
        """Run the NSGA-III algorithm"""
        print("=" * 80)
        print("NSGA-III Algorithm Started")
        print("=" * 80)
        print(f"Population size: {self.population_size}")
        print(f"Number of reference points: {len(self.ref_points)}")
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
                print(f"  Generation {generation:3d}: Pareto front size={pareto_count:3d}, "
                      f"Avg makespan={self.visualization_data['avg_makespan'][-1]:.2f}")

        # Final evaluation
        self.evaluate_population()
        self.fronts = self.fast_non_dominated_sort(self.population)
        self.update_best_solutions()

        print("\nEvolution completed!")
        print(f"Final Pareto front size: {len(self.best_solutions)}")

        return self.best_solutions

    def print_results(self):
        """Print scheduling results"""
        if not self.best_solutions:
            print("No optimal solutions found")
            return

        print("=" * 80)
        print("NSGA-III Scheduling Results - Pareto Front Solutions")
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
    """Main function: read file, run NSGA-III algorithm and display results"""
    print("=" * 80)
    print("NSGA-III Algorithm: Multi-objective Flexible Job Shop Scheduling (File-driven)")
    print("=" * 80)

    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "DataSet", "mo_fjsp_instances", "mo_fjsp_000_small_train.txt")  # ensure file exists

    print("\nReading scheduling instance...")
    try:
        jobs, machine_ids = read_fjsp_instance(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Number of jobs: {len(jobs)}")
    print(f"Number of machines: {len(machine_ids)}")
    print(f"Total number of operations: {sum(len(job.operations) for job in jobs)}")

    print("\nConfiguring NSGA-III parameters...")
    scheduler = NSGA3Scheduler(
        jobs=jobs,
        machines=machine_ids,
        population_size=50,
        max_generations=100,
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    print("\nRunning NSGA-III algorithm...")
    scheduler.run()

    print("\n" + "=" * 80)
    print("Scheduling Results Analysis")
    print("=" * 80)
    scheduler.print_results()

    print("\nGenerating visualizations...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "Figure_And_File", "Meta_Heuristic", "NSGA_III")
    os.makedirs(output_dir, exist_ok=True)
    scheduler.visualize_convergence(save_path=os.path.join(output_dir, "convergence.png"))
    scheduler.visualize_pareto_front_3d(save_path=os.path.join(output_dir, "pareto_front_3d.png"))
    scheduler.visualize_best_schedule_gantt(0, save_path=os.path.join(output_dir, "gantt_chart.png"))

    print("\nNSGA-III scheduling completed!")


# Run main function

if __name__ == "__main__":
    main()
