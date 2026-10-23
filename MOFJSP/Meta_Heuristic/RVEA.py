"""
Metaheuristic Algorithm Implementation: Reference Vector Guided Evolutionary Algorithm
(RVEA)

RVEA guides the search with a set of uniformly distributed reference vectors. Convergence
and diversity are balanced by the angle penalized distance (APD), whose penalty grows with
the generation counter so that diversity is emphasised in early generations and convergence
in late generations. The encoding, decoding, objective evaluation and the genetic operators
are inherited from ``NSGA_II.py``; only the environmental selection is re-implemented.

Reference:
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, "A Reference Vector Guided Evolutionary
    Algorithm for Many-Objective Optimization," IEEE Transactions on Evolutionary
    Computation, vol. 20, no. 5, pp. 773-791, 2016.
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

# The representation and the evolutionary operators are shared with NSGA-II so that the
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


class RVEAScheduler(NSGA2Scheduler):
    """
    RVEA based scheduler.

    Every individual is assigned to the reference vector with the smallest angle and the
    angle penalized distance decides which individual survives for each reference vector.
    """

    # Name used by the inherited visualisation helpers of NSGA2Scheduler
    algorithm_name = "RVEA"

    def __init__(self, jobs: List[Job], machines: List[int],
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.1,
                 alpha: float = 2.0,
                 reference_vectors: np.ndarray = None):
        """
        Initialize the RVEA scheduler.

        Args:
            jobs: list of jobs
            machines: list of machine IDs
            population_size: number of reference vectors / population size
            max_generations: maximum number of generations
            crossover_rate: crossover probability
            mutation_rate: mutation probability
            alpha: exponent controlling how fast the diversity penalty decreases
            reference_vectors: optional user supplied reference vectors; by default the
                Das-Dennis lattice with at least ``population_size`` points is used
        """
        super().__init__(jobs, machines, population_size, max_generations,
                         crossover_rate, mutation_rate)

        self.alpha = float(alpha)
        if reference_vectors is None:
            self.ref_vectors = self._generate_reference_vectors(population_size, 3)
        else:
            self.ref_vectors = np.asarray(reference_vectors, dtype=float)
        self.population_size = len(self.ref_vectors)

        # Smallest angle between each reference vector and all the other reference vectors
        self.gamma = self._compute_gamma(self.ref_vectors)

        # Normalisation data of the current generation
        self.ideal_point = np.zeros(3, dtype=float)
        self.nadir_point = np.ones(3, dtype=float)

    # Reference vectors

    @staticmethod
    def _generate_reference_vectors(n_points: int, n_obj: int) -> np.ndarray:
        """
        Generate the uniformly distributed reference vectors of the original RVEA: the
        Das-Dennis lattice on the unit simplex, i.e. exactly the reference-point set used by
        NSGA-III. The smallest lattice containing at least ``n_points`` points is used, so
        the effective population size is the lattice size (as in the original definition) and
        both many-objective baselines are guided by the same reference directions.

        Sampling the directions from a symmetric Dirichlet distribution instead would give a
        random, non-uniform set of directions, which is not the "uniformly distributed
        reference vectors" of the method that is being reproduced.
        """
        if n_obj != 3:
            raise ValueError("Only three-objective problems are supported.")

        divisions = 1
        while math.comb(divisions + n_obj - 1, n_obj - 1) < n_points:
            divisions += 1

        vectors = []
        for i in range(divisions + 1):
            for j in range(divisions + 1 - i):
                k = divisions - i - j
                vectors.append([i / divisions, j / divisions, k / divisions])
        return np.array(vectors, dtype=float)

    @staticmethod
    def _compute_gamma(vectors: np.ndarray) -> np.ndarray:
        """
        Return, for every reference vector, the smallest angle (in radians) to any other
        reference vector. It is the normalisation factor of the diversity penalty.
        """
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        cosine = np.clip(unit @ unit.T, -1.0, 1.0)
        angles = np.arccos(cosine)
        np.fill_diagonal(angles, np.inf)
        gamma = angles.min(axis=1)
        # Guard against duplicated reference vectors
        return np.where(np.isfinite(gamma) & (gamma > 1e-8), gamma, 1.0)

    # Angle penalized distance

    def _update_normalization(self, combined: List[Chromosome]):
        """Update the ideal and nadir points used to normalise the objective values."""
        objectives = np.array([c.objectives for c in combined], dtype=float)
        self.ideal_point = objectives.min(axis=0)
        self.nadir_point = objectives.max(axis=0)

    def _angles_to_reference_vectors(self, normalized: np.ndarray) -> np.ndarray:
        """Angle (in radians) between every normalised solution and every reference vector."""
        vectors = self.ref_vectors / np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        cosine = np.clip((normalized @ vectors.T) / (norms + 1e-12), -1.0, 1.0)
        return np.arccos(cosine)

    def _angle_penalized_distance(self, normalized: np.ndarray, generation: int) -> np.ndarray:
        """
        Compute the angle penalized distance APD_{i,j} of every solution with respect to
        every reference vector.

        APD_{i,j} = (1 + M * (t / t_max)^alpha * theta_{i,j} / gamma_j) * ||f'_i||
        """
        n_obj = normalized.shape[1]
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        angles = self._angles_to_reference_vectors(normalized)

        progress = generation / max(1, self.max_generations - 1)
        penalty = (1.0 + n_obj * (progress ** self.alpha) * angles / self.gamma[None, :])
        return penalty * norms

    def _environmental_selection(self, combined: List[Chromosome], generation: int) -> List[Chromosome]:
        """Select the next population from the union of parents and offspring."""
        self._update_normalization(combined)

        objectives = np.array([c.objectives for c in combined], dtype=float)
        span = np.where(self.nadir_point - self.ideal_point < 1e-12,
                        1.0, self.nadir_point - self.ideal_point)
        normalized = (objectives - self.ideal_point) / span

        apd = self._angle_penalized_distance(normalized, generation)
        angles = self._angles_to_reference_vectors(normalized)

        # Association follows the original RVEA: a solution is assigned to the reference
        # vector with the smallest angle, and only then does the APD decide which of the
        assignment = np.argmin(angles, axis=1)

        survivors = []
        for j in range(len(self.ref_vectors)):
            candidates = np.where(assignment == j)[0]
            if len(candidates) == 0:
                continue
            survivors.append(int(candidates[np.argmin(apd[candidates, j])]))

        # If some reference vectors stayed empty, fill the population with the remaining
        # solutions having the smallest APD values
        if len(survivors) < self.population_size:
            chosen = set(survivors)
            remaining = [i for i in range(len(combined)) if i not in chosen]
            remaining.sort(key=lambda i: float(np.min(apd[i])))
            survivors.extend(remaining[:self.population_size - len(survivors)])

        return [combined[i] for i in survivors]

    # Main algorithm flow

    def run_generation(self, generation: int):
        """Run one generation of RVEA."""
        self.evaluate_population()
        self.fronts = self.fast_non_dominated_sort(self.population)
        for front in self.fronts:
            self.calculate_crowding_distance(front)
        self.record_generation_data(generation)

        # Mating selection and offspring generation (shared operators)
        parents = self.selection(self.population)
        offspring = self.create_offspring(parents)

        combined = self.population + offspring
        for chromosome in combined:
            if chromosome.objectives is None:
                chromosome.objectives = self.evaluate_chromosome(chromosome)

        self.population = self._environmental_selection(combined, generation)

        self.fronts = self.fast_non_dominated_sort(self.population)
        self.update_best_solutions()

    def run(self):
        """Run the RVEA algorithm"""
        print("=" * 80)
        print("RVEA Algorithm Started")
        print("=" * 80)
        print(f"Population size: {self.population_size}")
        print(f"Number of reference vectors: {len(self.ref_vectors)}")
        print(f"Maximum generations: {self.max_generations}")
        print(f"Total number of operations: {self.total_operations}")
        print(f"Number of jobs: {len(self.jobs)}")
        print(f"Number of machines: {len(self.machines)}")

        print("\nInitializing population...")
        self.initialize_population()

        print("\nStarting evolution...")
        for generation in range(self.max_generations):
            self.run_generation(generation)

            if generation % 10 == 0:
                pareto_count = len(self.fronts[0]) if self.fronts and self.fronts[0] else 0
                print(f"  Generation {generation:3d}: Pareto front size={pareto_count:3d}, "
                      f"Avg makespan={self.visualization_data['avg_makespan'][-1]:.2f}")

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
        print("RVEA Scheduling Results - Pareto Front Solutions")
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
    """Main function: read file, run RVEA algorithm and display results"""
    print("=" * 80)
    print("RVEA Algorithm: Multi-objective Flexible Job Shop Scheduling (File-driven)")
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

    print("\nConfiguring RVEA parameters...")
    scheduler = RVEAScheduler(
        jobs=jobs,
        machines=machine_ids,
        population_size=50,
        max_generations=100,
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    print("\nRunning RVEA algorithm...")
    scheduler.run()

    print("\n" + "=" * 80)
    print("Scheduling Results Analysis")
    print("=" * 80)
    scheduler.print_results()

    print("\nGenerating visualizations...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "Figure_And_File", "Meta_Heuristic", "RVEA")
    os.makedirs(output_dir, exist_ok=True)
    scheduler.visualize_convergence(save_path=os.path.join(output_dir, "convergence.png"))
    scheduler.visualize_pareto_front_3d(save_path=os.path.join(output_dir, "pareto_front_3d.png"))
    scheduler.visualize_best_schedule_gantt(0, save_path=os.path.join(output_dir, "gantt_chart.png"))

    print("\nRVEA scheduling completed!")


# Run main function

if __name__ == "__main__":
    main()
