# -*- coding: utf-8 -*-
"""Standard genetic algorithm and tabu search for public FJSP instances.

The two methods operate on the classical two-part FJSP representation:

* an operation sequence (OS) that lists job identifiers in scheduling order;
* a machine assignment (MS) that stores one eligible machine per operation.

The decoder is the usual earliest-start schedule builder.  Neither method uses
the rule-candidate/simulated-annealing protocol of the main Experiment 1, so the
public metaheuristic experiment remains independent of the hybrid decoder.

The module is intentionally framework-free.  It only needs the parsed
``BenchInstance`` objects from ``common/public_benchmark.py``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


WEIGHTS = (0.4, 0.3, 0.3)


@dataclass
class Candidate:
    order: List[int]
    assignment: List[List[int]]
    cmax: float = 0.0
    lb: float = 0.0
    tardy: float = 0.0
    score: float = 0.0

    def clone(self) -> "Candidate":
        return Candidate(list(self.order), [list(row) for row in self.assignment],
                         self.cmax, self.lb, self.tardy, self.score)


def _eligible(instance, job: int, op: int) -> Dict[int, float]:
    return instance.jobs[job][op]


def decode_metrics(instance, order: Sequence[int], assignment: Sequence[Sequence[int]]):
    """Build the earliest-start schedule and return (Cmax, LB, tardiness)."""
    job_ready = [0.0] * instance.n_jobs
    machine_free = [0.0] * instance.n_machines
    machine_load = [0.0] * instance.n_machines
    next_op = [0] * instance.n_jobs
    job_completion = [0.0] * instance.n_jobs
    for job in order:
        op = next_op[job]
        times = _eligible(instance, job, op)
        machine = int(assignment[job][op])
        if machine not in times:
            raise ValueError("ineligible machine %d for job %d op %d" % (machine, job, op))
        start = max(job_ready[job], machine_free[machine])
        finish = start + float(times[machine])
        job_ready[job] = finish
        job_completion[job] = finish
        machine_free[machine] = finish
        machine_load[machine] += float(times[machine])
        next_op[job] += 1
    if not job_completion:
        return 0.0, 0.0, 0.0
    cmax = float(max(job_completion))
    lb = float(np.std(machine_load))
    tardy = float(sum(max(0.0, job_completion[job] - float(instance.due_dates[job]))
                      for job in range(instance.n_jobs)))
    return cmax, lb, tardy


def decode(instance, order: Sequence[int], assignment: Sequence[Sequence[int]]) -> float:
    """Build the earliest-start schedule and return its makespan."""
    return decode_metrics(instance, order, assignment)[0]


def evaluate_candidate(instance, candidate: Candidate, objective: str = "makespan"):
    candidate.cmax, candidate.lb, candidate.tardy = decode_metrics(
        instance, candidate.order, candidate.assignment)
    candidate.score = (candidate.cmax if objective == "makespan"
                       else (WEIGHTS[0] * candidate.cmax
                             + WEIGHTS[1] * candidate.lb
                             + WEIGHTS[2] * candidate.tardy))
    return candidate


def random_solution(instance, rng: random.Random) -> Candidate:
    order: List[int] = []
    for job, operations in enumerate(instance.jobs):
        order.extend([job] * len(operations))
    rng.shuffle(order)
    assignment = []
    for operations in instance.jobs:
        row = []
        for times in operations:
            row.append(rng.choice(sorted(times)))
        assignment.append(row)
    return evaluate_candidate(instance, Candidate(order, assignment))


def greedy_assignment(instance) -> List[List[int]]:
    return [[min(times, key=lambda m: (times[m], m)) for times in operations]
            for operations in instance.jobs]


def _required_counts(instance) -> List[int]:
    return [len(operations) for operations in instance.jobs]


def _ox_crossover(parent_a: Sequence[int], parent_b: Sequence[int],
                  counts: Sequence[int], rng: random.Random) -> List[int]:
    n = len(parent_a)
    left, right = sorted(rng.sample(range(n + 1), 2))
    if left == right:
        right = min(n, left + 1)
    child: List[Optional[int]] = [None] * n
    child[left:right] = parent_a[left:right]
    used = [0] * len(counts)
    for job in child:
        if job is not None:
            used[job] += 1
    fill_positions = list(range(right, n)) + list(range(0, left))
    cursor = 0
    for position in fill_positions:
        while cursor < n:
            job = parent_b[(right + cursor) % n]
            cursor += 1
            if used[job] < counts[job]:
                child[position] = job
                used[job] += 1
                break
    if any(job is None for job in child):
        raise RuntimeError("OX crossover produced an incomplete operation order")
    return [int(job) for job in child]


def _uniform_assignment(parent_a: Sequence[Sequence[int]],
                        parent_b: Sequence[Sequence[int]],
                        rng: random.Random) -> List[List[int]]:
    child = []
    for row_a, row_b in zip(parent_a, parent_b):
        child.append([a if rng.random() < 0.5 else b for a, b in zip(row_a, row_b)])
    return child


def _mutate(candidate: Candidate, instance, rng: random.Random,
            swap_rate: float = 0.30, machine_rate: float = 0.20) -> None:
    if len(candidate.order) > 1 and rng.random() < swap_rate:
        i, j = rng.sample(range(len(candidate.order)), 2)
        candidate.order[i], candidate.order[j] = candidate.order[j], candidate.order[i]
    if rng.random() < machine_rate:
        job = rng.randrange(instance.n_jobs)
        op = rng.randrange(len(instance.jobs[job]))
        choices = sorted(instance.jobs[job][op])
        if len(choices) > 1:
            current = candidate.assignment[job][op]
            alternatives = [m for m in choices if m != current]
            candidate.assignment[job][op] = rng.choice(alternatives)


def _tournament(population: Sequence[Candidate], rng: random.Random, size: int = 3) -> Candidate:
    contenders = [rng.choice(population) for _ in range(min(size, len(population)))]
    return min(contenders, key=lambda item: item.score)


def run_ga(instance, evaluations: int = 6000, population_size: int = 40,
           seed: int = 42, objective: str = "makespan") -> Candidate:
    """Run a generational GA under a fixed schedule-evaluation budget."""
    rng = random.Random(seed)
    counts = _required_counts(instance)
    population = [evaluate_candidate(instance, random_solution(instance, rng), objective)
                  for _ in range(population_size)]
    spent = population_size
    while spent < evaluations:
        population.sort(key=lambda item: item.score)
        next_generation = [population[0].clone(), population[1].clone()]
        while len(next_generation) < population_size and spent < evaluations:
            parent_a = _tournament(population, rng)
            parent_b = _tournament(population, rng)
            order = _ox_crossover(parent_a.order, parent_b.order, counts, rng)
            assignment = _uniform_assignment(parent_a.assignment, parent_b.assignment, rng)
            child = evaluate_candidate(instance, Candidate(order, assignment), objective)
            spent += 1
            _mutate(child, instance, rng)
            evaluate_candidate(instance, child, objective)
            spent += 1
            next_generation.append(child)
        population = next_generation
    population.sort(key=lambda item: item.score)
    return population[0]


def _neighbor(candidate: Candidate, instance, rng: random.Random,
              objective: str = "makespan") -> Tuple[Candidate, tuple]:
    child = candidate.clone()
    if rng.random() < 0.6 and len(child.order) > 1:
        i, j = rng.sample(range(len(child.order)), 2)
        child.order[i], child.order[j] = child.order[j], child.order[i]
        move = ("swap", i, j)
    else:
        job = rng.randrange(instance.n_jobs)
        op = rng.randrange(len(instance.jobs[job]))
        choices = sorted(instance.jobs[job][op])
        if len(choices) > 1:
            current = child.assignment[job][op]
            alternatives = [m for m in choices if m != current]
            child.assignment[job][op] = rng.choice(alternatives)
            move = ("machine", job, op)
        else:
            evaluate_candidate(instance, child, objective)
            return child, ("noop", job, op)
    evaluate_candidate(instance, child, objective)
    return child, move


def run_ts(instance, evaluations: int = 6000, neighbourhood: int = 50,
           tenure: int = 7, seed: int = 42, objective: str = "makespan") -> Candidate:
    """Run a tabu search with swap and machine-reassignment neighbourhoods."""
    rng = random.Random(seed)
    current = evaluate_candidate(instance, random_solution(instance, rng), objective)
    best = current.clone()
    tabu: List[tuple] = []
    spent = 1
    while spent < evaluations:
        neighbours = []
        for _ in range(min(neighbourhood, evaluations - spent)):
            neighbour, move = _neighbor(current, instance, rng, objective)
            spent += 1
            neighbours.append((neighbour, move))
        if not neighbours:
            break
        candidates = []
        for neighbour, move in neighbours:
            is_tabu = move in tabu
            if is_tabu and neighbour.score >= best.score - 1e-9:
                continue
            candidates.append((neighbour, move))
        if not candidates:
            candidates = neighbours
        neighbour, move = min(candidates, key=lambda item: item[0].score)
        current = neighbour
        tabu.append(move)
        if len(tabu) > tenure:
            tabu.pop(0)
        if current.score < best.score:
            best = current.clone()
    return best


def evaluate_method(instance, method: str, evaluations: int, seed: int,
                    objective: str = "makespan") -> float:
    if method == "GA":
        return run_ga(instance, evaluations=evaluations, seed=seed,
                      objective=objective).score
    if method == "TS":
        return run_ts(instance, evaluations=evaluations, seed=seed,
                      objective=objective).score
    raise ValueError("unknown metaheuristic %s" % method)


def evaluate_instances(instances: Sequence, evaluations: int = 6000,
                       seed: int = 42,
                       methods: Sequence[str] = ("GA", "TS"),
                       objective: str = "makespan") -> List[dict]:
    rows = []
    for instance in instances:
        row = {
            "name": instance.name,
            "family": instance.family,
            "jobs": instance.n_jobs,
            "machines": instance.n_machines,
            "operations": instance.total_ops,
            "bks": instance.bks,
        }
        for method in methods:
            row[method] = evaluate_method(instance, method, evaluations, seed, objective)
            if instance.bks:
                row["gap_%s" % method] = (row[method] - instance.bks) / instance.bks * 100.0
            else:
                row["gap_%s" % method] = float("nan")
        rows.append(row)
    return rows
