# -*- coding: utf-8 -*-
"""Inference-time schedule refinement for Experiment 1.

The trained model is not modified.  A dispatching rule or a policy rollout provides a
feasible schedule, which is represented as an operation priority order plus a machine
assignment.  A fast list-scheduling decoder evaluates the schedule, and a portfolio of
simulated annealing and ruin-and-recreate moves improves it.  The refinement is an
explicit part of the proposed hybrid pipeline and its running time is reported.
"""
import heapq
import importlib
import random

import numpy as np

WEIGHT = (0.4, 0.3, 0.3)


def value(objective):
    return WEIGHT[0] * objective[0] + WEIGHT[1] * objective[1] + WEIGHT[2] * objective[2]


def run_rule_with_order(scheduler_class, read_func, path):
    """Run a dispatching rule and return its scheduling order and machine assignment."""
    jobs, machines = read_func(path)
    module = importlib.import_module(scheduler_class.__module__)
    machine_class = module.Machine
    original_assign = machine_class.assign_operation
    order = []

    def patched(self, operation, start_time):
        result = original_assign(self, operation, start_time)
        order.append((operation.job_id, operation.op_id, self.machine_id))
        return result

    machine_class.assign_operation = patched
    try:
        scheduler = scheduler_class(jobs, machines)
        scheduler.run_schedule()
    finally:
        machine_class.assign_operation = original_assign

    makespan = max(job.get_completion_time() for job in jobs)
    loads = [machine.get_total_load() for machine in machines]
    tardiness = sum(max(0.0, job.get_completion_time() - job.due_date) for job in jobs)
    return order, (float(makespan), float(np.std(loads)), float(tardiness))


class Decoder:
    """List-scheduling decoder with a fixed operation priority order."""

    def __init__(self, instance):
        self.instance = instance
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.ops_per_job = list(instance.ops_per_job)
        self.total_ops = instance.total_ops
        self.proc = instance.proc_time_matrix.detach().cpu().numpy()
        self.job_op_to_idx = instance.job_op_to_idx.detach().cpu().numpy()
        self.job_of_op = instance.op_idx_to_job.detach().cpu().numpy()
        self.due = np.asarray(instance.due_dates, dtype=float)
        self.eligible = [np.nonzero(self.proc[op] < 99998.0)[0]
                         for op in range(self.total_ops)]

    def decode(self, order, forced_machine=None):
        rank = np.empty(self.total_ops, dtype=np.int64)
        for position, op in enumerate(order):
            rank[op] = position
        next_op = [0] * self.n_jobs
        job_ready = np.zeros(self.n_jobs, dtype=float)
        machine_ready = np.zeros(self.n_machines, dtype=float)
        machine_load = np.zeros(self.n_machines, dtype=float)
        job_completion = np.zeros(self.n_jobs, dtype=float)

        heap = []
        for job in range(self.n_jobs):
            if self.ops_per_job[job] > 0:
                op = int(self.job_op_to_idx[job, 0])
                heapq.heappush(heap, (int(rank[op]), op))

        while heap:
            _rank, op = heapq.heappop(heap)
            job = int(self.job_of_op[op])
            if forced_machine is not None and op in forced_machine:
                machine = int(forced_machine[op])
            else:
                machine = -1
                best_end = float("inf")
                for candidate in self.eligible[op]:
                    candidate = int(candidate)
                    start = max(job_ready[job], machine_ready[candidate])
                    end = start + self.proc[op, candidate]
                    if end < best_end:
                        best_end = end
                        machine = candidate
            start = max(job_ready[job], machine_ready[machine])
            end = start + self.proc[op, machine]
            machine_ready[machine] = end
            machine_load[machine] += self.proc[op, machine]
            job_ready[job] = end
            job_completion[job] = end

            next_index = next_op[job] + 1
            next_op[job] = next_index
            if next_index < self.ops_per_job[job]:
                next_op_idx = int(self.job_op_to_idx[job, next_index])
                heapq.heappush(heap, (int(rank[next_op_idx]), next_op_idx))

        cmax = float(job_completion.max())
        load_balance = float(np.std(machine_load))
        tardiness = float(np.sum(np.maximum(0.0, job_completion - self.due)))
        return cmax, load_balance, tardiness


def anneal(decoder, order, forced, budget=4000, seed=7):
    """Simulated annealing over machine reassignment, swap and insertion moves."""
    rng = random.Random(seed)
    current_order = list(order)
    current_forced = dict(forced)
    current = decoder.decode(current_order, current_forced)
    best = current
    best_order = list(current_order)
    best_forced = dict(current_forced)
    start_value = max(1e-9, value(current))
    temperature = 0.01 * start_value
    cooling = (0.0001 / 0.01) ** (1.0 / max(1, budget))

    for _ in range(budget):
        move = rng.random()
        if move < 0.45:
            op = rng.randrange(decoder.total_ops)
            choices = [int(m) for m in decoder.eligible[op]
                       if int(m) != current_forced.get(op, -1)]
            if not choices:
                continue
            previous = current_forced.get(op, None)
            current_forced[op] = rng.choice(choices)
            candidate = decoder.decode(current_order, current_forced)
            delta = value(candidate) - value(current)
            accept = delta < 0 or rng.random() < np.exp(-delta / max(1e-9, temperature))
            if accept:
                current = candidate
                if value(current) < value(best):
                    best = current
                    best_order = list(current_order)
                    best_forced = dict(current_forced)
            else:
                if previous is None:
                    current_forced.pop(op, None)
                else:
                    current_forced[op] = previous
        elif move < 0.75:
            if len(current_order) < 2:
                continue
            i, j = rng.sample(range(len(current_order)), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            candidate = decoder.decode(current_order, current_forced)
            delta = value(candidate) - value(current)
            accept = delta < 0 or rng.random() < np.exp(-delta / max(1e-9, temperature))
            if accept:
                current = candidate
                if value(current) < value(best):
                    best = current
                    best_order = list(current_order)
                    best_forced = dict(current_forced)
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]
        else:
            if len(current_order) < 2:
                continue
            i, j = rng.sample(range(len(current_order)), 2)
            item = current_order.pop(i)
            current_order.insert(j, item)
            candidate = decoder.decode(current_order, current_forced)
            delta = value(candidate) - value(current)
            accept = delta < 0 or rng.random() < np.exp(-delta / max(1e-9, temperature))
            if accept:
                current = candidate
                if value(current) < value(best):
                    best = current
                    best_order = list(current_order)
                    best_forced = dict(current_forced)
            else:
                current_order.pop(j)
                current_order.insert(i, item)
        temperature *= cooling

    return best, best_order, best_forced


def ruin_recreate(decoder, order, forced, remove_count=8, iterations=10, stride=1, seed=13):
    """Large-neighbourhood search: remove operations and reinsert them greedily."""
    rng = random.Random(seed)
    best = decoder.decode(order, forced)
    best_order = list(order)
    best_forced = dict(forced)
    current_order = list(order)
    current_forced = dict(forced)
    current = best

    for _ in range(iterations):
        removed = rng.sample(current_order, min(remove_count, len(current_order)))
        removed_set = set(removed)
        new_order = [op for op in current_order if op not in removed_set]
        new_forced = {op: machine for op, machine in current_forced.items()
                      if op not in removed_set}
        for op in removed:
            positions = list(range(0, len(new_order) + 1, max(1, stride)))
            if positions[-1] != len(new_order):
                positions.append(len(new_order))
            best_position = 0
            best_machine = None
            best_value = float("inf")
            for position in positions:
                candidate_order = new_order[:position] + [op] + new_order[position:]
                candidate = decoder.decode(candidate_order, new_forced)
                candidate_value = value(candidate)
                if candidate_value < best_value:
                    best_value = candidate_value
                    best_position = position
                    best_machine = None
                original_machine = current_forced.get(op)
                if original_machine is not None:
                    forced_candidate = dict(new_forced)
                    forced_candidate[op] = original_machine
                    forced_value = value(decoder.decode(candidate_order, forced_candidate))
                    if forced_value < best_value:
                        best_value = forced_value
                        best_position = position
                        best_machine = original_machine
            new_order.insert(best_position, op)
            if best_machine is not None:
                new_forced[op] = best_machine
        candidate = decoder.decode(new_order, new_forced)
        if value(candidate) < value(best):
            best = candidate
            best_order = list(new_order)
            best_forced = dict(new_forced)
        if value(candidate) < value(current):
            current = candidate
            current_order = list(new_order)
            current_forced = dict(new_forced)
    return best, best_order, best_forced


def refine(decoder, order, forced, budget, lns_iterations, stride, seed=7):
    """Anneal first, then improve the best basin with ruin-and-recreate."""
    best, best_order, best_forced = anneal(decoder, order, forced, budget=budget, seed=seed)
    candidate, candidate_order, candidate_forced = ruin_recreate(
        decoder, best_order, best_forced, remove_count=8,
        iterations=lns_iterations, stride=stride, seed=seed + 1)
    if value(candidate) < value(best):
        return candidate, candidate_order, candidate_forced
    return best, best_order, best_forced
