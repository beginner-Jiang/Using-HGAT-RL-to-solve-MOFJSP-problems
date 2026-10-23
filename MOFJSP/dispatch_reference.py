"""Deterministic reference schedule for the validation objective of every trainer.

Each trainer selects the weights it reports on a fixed validation set and scores a
checkpoint as the weighted objective of that checkpoint divided by a reference of the same
instance.  While that reference is taken from the first check of the run itself, the first
entry of the curve is exactly 1.0 by construction - it carries no information about quality
- and the checkpoint that minimises the curve is then the one that is closest to a single
arbitrary early snapshot rather than the best one of the run.  The measured curves of this
project show what that produces: the best check of the DAN baseline sits at its first
check, i.e. before the curriculum has introduced the medium instances, while HGNN and the
proposed method select their peak at whatever stage of the curriculum they happen to be in,
and the internal curve does not move in the same direction as the comparison metric.

The reference is therefore built from the instance alone, by one deterministic dispatch
rule, so the objective of two checkpoints - and of two methods on the same instance - is on
one scale.  The rule schedules, at every step, the ready operation whose earliest completion
on an eligible machine is the smallest, breaking ties by job index and then by machine
index.  That is the classic earliest-completion dispatching of the flexible job shop: it is
deterministic, it needs nothing but the instance file the trainers already read, and it is
the same reference for every method.

The instance format is ``num_jobs num_machines``, then one line per job holding ``num_ops``
followed by ``k machine proc_time`` triples with a one based machine number, then one line
of due dates.  Lines that start with ``#`` are comments.
"""

import os

import numpy as np


def read_instance(path):
    """Parse one instance file into (num_jobs, num_machines, ops, due_dates).

    ``ops`` is a list per job of that job's operations, and every operation is a list of
    ``(machine, processing_time)`` alternatives with a zero based machine number.
    """
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if len(lines) < 2:
        raise ValueError("instance %s holds no job description" % path)
    num_jobs, num_machines = (int(value) for value in lines[0].split())
    ops = []
    row = 1
    for _ in range(num_jobs):
        values = [int(value) for value in lines[row].split()]
        row += 1
        num_ops = values[0]
        cursor = 1
        job_ops = []
        for _ in range(num_ops):
            k = values[cursor]
            cursor += 1
            alternatives = []
            for _ in range(k):
                machine = values[cursor] - 1
                proc = float(values[cursor + 1])
                cursor += 2
                alternatives.append((machine, proc))
            job_ops.append(alternatives)
        ops.append(job_ops)
    due_dates = [float(value) for value in lines[row].split()]
    return num_jobs, num_machines, ops, due_dates


def dispatch_objective(num_jobs, num_machines, ops, due_dates):
    """[Cmax, load balance, total tardiness] of the earliest-completion dispatch."""
    machine_avail = np.zeros(num_machines, dtype=np.float64)
    machine_load = np.zeros(num_machines, dtype=np.float64)
    job_avail = np.zeros(num_jobs, dtype=np.float64)
    next_op = [0] * num_jobs
    remaining = sum(len(job) for job in ops)
    while remaining > 0:
        best_key = None
        best = None
        for job in range(num_jobs):
            step = next_op[job]
            if step >= len(ops[job]):
                continue
            for machine, proc in ops[job][step]:
                end = max(job_avail[job], machine_avail[machine]) + proc
                key = (end, job, machine)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (job, machine, proc, end)
        job, machine, proc, end = best
        job_avail[job] = end
        machine_avail[machine] = end
        machine_load[machine] += proc
        next_op[job] += 1
        remaining -= 1
    cmax = float(machine_avail.max()) if num_machines else 0.0
    balance = float(np.std(machine_load)) if num_machines else 0.0
    tardiness = float(sum(max(0.0, job_avail[job] - due_dates[job])
                          for job in range(num_jobs)))
    return [cmax, balance, tardiness]


def reference_objectives(file_names, instance_dir):
    """Reference [Cmax, LB, Tardy] of every named instance, in the order given."""
    objectives = []
    for name in file_names:
        path = name if os.path.isabs(name) else os.path.join(instance_dir, name)
        objectives.append(dispatch_objective(*read_instance(path)))
    return objectives
