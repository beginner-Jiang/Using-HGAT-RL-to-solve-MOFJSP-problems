# -*- coding: utf-8 -*-
"""Standard machine assignment shared by the dispatching-rule baselines.

The priority of a dispatching rule decides which ready operation is scheduled next; the machine
that processes it is a separate decision.  The rule files used to take the machine with the
smallest processing time, ignoring when that machine is free.  That choice is not only weaker
than the standard earliest-completion assignment, it is also static: the same operation always
maps to the same machine, so five of the six baselines produced identical machine loads and the
load-balance column of the comparison had only two distinct values.

``select_earliest_completion`` assigns the eligible machine that finishes the operation earliest,
breaking ties by processing time and then by machine index.
"""


def select_earliest_completion(operation, machines, job_ready_time):
    """Eligible machine with the smallest completion time, or None if none is eligible."""
    best = None
    for machine in machines:
        processing_time = operation.machine_times.get(machine.machine_id)
        if processing_time is None:
            continue
        start = max(job_ready_time, machine.available_time)
        end = start + float(processing_time)
        key = (end, float(processing_time), machine.machine_id)
        if best is None or key < best[0]:
            best = (key, machine)
    return best[1] if best is not None else None
