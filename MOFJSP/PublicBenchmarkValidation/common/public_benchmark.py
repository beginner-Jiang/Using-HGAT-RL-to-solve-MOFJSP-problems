# -*- coding: utf-8 -*-
"""Core utilities for makespan-only validation on public FJSP instances.

The module deliberately keeps the public benchmark experiment separate from the
three-objective training pipeline.  It parses the standard instance format,
constructs the graph state expected by the adopted HGAT-PPO checkpoint, decodes
the five dispatching-rule baselines, and provides the statistical tests used by
the report.

Public instance format
----------------------
The first non-comment line is ``<jobs> <machines>``.  Each following job line
starts with the number of operations.  An operation is encoded as
``<eligible machines> (machine processing_time)*`` with zero-based machine
indices, exactly as distributed by the fjsp-instances repository.

The experiment uses a single seed.  The statistical unit is an instance, not a
training seed: every method is evaluated on the same instances and the paired
tests therefore use the per-instance makespan values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
    from scipy.stats import f as f_dist
    HAS_SCIPY = True
except Exception:  # pragma: no cover - the report still writes descriptive rows
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Paths and the public benchmark manifest
# ---------------------------------------------------------------------------

BRANDIMARTE_BKS = {
    "mk01": 40,
    "mk02": 26,
    "mk03": 204,
    "mk04": 60,
    "mk05": 172,
    "mk06": 57,
    "mk07": 139,
    "mk08": 523,
    "mk09": 307,
    "mk10": 197,
}

# The e-data files distributed with the fjsp-instances repository.  The values
# below are the proven/current best values recorded in its instances.json.  The
# six e-la rows without a published value are deliberately left empty instead
# of inventing a lower bound; their makespans still enter the paired tests.
LAWRENCE_BKS = {
    "la01": 609, "la02": 655, "la03": 550, "la04": 568, "la05": 503,
    "la06": 833, "la07": 762, "la08": 845, "la09": 878, "la10": 866,
    "la11": 1103, "la12": 960, "la13": 1053, "la14": 1123, "la15": 1111,
    "la16": 892, "la17": 707, "la18": 842, "la19": 796, "la20": 857,
    "la21": None, "la22": 880, "la23": 950, "la24": 908, "la25": 936,
    "la26": None, "la27": None, "la28": None, "la29": None, "la30": None,
    "la31": None, "la32": 1698, "la33": 1547, "la34": 1599, "la35": 1736,
    "la36": 1160, "la37": 1397, "la38": 1141, "la39": 1184, "la40": 1144,
}

BRANDIMARTE_NAMES = ["mk%02d" % i for i in range(1, 11)]
LAWRENCE_NAMES = ["la%02d" % i for i in range(1, 41)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class BenchInstance:
    """One public FJSP instance in a lightweight, framework-free form."""

    name: str
    family: str
    jobs: List[List[Dict[int, float]]]
    n_machines: int
    bks: Optional[float]
    source: str
    split: str
    due_dates: List[float]
    source_sha256: str = ""

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def total_ops(self) -> int:
        return sum(len(job) for job in self.jobs)

    @property
    def size_label(self) -> str:
        if self.n_jobs <= 10:
            return "small"
        if self.n_jobs <= 20:
            return "medium"
        return "large"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "jobs": self.n_jobs,
            "machines": self.n_machines,
            "operations": self.total_ops,
            "bks": self.bks,
            "source": self.source,
            "split": self.split,
            "size": self.size_label,
            "due_dates": self.due_dates,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "BenchInstance":
        source = Path(row["source"])
        instance = parse_public_instance(
            source, row["family"], row["name"], row["split"], row.get("bks")
        )
        if row.get("due_dates"):
            instance.due_dates = list(row["due_dates"])
        return instance


def parse_public_instance(path: Path, family: str, name: str, split: str,
                          bks: Optional[float]) -> BenchInstance:
    """Parse a zero-based standard FJSP file and attach synthetic due dates.

    The public files do not contain due dates.  A makespan-only experiment does
    not use a due-date objective, but the EDD baseline needs a deterministic
    ordering.  We therefore use 1.5 times the job's minimum remaining work as a
    fixed, instance-derived due date.  The policy is trained with w3=0, so this
    convention cannot change its makespan reward.
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("instance %s has too few non-comment lines" % path)
    n_jobs, n_machines = map(int, lines[0].split())
    jobs: List[List[Dict[int, float]]] = []
    for job_idx in range(n_jobs):
        nums = [int(value) for value in lines[1 + job_idx].split()]
        cursor = 0
        n_ops = nums[cursor]
        cursor += 1
        operations: List[Dict[int, float]] = []
        for _ in range(n_ops):
            n_eligible = nums[cursor]
            cursor += 1
            machine_times: Dict[int, float] = {}
            for _ in range(n_eligible):
                machine = nums[cursor]
                processing = float(nums[cursor + 1])
                cursor += 2
                if not 0 <= machine < n_machines:
                    raise ValueError("machine %d outside 0..%d in %s"
                                     % (machine, n_machines - 1, path))
                machine_times[machine] = processing
            if not machine_times:
                raise ValueError("operation without an eligible machine in %s" % path)
            operations.append(machine_times)
        jobs.append(operations)

    work_per_job = [sum(min(op.values()) for op in job) for job in jobs]
    due_dates = [1.5 * work for work in work_per_job]
    return BenchInstance(
        name=name,
        family=family,
        jobs=jobs,
        n_machines=n_machines,
        bks=bks,
        source=str(path),
        split=split,
        due_dates=due_dates,
        source_sha256=sha256_file(path),
    )


def split_for(family: str, name: str) -> str:
    """Return the label of the single combined public split.

    Adaptation uses Lawrence la01--la20 for training and la21--la25 for
    validation.  The held-out test set contains all ten Brandimarte instances
    (MK01--MK10) and Lawrence la26--la40, 25 instances in total.  The two
    families are therefore evaluated together in one protocol rather than
    trained and evaluated independently.
    """
    if family == "Brandimarte":
        return "test"
    index = int(name[2:])
    if index <= 20:
        return "train"
    if index <= 25:
        return "validation"
    return "test"


def prepare_public_data(source_root: Path, public_root: Path, code_data_root: Path,
                        verbose: bool = True) -> List[BenchInstance]:
    """Copy only the two benchmark families into the delivery tree and write a manifest."""
    source_root = source_root.resolve()
    public_root.mkdir(parents=True, exist_ok=True)
    code_data_root.mkdir(parents=True, exist_ok=True)

    sources = []
    for name in BRANDIMARTE_NAMES:
        sources.append(("Brandimarte", name, source_root / "brandimarte" / (name + ".txt")))
    for name in LAWRENCE_NAMES:
        sources.append(("Lawrence", name, source_root / "hurink" / "edata" / (name + ".txt")))

    rows: List[BenchInstance] = []
    for family, name, source in sources:
        if not source.exists():
            raise FileNotFoundError("missing public instance: %s" % source)
        destination_family = public_root / family
        destination_family.mkdir(parents=True, exist_ok=True)
        destination = destination_family / source.name
        if not destination.exists() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)
        code_copy = code_data_root / "raw" / family / source.name
        code_copy.parent.mkdir(parents=True, exist_ok=True)
        if not code_copy.exists() or sha256_file(code_copy) != sha256_file(source):
            shutil.copy2(source, code_copy)

        bks = BRANDIMARTE_BKS.get(name) if family == "Brandimarte" else LAWRENCE_BKS.get(name)
        instance = parse_public_instance(destination, family, name, split_for(family, name), bks)
        rows.append(instance)
        if verbose:
            print("prepared %-6s %-12s %2dx%-2d ops=%3d BKS=%s"
                  % (family, name, instance.n_jobs, instance.n_machines,
                     instance.total_ops, bks), flush=True)

    manifest = {
        "description": "Makespan-only public FJSP validation manifest",
        "source_root": str(source_root),
        "count": len(rows),
        "families": {
            family: {
                split: [row.name for row in rows
                        if row.family == family and row.split == split]
                for split in ("train", "validation", "test")
            }
            for family in ("Brandimarte", "Lawrence")
        },
        "instances": [row.to_dict() for row in rows],
    }
    manifest_path = code_data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (code_data_root / "splits.json").write_text(
        json.dumps(manifest["families"], indent=2, ensure_ascii=False), encoding="utf-8")
    readme = (
        "# Public Data Set\n\n"
        "This folder contains the two public FJSP families used by the "
        "makespan-only external validation.\n\n"
        "- `Brandimarte/`: MK01-MK10, Brandimarte (1993), standard FJSP format.\n"
        "- `Lawrence/`: la01-la40, Lawrence/Hurink e-data as distributed by "
        "the `fjsp-instances` repository.\n\n"
        "The machine indices in these files are zero-based.  Brandimarte "
        "best-known values are the standard values; Lawrence values are the "
        "proven/best values recorded in the repository's `instances.json`. "
        "Missing published values are left empty rather than guessed.\n\n"
        "Source copy used for this run: `%s`\n" % source_root
    )
    (public_root / "README.md").write_text(readme, encoding="utf-8")
    if verbose:
        print("manifest written to %s" % manifest_path, flush=True)
    return rows


def load_manifest(code_data_root: Path) -> Tuple[List[BenchInstance], dict]:
    manifest_path = code_data_root / "manifest.json"
    blob = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [BenchInstance.from_dict(row) for row in blob["instances"]], blob


def write_internal_instance(instance: BenchInstance, path: Path) -> None:
    """Write a temporary one-based copy in the format used by the project trainers.

    The public files are not modified.  The copy only adds the two metadata
    lines that the original MOFJSP reader expects (fixed-preference due dates and
    neutral machine capabilities).
    """
    lines = ["%d %d" % (instance.n_jobs, instance.n_machines)]
    for job in instance.jobs:
        values = [str(len(job))]
        for operation in job:
            values.append(str(len(operation)))
            for machine, processing in sorted(operation.items()):
                rendered = str(int(processing)) if float(processing).is_integer() else str(processing)
                values.extend([str(machine + 1), rendered])
        lines.append(" ".join(values))
    lines.append(" ".join("%.6g" % value for value in instance.due_dates))
    lines.append(" ".join(["1"] * instance.n_machines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_internal_files(instances: Sequence[BenchInstance], root: Path) -> Dict[str, Path]:
    """Write one-based copies for the three pretrained baseline readers."""
    paths = {}
    root.mkdir(parents=True, exist_ok=True)
    for instance in instances:
        path = root / ("public_%s_%s.txt" % (instance.name, instance.size_label))
        # Parallel evaluators may materialise the same copy; write once through a
        # temporary file so that a concurrent reader never sees a partial file.
        if not (path.exists() and path.stat().st_size > 0):
            temporary = path.with_name(path.name + ".tmp")
            write_internal_instance(instance, temporary)
            os.replace(str(temporary), str(path))
        paths[instance.name] = path
    return paths


# ---------------------------------------------------------------------------
# Dispatching rules
# ---------------------------------------------------------------------------

RULES = ("SPT", "LBD", "FIFO", "MOPNR", "MWKR")
LEARNING_BASELINES = ("MLP+DQN", "MLP+PPO", "GAT+PPO", "HGNN", "DAN", "RESCHED")


def dispatch_rule(instance: BenchInstance, rule: str) -> float:
    """Return the makespan of one of the five deterministic dispatching rules.

    Operation priority follows the definitions used in the comparison study.
    Machine assignment is earliest completion for four rules; LBD assigns the
    eligible machine with the smallest current load, breaking ties by earliest
    completion.  All rules see the same release/finish times.
    """
    if rule not in RULES:
        raise ValueError("unknown rule %s" % rule)
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines
    next_op = [0] * n_jobs
    job_ready = [0.0] * n_jobs
    job_completion = [0.0] * n_jobs
    machine_free = [0.0] * n_machines
    machine_load = [0.0] * n_machines

    for _ in range(instance.total_ops):
        candidates = []
        for job in range(n_jobs):
            op_index = next_op[job]
            if op_index >= len(instance.jobs[job]):
                continue
            machine_times = instance.jobs[job][op_index]
            if rule == "LBD":
                machine = min(
                    machine_times,
                    key=lambda m: (machine_load[m],
                                   max(job_ready[job], machine_free[m]) + machine_times[m], m),
                )
            else:
                machine = min(
                    machine_times,
                    key=lambda m: (max(job_ready[job], machine_free[m]) + machine_times[m],
                                   machine_times[m], m),
                )
            processing = machine_times[machine]
            start = max(job_ready[job], machine_free[machine])
            end = start + processing
            remaining_ops = len(instance.jobs[job]) - op_index
            remaining_work = sum(min(op.values()) for op in instance.jobs[job][op_index:])
            due = instance.due_dates[job]

            if rule == "SPT":
                key = (processing, end, job, op_index)
            elif rule == "LBD":
                key = (machine_load[machine], end, job, op_index)
            elif rule == "FIFO":
                key = (job_ready[job], job, op_index)
            elif rule == "MOPNR":
                key = (-remaining_ops, due, job, op_index)
            else:  # MWKR
                key = (-remaining_work, due, job, op_index)
            candidates.append((key, job, op_index, machine, start, end, processing))

        _, job, op_index, machine, start, end, processing = min(candidates, key=lambda row: row[0])
        next_op[job] += 1
        job_ready[job] = end
        job_completion[job] = end
        machine_free[machine] = end
        machine_load[machine] += processing

    return float(max(job_completion))


def evaluate_rules(instances: Sequence[BenchInstance]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for instance in instances:
        result[instance.name] = {rule: dispatch_rule(instance, rule) for rule in RULES}
    return result


# ---------------------------------------------------------------------------
# Adopted model adapter
# ---------------------------------------------------------------------------

def _import_param(project_root: Path):
    param_dir = project_root / "PPO_HGAT" / "Param_Experiment"
    if str(param_dir) not in sys.path:
        sys.path.insert(0, str(param_dir))
    import Param  # type: ignore
    return Param


def _allow_foreign_numpy_checkpoint() -> None:
    """Allow a checkpoint written with numpy 2 to be read by numpy 1.x."""
    import numpy as np_local

    if hasattr(np_local, "_core"):
        return
    for name in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        module = getattr(np_local.core, name, None)
        if module is not None:
            sys.modules.setdefault("numpy._core." + name, module)
    sys.modules.setdefault("numpy._core", np_local.core)


def load_adopted_agent(project_root: Path, checkpoint_path: Path):
    Param = _import_param(project_root)
    _allow_foreign_numpy_checkpoint()
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError("adopted checkpoint not found: %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=Param.DEVICE, weights_only=False)
    if "gat" not in checkpoint:
        raise KeyError("checkpoint does not contain the HGAT state: %s" % checkpoint_path)

    n_heads = int(checkpoint["gat"].get("l1.att_seq").shape[0])
    gat = Param.HeteroGAT(3, 3, 256, 256, num_heads=n_heads).to(Param.DEVICE)
    upper = Param.UpperPolicy(256, 512).to(Param.DEVICE)
    lower = Param.LowerPolicy(256, 512).to(Param.DEVICE)
    critic_u = Param.Critic(256, 512).to(Param.DEVICE)
    critic_l = Param.Critic(256, 512).to(Param.DEVICE)
    state_norm = Param.TorchRunningMeanStd(shape=(256,), device=Param.DEVICE)
    if checkpoint.get("state_norm_mean") is not None:
        state_norm.mean = checkpoint["state_norm_mean"].to(Param.DEVICE).float()
        state_norm.var = checkpoint["state_norm_var"].to(Param.DEVICE).float()
        state_norm.count = float(checkpoint.get("state_norm_count", state_norm.count))

    cfg = SimpleNamespace(
        w1=1.0,
        w2=0.0,
        w3=0.0,
        gamma=1.0,
        beta=0.2,
        K=n_heads,
        reward_scaling=5.0,
        reward_clip=2.0,
    )
    agent = Param.HierarchicalPPOAgent(
        None, gat, upper, lower, critic_u, critic_l,
        gamma=cfg.gamma, beta=cfg.beta, lr=2e-4, lam=0.95,
        state_norm=state_norm,
    )
    # The lower policy's fixed-preference term is a module global in Param.py.
    Param.W1_DEFAULT = 1.0
    Param.W2_DEFAULT = 0.0
    Param.W3_DEFAULT = 0.0
    Param.EVAL_BATCH = 4
    Param.EVAL_ROLLOUTS = 5

    for name, module in (("gat", gat), ("upper", upper), ("lower", lower),
                         ("critic_u", critic_u), ("critic_l", critic_l)):
        module.load_state_dict(checkpoint[name], strict=True)
    for module in (gat, upper, lower, critic_u, critic_l):
        module.eval()
    print("loaded adopted checkpoint %s (K=%d, beta=%.3f)" %
          (checkpoint_path, n_heads, float(checkpoint.get("beta", 0.2))), flush=True)
    return Param, agent, cfg, n_heads


def to_param_instance(Param, instance: BenchInstance, global_max_proc: float,
                      global_max_capability: float = 1.0):
    jobs = []
    for job_idx, operations in enumerate(instance.jobs):
        jobs.append([
            Param.Operation(job_idx, op_idx, dict(machine_times))
            for op_idx, machine_times in enumerate(operations)
        ])
    due_dates = list(instance.due_dates)
    capabilities = [1] * instance.n_machines
    return Param.MOFJSPInstance(
        jobs, capabilities, due_dates, instance.name, instance.size_label,
        float(global_max_proc), float(global_max_capability),
    )


def benchmark_geometry(instances: Sequence[BenchInstance]) -> Tuple[int, int, int, float, float]:
    max_jobs = max(item.n_jobs for item in instances)
    max_ops = max(item.total_ops for item in instances)
    max_machines = max(item.n_machines for item in instances)
    max_proc = max(max(op.values()) for item in instances for job in item.jobs for op in job)
    max_due = max(max(item.due_dates) for item in instances)
    return max_jobs, max_ops, max_machines, float(max_proc), float(max_due)


def set_param_eval_settings(Param, rollouts: int, batch: int) -> None:
    Param.EVAL_ROLLOUTS = int(max(1, rollouts))
    Param.EVAL_BATCH = int(max(1, min(batch, rollouts)))


def evaluate_policy(Param, agent, cfg, instances: Sequence[BenchInstance],
                    geometry: Tuple[int, int, int, float, float],
                    rollouts: int = 21, batch: int = 7) -> Dict[str, float]:
    """Return the best makespan of one greedy and ``rollouts-1`` sampled decodes."""
    set_param_eval_settings(Param, rollouts, batch)
    param_instances = [
        to_param_instance(Param, item, geometry[3], 1.0) for item in instances
    ]
    _, _, _, _, f_list = Param.evaluate(
        agent, param_instances, geometry[0], geometry[1], geometry[2],
        geometry[3], geometry[4], cfg, deterministic=True,
    )
    return {item.name: float(value) for item, value in zip(instances, f_list)}


def save_public_checkpoint(path: Path, agent, state_norm, cfg) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gat": agent.gat.state_dict(),
        "upper": agent.upper.state_dict(),
        "lower": agent.lower.state_dict(),
        "critic_u": agent.critic_u.state_dict(),
        "critic_l": agent.critic_l.state_dict(),
        "state_norm_mean": state_norm.mean,
        "state_norm_var": state_norm.var,
        "state_norm_count": state_norm.count,
        "gamma": cfg.gamma,
        "beta": cfg.beta,
        "recipe": "public-benchmark-makespan-only",
    }, path)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    diff = a_arr[:, None] - b_arr[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    order = np.argsort(np.where(np.isnan(values), np.inf, values))
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        if np.isnan(values[index]):
            continue
        value = min(1.0, (m - rank) * float(values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def friedman_summary(rows: Sequence[dict], methods: Sequence[str]) -> dict:
    if not rows:
        return {"n": 0, "statistic": None, "p": None,
                "iman_davenport_F": None, "iman_davenport_p": None, "ranks": {}}
    matrix = np.asarray([[row[m] for m in methods] for row in rows], dtype=float)
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    if matrix.shape[0] < 2 or matrix.shape[1] < 3 or not HAS_SCIPY:
        return {"n": int(matrix.shape[0]), "statistic": None, "p": None,
                "iman_davenport_F": None, "iman_davenport_p": None, "ranks": {}}
    statistic, p_value = friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    n, k = matrix.shape
    denominator = n * (k - 1) - statistic
    if denominator > 1e-12:
        f_stat = (n - 1) * statistic / denominator
        p_f = float(f_dist.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    else:
        f_stat, p_f = float("inf"), 0.0
    ranks = rankdata(matrix, axis=1).mean(axis=0)
    return {
        "n": int(n),
        "statistic": float(statistic),
        "p": float(p_value),
        "iman_davenport_F": float(f_stat),
        "iman_davenport_p": float(p_f),
        "ranks": {method: float(rank) for method, rank in zip(methods, ranks)},
    }


def pairwise_summary(rows: Sequence[dict], reference: str,
                     methods: Sequence[str]) -> List[dict]:
    out = []
    ref = np.asarray([row[reference] for row in rows], dtype=float)
    p_values = []
    records = []
    for method in methods:
        if method == reference:
            continue
        other = np.asarray([row[method] for row in rows], dtype=float)
        mask = ~(np.isnan(ref) | np.isnan(other))
        a, b = ref[mask], other[mask]
        if a.size < 2:
            p_value = float("nan")
            statistic = float("nan")
        elif np.allclose(a, b):
            p_value, statistic = 1.0, 0.0
        elif HAS_SCIPY:
            statistic, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            statistic, p_value = float(statistic), float(p_value)
        else:
            statistic, p_value = float("nan"), float("nan")
        delta = cliffs_delta(a, b)
        p_values.append(p_value)
        records.append({
            "reference": reference,
            "method": method,
            "n": int(a.size),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "cliffs_delta": delta,
            "effect": effect_label(delta),
        })
    adjusted = holm_bonferroni(p_values)
    for record, p_adj in zip(records, adjusted):
        record["p_holm"] = p_adj
    return records


def effect_label(delta: float) -> str:
    magnitude = abs(delta) if not math.isnan(delta) else float("nan")
    if math.isnan(magnitude):
        return "na"
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _save_figure(fig, figure_dir: Path, stem: str) -> List[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf", ".eps", ".svg"):
        path = figure_dir / (stem + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    return paths


def make_figures(per_instance: Sequence[dict], stats: dict, figure_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    })
    colors = {
        "SPT": "#8c8c8c", "LBD": "#7f9c90",
        "FIFO": "#9a8fb5", "MOPNR": "#6f8faf", "MWKR": "#b07a7a",
        "MLP+DQN": "#b17f59", "MLP+PPO": "#7c9a92", "GAT+PPO": "#7b86a8",
        "HGNN": "#4c7a6f", "DAN": "#876f9f", "RESCHED": "#6b7f9e",
        "Proposed-policy": "#1f4e79", "Proposed-hybrid": "#c55a11",
    }
    methods = list(RULES) + list(LEARNING_BASELINES) + [
        "Proposed-policy", "Proposed-hybrid"]
    labels = {"Proposed-policy": "HGAT+PPO", "Proposed-hybrid": "HGAT+PPO"}
    output = []

    # Figure 1: gap to BKS by family.
    families = ["Brandimarte", "Lawrence"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, family in zip(axes, families):
        subset = [row for row in per_instance
                  if row["family"] == family and row.get("bks")]
        if not subset:
            ax.set_visible(False)
            continue
        data = [[row["gap_%s" % m] for row in subset if not math.isnan(row["gap_%s" % m])]
                for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False,
                        medianprops={"linewidth": 1.2, "color": "#333333"},
                        whiskerprops={"linewidth": 0.8},
                        capprops={"linewidth": 0.8})
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[method])
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.78)
        ax.axhline(0, color="#c55a11", linewidth=0.9, linestyle="--")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([labels.get(m, m) for m in methods], rotation=35, ha="right")
        ax.set_title("%s (n=%d)" % (family, len(subset)), pad=6)
        _style_axes(ax)
    axes[0].set_ylabel("Gap to BKS (%)")
    fig.tight_layout()
    output += _save_figure(fig, figure_dir, "Fig_PublicBenchmark_GapByFamily")
    plt.close(fig)

    # Figure 2: average Friedman ranks.
    all_stats = stats["overall"]
    if all_stats.get("ranks"):
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        methods_sorted = sorted(all_stats["ranks"], key=lambda m: all_stats["ranks"][m])
        values = [all_stats["ranks"][m] for m in methods_sorted]
        bars = ax.barh(range(len(methods_sorted)), values,
                       color=[colors[m] for m in methods_sorted],
                       edgecolor="#333333", linewidth=0.6, height=0.62)
        ax.set_yticks(range(len(methods_sorted)))
        ax.set_yticklabels([labels.get(m, m) for m in methods_sorted])
        ax.invert_yaxis()
        ax.set_xlabel("Average Friedman rank (lower is better)")
        ax.set_title("Ranking on %d held-out instances" % all_stats["n"], pad=6)
        for bar, value in zip(bars, values):
            ax.text(value + 0.04, bar.get_y() + bar.get_height() / 2,
                    "%.2f" % value, va="center", fontsize=7.5)
        _style_axes(ax)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.9)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        output += _save_figure(fig, figure_dir, "Fig_PublicBenchmark_AverageRanks")
        plt.close(fig)

    # Figure 3: win/tie/loss of the hybrid decoder against the best rule.
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    categories = ["Win", "Tie", "Loss"]
    bottoms = np.zeros(len(families))
    win_counts = statistics_counts(per_instance)
    for category, color in zip(categories, ["#2f6b5e", "#b9b9b9", "#b07a7a"]):
        values = np.asarray([win_counts[family][category] for family in families], dtype=float)
        ax.bar(families, values, bottom=bottoms, color=color,
               edgecolor="#333333", linewidth=0.6, label=category, width=0.55)
        bottoms += values
    ax.set_ylabel("Number of instances")
    ax.set_title("HGAT+PPO against the best rule", pad=6)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    _style_axes(ax)
    fig.tight_layout()
    output += _save_figure(fig, figure_dir, "Fig_PublicBenchmark_WinTieLoss")
    plt.close(fig)

    # Figure 4: per-instance relative makespan of the hybrid decoder.
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ordered = sorted([row for row in per_instance if row["family"] == "Brandimarte"],
                     key=lambda row: row["name"]) + \
              sorted([row for row in per_instance if row["family"] == "Lawrence"],
                     key=lambda row: row["name"])
    x = np.arange(len(ordered))
    y = [row["ratio_hybrid_best_rule"] for row in ordered]
    ax.plot(x, y, color=colors["Proposed-hybrid"], marker="o", markersize=2.6,
            linewidth=1.1)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([row["name"] for row in ordered], rotation=90, fontsize=6.5)
    ax.set_ylabel("Proposed / best rule")
    ax.set_title("Per-instance makespan relative to the best rule", pad=6)
    _style_axes(ax)
    fig.tight_layout()
    output += _save_figure(fig, figure_dir, "Fig_PublicBenchmark_PerInstanceRatio")
    plt.close(fig)
    return output


def statistics_counts(per_instance: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    counts = {family: {"Win": 0, "Tie": 0, "Loss": 0}
              for family in ("Brandimarte", "Lawrence")}
    for row in per_instance:
        if row["family"] not in counts:
            continue
        proposed = row["Proposed-hybrid"]
        best = min(row[rule] for rule in RULES)
        if proposed < best - 1e-9:
            counts[row["family"]]["Win"] += 1
        elif proposed > best + 1e-9:
            counts[row["family"]]["Loss"] += 1
        else:
            counts[row["family"]]["Tie"] += 1
    return counts


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_per_instance_rows(instances: Sequence[BenchInstance],
                            policy: Dict[str, float],
                            rules: Dict[str, Dict[str, float]],
                            baselines: Dict[str, Dict[str, float]],
                            tag: str) -> List[dict]:
    rows = []
    for instance in instances:
        values = dict(rules[instance.name])
        values.update({name: float(scores[instance.name])
                      for name, scores in baselines.items()})
        values["Proposed-policy"] = float(policy[instance.name])
        values["Proposed-hybrid"] = float(min(values["Proposed-policy"],
                                              min(values[rule] for rule in RULES)))
        best_rule = min(values[rule] for rule in RULES)
        row = {
            "name": instance.name,
            "family": instance.family,
            "jobs": instance.n_jobs,
            "machines": instance.n_machines,
            "operations": instance.total_ops,
            "bks": instance.bks,
            "best_rule": best_rule,
            "best_rule_name": min(RULES, key=lambda rule: values[rule]),
            "ratio_hybrid_best_rule": values["Proposed-hybrid"] / best_rule,
            "ratio_policy_best_rule": values["Proposed-policy"] / best_rule,
            "candidate_source": "policy" if values["Proposed-policy"] <= best_rule + 1e-9 else "rule",
        }
        for method in list(RULES) + list(baselines) + ["Proposed-policy", "Proposed-hybrid"]:
            row[method] = values[method]
            row["gap_%s" % method] = (
                (values[method] - instance.bks) / instance.bks * 100.0
                if instance.bks else float("nan")
            )
        rows.append(row)
    return rows


def summarize_methods(rows: Sequence[dict], methods: Sequence[str]) -> List[dict]:
    out = []
    for method in methods:
        values = np.asarray([row[method] for row in rows], dtype=float)
        gaps = np.asarray([row["gap_%s" % method] for row in rows], dtype=float)
        valid_gaps = gaps[~np.isnan(gaps)]
        out.append({
            "method": method,
            "n": int(values.size),
            "mean_makespan": float(values.mean()),
            "median_makespan": float(np.median(values)),
            "mean_gap_percent": float(valid_gaps.mean()) if valid_gaps.size else float("nan"),
            "median_gap_percent": float(np.median(valid_gaps)) if valid_gaps.size else float("nan"),
            "bks_count": int(valid_gaps.size),
        })
    return out


def compute_statistics(rows: Sequence[dict]) -> dict:
    # The primary test contains the deployed decoder once.  Policy-only is a
    # supplementary diagnostic and is not entered as a second proposed variant.
    methods = list(RULES) + list(LEARNING_BASELINES) + ["Proposed-hybrid"]
    policy_methods = list(LEARNING_BASELINES)
    overall = friedman_summary(rows, methods)
    pairwise = pairwise_summary(rows, "Proposed-hybrid", methods)
    policy_pairwise = pairwise_summary(rows, "Proposed-policy", policy_methods)
    families = {}
    for family in ("Brandimarte", "Lawrence"):
        subset = [row for row in rows if row["family"] == family]
        families[family] = {
            "n": len(subset),
            "friedman": friedman_summary(subset, methods),
            "pairwise_hybrid": pairwise_summary(subset, "Proposed-hybrid", methods),
            "pairwise_policy": pairwise_summary(subset, "Proposed-policy", policy_methods),
            "counts_hybrid_vs_best_rule": statistics_counts(subset),
        }
    return {
        "methods": methods,
        "supplementary_policy_method": "Proposed-policy",
        "overall": overall,
        "pairwise_hybrid": pairwise,
        "pairwise_policy": policy_pairwise,
        "counts_hybrid_vs_best_rule": statistics_counts(rows),
        "families": families,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def default_paths(project_root: Path):
    return (project_root / "Public_DataSet" / "PublicBenchmarkValidation",
            project_root / "Public_DataSet",
            project_root / "Figure_And_File" / "PublicBenchmarkValidation")


def project_root_from_script() -> Path:
    here = Path(__file__).resolve().parent
    if here.name == "common":
        return here.parent.parent
    return here.parent.parent if here.parent.name == "PPO_HGAT" else here.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--source-root", type=Path,
                        default=Path(r"D:\360MoveData\Users\lenovo\Desktop\撰写的论文\实验报告\参考数据集\fjsp-instances-main"))
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    code_data, public_data, _ = default_paths(args.project_root)
    if args.prepare:
        prepare_public_data(args.source_root, public_data, code_data)


if __name__ == "__main__":
    main()


