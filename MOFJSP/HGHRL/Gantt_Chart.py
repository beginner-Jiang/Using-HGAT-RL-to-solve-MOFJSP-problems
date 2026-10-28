# -*- coding: utf-8 -*-
"""Publication-quality Gantt chart of the trained HGHRL model.

The canonical environment and networks of HGHRL.HGHRL are reused, so the node features,
the disjunctive edges and the decoding protocol are identical to the selected checkpoint.
The feature width is read from the checkpoint (3, 5 or 7 operation columns), hence the script
works with the legacy 3/3 model, the intermediate 5/3 model and the current 7/4 model.

The chart is drawn for the best of one greedy rollout and ``--samples`` sampled rollouts, the
same protocol the comparison experiments use.  Selecting the best schedule by the raw weighted
objective avoids the pessimistic picture a single greedy decode gives on the large instances.

The three objectives are annotated in a panel outside the axes: makespan, load balance and
total tardiness, each with its weight and its share of the normalised weighted objective,
together with the raw weighted objective J, the normalised objective F, machine utilisation
and the number of tardy jobs.  The last operation of a tardy job is outlined in dark red.
Legend and panel are stacked outside the axes, so no annotation overlaps another one or the
bars.  The default instance is the largest comparison scale, 100x30.
"""
import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from HGHRL import (
    MOFJSPInstance, HeteroGraphEnv, HeteroGAT, UpperPolicy, LowerPolicy,
    load_all_instances, read_fjsp_instance, cfg as HGAT_CFG,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
W1, W2, W3 = 0.4, 0.3, 0.3
DEFAULT_INSTANCE = os.path.join(ROOT, "DataSet", "comparison_instances", "comp_large_100x30_081.txt")
DEFAULT_MODEL = os.path.join(ROOT, "model", "hghrl", "hghrl_best.pth")
DEFAULT_OUTPUT = os.path.join(ROOT, "Figure_And_File", "HGHRL", "Gantt_Chart", "gantt_chart_100x30.png")

PAPER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
ACCENT = "#8B1A1A"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 600


class RunningStats:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def normalize(self, values):
        return (values - self.mean) / (self.var.sqrt() + EPS)


class GreedyDecoder:
    def __init__(self, instance, gat, upper, lower, stats):
        self.inst = instance
        self.gat = gat
        self.upper = upper
        self.lower = lower
        self.stats = stats

    @torch.no_grad()
    def act(self, state, deterministic=True):
        h_op, h_mac = self.gat(state)
        graph = h_op.mean(dim=0)
        if self.stats is not None:
            graph = self.stats.normalize(graph)
        op_probs = self.upper(graph, h_op, state["op_mask"])
        if deterministic:
            op_idx = int(torch.argmax(op_probs).item())
        else:
            op_idx = int(torch.distributions.Categorical(op_probs).sample().item())
        job = self.inst.op_idx_to_job[op_idx]
        op = self.inst.op_idx_to_op[op_idx]
        mac_mask = torch.zeros(h_mac.shape[0], dtype=torch.float32, device=DEVICE)
        mac_mask[:self.inst.n_machines] = self.inst.op_mac_mask[op_idx]
        mac_probs = self.lower(graph, h_op[op_idx], h_mac, mac_mask)
        if deterministic:
            machine = torch.argmax(mac_probs)
        else:
            machine = torch.distributions.Categorical(mac_probs).sample()
        return job, op, machine


def load_training_globals():
    _, max_jobs, max_machines, max_ops, max_proc, max_due, max_cap = load_all_instances(
        os.path.join(ROOT, "DataSet", "mo_fjsp_instances"), "mo_fjsp_*_train.txt")
    return max_jobs, max_machines, max_ops, max_proc, max_due, max_cap


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    dim_op = int(checkpoint["gat"]["l1.W_op.weight"].shape[1])
    dim_mac = int(checkpoint["gat"]["l1.W_mac.weight"].shape[1])
    mode = {3: "legacy", 5: "augmented", 7: "full"}.get(dim_op)
    if mode is None:
        raise ValueError("unsupported operation-feature width %d" % dim_op)
    gat = HeteroGAT(dim_op, dim_mac, HGAT_CFG.gat_hidden_dim, HGAT_CFG.gat_out_dim).to(DEVICE)
    upper = UpperPolicy(HGAT_CFG.gat_out_dim, HGAT_CFG.policy_hidden_dim).to(DEVICE)
    lower = LowerPolicy(HGAT_CFG.gat_out_dim, HGAT_CFG.policy_hidden_dim).to(DEVICE)
    gat.load_state_dict(checkpoint["gat"])
    upper.load_state_dict(checkpoint["upper"])
    lower.load_state_dict(checkpoint["lower"])
    gat.eval()
    upper.eval()
    lower.eval()
    stats = None
    if "state_norm_mean" in checkpoint and "state_norm_var" in checkpoint:
        stats = RunningStats(checkpoint["state_norm_mean"].to(DEVICE).float(),
                             checkpoint["state_norm_var"].to(DEVICE).float())
    return gat, upper, lower, stats, mode, dim_op, dim_mac


def rollout(instance, mode, decoder, maxima, actions=None, deterministic=True):
    """One rollout.  With ``actions`` the sequence is replayed instead of sampled."""
    max_jobs, max_machines, max_ops, max_proc, max_due, _ = maxima
    env = HeteroGraphEnv(instance, max_jobs, max(max_ops, instance.total_ops), max_machines,
                         max_proc, max_due, HGAT_CFG, feature_mode=mode)
    state = env.reset()
    records = []
    taken = []
    steps = 0
    done = False
    while not done and steps <= 20000:
        action = (actions[steps] if actions is not None
                  else decoder.act(state, deterministic=deterministic))
        job, op, machine = action
        op_idx = env.job_op_to_idx[job, op]
        predecessor = env.op_predecessor[op_idx]
        job_ready = env.op_finish_time[predecessor] if predecessor >= 0 else env.zero_scalar
        machine_ready = env.machine_available_time[machine]
        start = torch.maximum(job_ready, machine_ready)
        end = start + env.proc_time_matrix[op_idx, machine]
        records.append((int(job), int(op), int(machine), float(start), float(end)))
        taken.append((job, op, machine))
        state, _, done = env.step((job, op, machine))
        steps += 1
    return env, records, taken, steps


def raw_objectives(instance, records):
    cmax = max(end for _, _, _, _, end in records)
    loads = np.zeros(instance.n_machines, dtype=float)
    completion = np.zeros(instance.n_jobs, dtype=float)
    for job, _, machine, start, end in records:
        loads[machine] += end - start
        completion[job] = max(completion[job], end)
    lb = float(np.sqrt(np.mean((loads - loads.mean()) ** 2)))
    due = np.asarray(instance.due_dates, dtype=float)
    tardy = float(np.sum(np.maximum(0.0, completion - due)))
    return cmax, lb, tardy, loads, completion


def validate_schedule(instance, records):
    """Check completeness, precedence, machine non-overlap, capability and timing."""
    seen = set()
    duplicate = 0
    timing = 0
    capability = 0
    by_job = {}
    by_machine = {}
    for job, op, machine, start, end in records:
        key = (job, op)
        if key in seen:
            duplicate += 1
        seen.add(key)
        idx = instance.op_index_map[key]
        proc = float(instance.proc_time_matrix[idx, machine].item())
        if abs((end - start) - proc) > 1e-4:
            timing += 1
        if float(instance.op_mac_mask[idx, machine].item()) <= 0.5:
            capability += 1
        by_job.setdefault(job, []).append((op, start, end))
        by_machine.setdefault(machine, []).append((start, end))

    precedence = 0
    for sequence in by_job.values():
        sequence.sort()
        for (_, _, previous_end), (_, start, _) in zip(sequence, sequence[1:]):
            if start < previous_end - 1e-4:
                precedence += 1

    overlaps = 0
    for sequence in by_machine.values():
        sequence.sort()
        for (_, previous_end), (start, _) in zip(sequence, sequence[1:]):
            if start < previous_end - 1e-4:
                overlaps += 1

    missing = instance.total_ops - len(seen)
    ok = (duplicate == 0 and missing == 0 and precedence == 0 and overlaps == 0
          and capability == 0 and timing == 0)
    return {"ok": ok, "scheduled": len(seen), "total": instance.total_ops,
            "missing": missing, "duplicate": duplicate, "precedence": precedence,
            "overlaps": overlaps, "capability": capability, "timing": timing}


def lower_bound(instance):
    """Job-chain and machine-capacity lower bound on the makespan."""
    job_loads = []
    for job in range(instance.n_jobs):
        total = 0.0
        for op in range(instance.ops_per_job[job]):
            idx = instance.op_index_map[(job, op)]
            row = instance.proc_time_matrix[idx, :instance.n_machines]
            eligible = row[row < 99998.0]
            total += float(eligible.min().item())
        job_loads.append(total)
    return max(max(job_loads), sum(job_loads) / max(1, instance.n_machines))


def compute_metrics(env, records, maxima):
    cmax, lb, tardy, loads, completion = raw_objectives(env.inst, records)
    _, _, _, max_proc, max_due, _ = maxima
    num_ops = float(env.total_op_nodes)
    c1 = W1 * cmax / (max_proc * num_ops + EPS)
    c2 = W2 * lb / (max(loads.mean(), EPS))
    c3 = W3 * tardy / (max_due * num_ops + EPS)
    f = c1 + c2 + c3
    total = max(f, EPS)
    due = np.asarray(env.inst.due_dates, dtype=float)
    return {
        "cmax": cmax, "lb": lb, "tardy": tardy, "f": f,
        "j": W1 * cmax + W2 * lb + W3 * tardy,
        "s1": c1 / total, "s2": c2 / total, "s3": c3 / total,
        "util": float(loads.sum()) / max(1e-8, env.inst.n_machines * cmax),
        "tardy_jobs": int(np.sum(completion > due + 1e-6)),
        "loads": np.asarray(loads, dtype=float),
        "completion": np.asarray(completion, dtype=float),
        "due": due,
        "n_jobs": int(env.inst.n_jobs),
    }


def job_palette(count):
    if count <= len(PAPER_COLORS):
        return [PAPER_COLORS[i % len(PAPER_COLORS)] for i in range(count)]
    hues = np.linspace(0.0, 1.0, count, endpoint=False)
    return [mcolors.hsv_to_rgb([h, 0.42, 0.86]) for h in hues]


def _plot_gantt_legacy(env, records, metrics, validation, candidates, save_path, show=False):
    if not records:
        raise RuntimeError("empty schedule")
    n_m = env.inst.n_machines
    n_j = env.inst.n_jobs
    cmax = metrics["cmax"]
    colors = job_palette(n_j)
    width = 20.0 if n_m >= 20 else 14.0
    height = max(7.0, 0.42 * n_m + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))

    completion = np.zeros(n_j, dtype=float)
    for job, _, _, _, end in records:
        completion[job] = max(completion[job], end)
    due = np.asarray(env.inst.due_dates, dtype=float)
    tardy = completion > due + 1e-6
    last_op = {}
    for job, op, machine, start, end in records:
        last_op[job] = max(last_op.get(job, -1), op)

    for job, op, machine, start, end in records:
        tail = bool(tardy[job]) and op == last_op[job]
        ax.barh(y=machine, width=end - start, left=start, height=0.82,
                color=colors[job], edgecolor=ACCENT if tail else "#333333",
                linewidth=1.4 if tail else 0.2,
                hatch="//" if tail else None, alpha=0.90, zorder=2)
        if len(records) <= 70 and end - start > 0.06 * cmax:
            ax.text(start + (end - start) / 2.0, machine, "J%dO%d" % (job + 1, op + 1),
                    ha="center", va="center", fontsize=6.5, zorder=3)

    ax.axvline(cmax, color=ACCENT, linestyle="--", linewidth=1.4, zorder=1)
    ax.set_xlim(0.0, cmax * 1.02)
    ax.set_ylim(-0.7, n_m - 0.3)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(["M%d" % (m + 1) for m in range(n_m)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.5, color="#B0B0B0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.0, 1.008,
            "Cmax = %.1f (dashed line)    |    LB = %.2f (machine-load std; shaded bottleneck row)    |    Tardiness = %.0f (red-hatched last operations)"
            % (metrics["cmax"], metrics["lb"], metrics["tardy"]),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10.2,
            color="#222222")

    handles = [Patch(facecolor="white", edgecolor=ACCENT, hatch="//",
                     label="Tardy job (last operation)"),
               plt.Line2D([0], [0], color=ACCENT, linestyle="--", linewidth=1.4,
                          label="Makespan Cmax")]
    if n_j <= 10:
        handles = [Patch(facecolor=colors[j], edgecolor="#333333", label="Job %d" % (j + 1))
                   for j in range(n_j)] + handles
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=True, framealpha=0.95, edgecolor="#999999", ncol=1, title="Legend")

    panel = "\n".join([
        "Multi-objective coordination",
        "Cmax = %.1f  (weight 0.4, share %.1f%%)" % (metrics["cmax"], 100.0 * metrics["s1"]),
        "LB = %.2f  (weight 0.3, share %.1f%%)" % (metrics["lb"], 100.0 * metrics["s2"]),
        "Tardiness = %.1f  (weight 0.3, share %.1f%%)" % (metrics["tardy"], 100.0 * metrics["s3"]),
        "",
        "Raw weighted J = %.2f" % metrics["j"],
        "Normalised F = %.5f" % metrics["f"],
        "Machine utilisation = %.1f %%" % (100.0 * metrics["util"]),
        "Tardy jobs = %d / %d" % (metrics["tardy_jobs"], n_j),
    ])
    ax.text(1.005, 0.58, panel, transform=ax.transAxes, va="center", ha="left", fontsize=10,
            linespacing=1.6, color="#222222",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="#999999", linewidth=0.8, alpha=0.95))

    validation_panel = "\n".join([
        "Feasibility: %s" % ("PASS" if validation["ok"] else "FAIL"),
        "Operations: %d / %d" % (validation["scheduled"], validation["total"]),
        "Missing / duplicate: %d / %d" % (validation["missing"], validation["duplicate"]),
        "Precedence violations: %d" % validation["precedence"],
        "Machine overlaps: %d" % validation["overlaps"],
        "Capability violations: %d" % validation["capability"],
        "Timing violations: %d" % validation["timing"],
        "Candidates = %d; J best/mean/std = %.1f / %.1f / %.1f"
        % (candidates["n"], candidates["best"], candidates["mean"], candidates["std"]),
    ])
    ax.text(1.005, 0.21, validation_panel, transform=ax.transAxes, va="center", ha="left",
            fontsize=8.5, linespacing=1.45, color="#222222",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA",
                      edgecolor="#999999", linewidth=0.7, alpha=0.95))
    ax.text(1.005, 0.055, "Colour encodes jobs; dark-red outline marks the last tardy operation.",
            transform=ax.transAxes, va="center", ha="left", fontsize=7.5, style="italic",
            color="#555555")

    fig.subplots_adjust(left=0.06, right=0.80, top=0.97, bottom=0.08)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    vector_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(vector_path, bbox_inches="tight")
    svg_path = os.path.splitext(save_path)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    print("Gantt chart saved to %s" % save_path)
    print("Vector version saved to %s" % vector_path)
    print("SVG version saved to %s" % svg_path)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gantt chart of the trained HGHRL model")
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=50,
                        help="sampled rollouts next to the greedy one (default 50)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    maxima = load_training_globals()
    gat, upper, lower, stats, mode, dim_op, dim_mac = load_model(args.model)
    print("model=%s mode=%s dim_op=%d dim_mac=%d device=%s"
          % (args.model, mode, dim_op, dim_mac, DEVICE))

    jobs, caps, due, name, size = read_fjsp_instance(args.instance)
    instance = MOFJSPInstance(jobs, caps, due, name, size, maxima[3], maxima[5])
    print("instance=%s jobs=%d machines=%d operations=%d"
          % (instance.file_name, instance.n_jobs, instance.n_machines, instance.total_ops))

    decoder = GreedyDecoder(instance, gat, upper, lower, stats)
    best_key = float("inf")
    best_actions = None
    best_index = -1
    total = max(1, args.samples + 1)
    j_values = []
    for index in range(total):
        _, records, actions, steps = rollout(
            instance, mode, decoder, maxima, deterministic=(index == 0))
        cmax, lb, tardy, _, _ = raw_objectives(instance, records)
        key = W1 * cmax + W2 * lb + W3 * tardy
        j_values.append(key)
        print("candidate %d/%d: Cmax=%.1f LB=%.2f Tardy=%.1f J=%.2f"
              % (index + 1, total, cmax, lb, tardy, key))
        if key < best_key:
            best_key = key
            best_actions = actions
            best_index = index
    print("selected candidate %d with J=%.2f" % (best_index + 1, best_key))


    env, records, _, steps = rollout(instance, mode, decoder, maxima, actions=best_actions)
    metrics = compute_metrics(env, records, maxima)
    print("decisions=%d Cmax=%.2f LB=%.2f Tardy=%.2f J=%.2f F=%.5f U=%.1f%% tardy=%d/%d"
          % (steps, metrics["cmax"], metrics["lb"], metrics["tardy"], metrics["j"],
             metrics["f"], 100.0 * metrics["util"], metrics["tardy_jobs"], instance.n_jobs))
    validation = validate_schedule(instance, records)
    candidate_stats = {"n": len(j_values), "best": best_key,
                       "mean": float(np.mean(j_values)), "std": float(np.std(j_values))}
    print("feasibility=%s ops=%d/%d missing=%d duplicate=%d precedence=%d overlaps=%d"
          " capability=%d timing=%d"
          % ("PASS" if validation["ok"] else "FAIL", validation["scheduled"],
             validation["total"], validation["missing"], validation["duplicate"],
             validation["precedence"], validation["overlaps"], validation["capability"],
             validation["timing"]))
    print("candidates=%d J(best/mean/std)=%.1f/%.1f/%.1f"
          % (candidate_stats["n"], candidate_stats["best"],
             candidate_stats["mean"], candidate_stats["std"]))
    plot_gantt_split(env, records, metrics, validation, candidate_stats, args.output, show=args.show)
    return 0


def _plot_gantt_v2(env, records, metrics, validation, candidates, save_path, show=False):
    """Draw the Gantt chart with a structured, readable objective dashboard."""
    if not records:
        raise RuntimeError("empty schedule")

    n_m = env.inst.n_machines
    n_j = env.inst.n_jobs
    cmax = metrics["cmax"]
    loads = np.asarray(metrics.get("loads", np.zeros(n_m)), dtype=float)
    completion = np.asarray(metrics.get("completion", np.zeros(n_j)), dtype=float)
    due = np.asarray(metrics.get("due", np.zeros(n_j)), dtype=float)
    colors = job_palette(n_j)

    width = 22.0 if n_m >= 20 else 16.0
    height = max(10.0, 0.50 * n_m + 3.6)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.0], wspace=0.035,
                          left=0.055, right=0.985, top=0.925, bottom=0.070)
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])
    panel.set_xlim(0.0, 1.0)
    panel.set_ylim(0.0, 1.0)
    panel.axis("off")

    bottleneck = int(np.argmax(loads)) if n_m else 0
    # End the highlight at Cmax, not at the axis margin: the shaded row must stay inside
    # the schedule's right boundary.
    ax.add_patch(Rectangle((0.0, bottleneck - 0.5), cmax, 1.0,
                           facecolor="#F3E7D3", edgecolor="none", alpha=0.55,
                           zorder=0, clip_on=True))

    last_op = {}
    for job, op, machine, start, end in records:
        last_op[job] = max(last_op.get(job, -1), op)

    for job, op, machine, start, end in records:
        tail = bool(completion[job] > due[job] + 1e-6) and op == last_op[job]
        ax.barh(y=machine, width=end - start, left=start, height=0.82,
                color=colors[job], edgecolor=ACCENT if tail else "#333333",
                linewidth=1.4 if tail else 0.2,
                hatch="//" if tail else None, alpha=0.90, zorder=2)
        if len(records) <= 70 and end - start > 0.06 * cmax:
            ax.text(start + (end - start) / 2.0, machine, "J%dO%d" % (job + 1, op + 1),
                    ha="center", va="center", fontsize=6.5, zorder=3)

    ax.axvline(cmax, color=ACCENT, linestyle="--", linewidth=1.5, zorder=1)
    ax.set_xlim(0.0, cmax * 1.06)
    ax.set_ylim(-0.7, n_m - 0.3)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(["M%d" % (m + 1) for m in range(n_m)])
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylabel("Machine", fontsize=15)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.5, color="#B0B0B0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate("Cmax = %.1f" % cmax, xy=(cmax, n_m - 0.55),
                xytext=(cmax * 0.965, n_m - 1.55), ha="right", va="top",
                fontsize=10.5, color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=ACCENT, linewidth=0.7, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
    ax.text(cmax * 0.985, bottleneck + 0.34,
            "Bottleneck M%d (load %.0f)" % (bottleneck + 1, loads[bottleneck]),
            ha="right", va="bottom", fontsize=9.5, color="#7A5A2E",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFF8EC",
                          edgecolor="#C9A66B", linewidth=0.6, alpha=0.95))

    # Only the three worst tardy jobs are annotated, so the chart stays readable.
    tardy_amount = np.maximum(0.0, completion - due)
    placed = []
    for job in np.argsort(-tardy_amount)[:3]:
        if tardy_amount[job] <= 0:
            continue
        matches = [r for r in records if r[0] == job and r[1] == last_op[job]]
        if not matches:
            continue
        _, _, machine, _, end = matches[-1]
        ax.plot(end, machine, marker="v", color=ACCENT, markersize=5.5, zorder=6)
        offset = 0.012 * cmax
        x_text = end + offset
        ha = "left"
        if x_text > cmax * 0.94:
            x_text = end - offset
            ha = "right"
        y_text = machine + 0.30
        if any(abs(y_text - p) < 0.55 and abs(x_text - q) < 0.18 * cmax for p, q in placed):
            y_text = machine - 0.55
        placed.append((y_text, x_text))
        ax.annotate("J%d tardy" % (job + 1), xy=(end, machine),
                    xytext=(x_text, y_text), ha=ha, va="center",
                    fontsize=8.2, color=ACCENT,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor=ACCENT, linewidth=0.5, alpha=0.90))

    # Right-hand dashboard: one logical flow from the objective trade-off to the schedule
    # quality and the feasibility checks, followed by the two small diagnostic charts.
    panel.text(0.0, 0.975, "Multi-objective coordination", fontsize=16,
               fontweight="bold", va="top", color="#1A1A1A")
    panel.text(0.0, 0.938, "Weights, normalised contributions and schedule quality",
               fontsize=9.8, va="top", color="#555555")

    for x, text in ((0.0, "Objective"), (0.43, "Value"), (0.64, "Weight"), (0.82, "Share")):
        panel.text(x, 0.900, text, fontsize=9.8, fontweight="bold", va="top", color="#333333")
    rows = [("Cmax", metrics["cmax"], 0.4, metrics["s1"], "%.1f"),
            ("Load balance (LB)", metrics["lb"], 0.3, metrics["s2"], "%.2f"),
            ("Tardiness", metrics["tardy"], 0.3, metrics["s3"], "%.0f")]
    for idx, (name, value, weight, share, fmt) in enumerate(rows):
        y = 0.862 - idx * 0.038
        panel.text(0.0, y, name, fontsize=10.2, va="top", color="#222222")
        panel.text(0.43, y, fmt % value, fontsize=10.2, va="top", color="#222222")
        panel.text(0.64, y, "%.2f" % weight, fontsize=10.2, va="top", color="#222222")
        panel.text(0.82, y, "%.1f%%" % (100.0 * share), fontsize=10.2, va="top",
                  color="#222222")

    handles = [
        Patch(facecolor="white", edgecolor=ACCENT, hatch="//",
              label="Tardy job (last operation)"),
        plt.Line2D([0], [0], color=ACCENT, linestyle="--", linewidth=1.4,
                   label="Makespan Cmax"),
        Patch(facecolor="#F3E7D3", edgecolor="#C9A66B", label="Bottleneck machine"),
        Patch(facecolor="#4C72B0", edgecolor="#333333", label="Bar colour = job ID"),
    ]
    panel.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.765),
                 frameon=False, fontsize=9.5, handlelength=1.8,
                 labelspacing=0.42, borderaxespad=0.0)

    panel.text(0.0, 0.672, "Share of normalised F", fontsize=9.8,
               fontweight="bold", va="top", color="#333333")
    left = 0.0
    for share, color in ((metrics["s1"], "#4C72B0"),
                         (metrics["s2"], "#DD8452"),
                         (metrics["s3"], "#C44E52")):
        panel.barh(0.642, share, left=left, height=0.022, color=color,
                   edgecolor="white", linewidth=0.6)
        if share > 0.10:
            panel.text(left + share / 2.0, 0.642, "%.0f%%" % (100.0 * share),
                       ha="center", va="center", fontsize=9.0, color="white",
                       fontweight="bold")
        left += share
    panel.text(0.0, 0.615, "Raw J = %.1f    Normalised F = %.5f"
               % (metrics["j"], metrics["f"]), fontsize=10.2, va="top", color="#222222")

    panel.text(0.0, 0.578, "Schedule quality", fontsize=12.5, fontweight="bold",
               va="top", color="#1A1A1A")
    quality = [
        "Machine utilisation: %.1f %%" % (100.0 * metrics["util"]),
        "Tardy jobs: %d / %d" % (metrics["tardy_jobs"], n_j),
        "Candidates: %d; J best / mean / std = %.1f / %.1f / %.1f"
        % (candidates["n"], candidates["best"], candidates["mean"], candidates["std"]),
    ]
    for idx, text in enumerate(quality):
        panel.text(0.0, 0.543 - idx * 0.033, text, fontsize=10.0, va="top", color="#222222")

    ok = validation["ok"]
    panel.text(0.0, 0.440, "Feasibility: %s" % ("PASS" if ok else "FAIL"),
               fontsize=12.5, fontweight="bold", va="top",
               color="#2E7D32" if ok else "#C62828")
    checks = [
        "Operations: %d / %d" % (validation["scheduled"], validation["total"]),
        "Missing / duplicate: %d / %d" % (validation["missing"], validation["duplicate"]),
        "Precedence / overlap: %d / %d" % (validation["precedence"], validation["overlaps"]),
        "Capability / timing: %d / %d" % (validation["capability"], validation["timing"]),
    ]
    for idx, text in enumerate(checks):
        panel.text(0.0, 0.405 - idx * 0.033, text, fontsize=10.0, va="top", color="#222222")

    ax_load = panel.inset_axes([0.02, 0.170, 0.96, 0.095])
    machines = np.arange(1, n_m + 1)
    ax_load.bar(machines, loads, color="#6C8EBF", width=0.75,
                edgecolor="white", linewidth=0.2)
    ax_load.axhline(float(np.mean(loads)), color="#C44E52", linestyle="--", linewidth=1.0)
    ax_load.set_title("Machine loads (LB = %.2f)" % metrics["lb"], fontsize=9.8, pad=2)
    ax_load.text(0.02, 0.90, "LB = %.2f (std of loads)" % metrics["lb"],
                 transform=ax_load.transAxes, fontsize=8.0, fontweight="bold",
                 color="#C44E52", va="top")
    step = max(1, n_m // 6)
    ax_load.set_xticks(machines[::step])
    ax_load.set_xticklabels([str(v) for v in machines[::step]], fontsize=6.8)
    ax_load.tick_params(axis="y", labelsize=6.8)
    ax_load.set_ylabel("Load", fontsize=7.5)
    ax_load.spines["top"].set_visible(False)
    ax_load.spines["right"].set_visible(False)

    ax_tardy = panel.inset_axes([0.02, 0.030, 0.96, 0.095])
    order = [int(j) for j in np.argsort(-tardy_amount)[:5] if tardy_amount[j] > 0]
    if order:
        positions = np.arange(len(order))
        ax_tardy.barh(positions, tardy_amount[order], color="#C44E52",
                      alpha=0.85, height=0.70)
        ax_tardy.set_yticks(positions)
        ax_tardy.set_yticklabels(["J%d" % (j + 1) for j in order], fontsize=7.0)
        ax_tardy.invert_yaxis()
        ax_tardy.tick_params(axis="x", labelsize=6.8)
        ax_tardy.set_xlabel("Tardiness", fontsize=7.5)
        ax_tardy.set_title("Top tardy jobs (total %.0f)" % metrics["tardy"],
                           fontsize=9.5, pad=2)
        ax_tardy.spines["top"].set_visible(False)
        ax_tardy.spines["right"].set_visible(False)
        tardy_legend = ax_tardy.legend(
            handles=[Patch(facecolor="#C44E52", alpha=0.85, label="Tardiness")],
            loc="lower right", fontsize=10, frameon=False)
        for text in tardy_legend.get_texts():
            text.set_fontweight("bold")
    else:
        panel.text(0.02, 0.070, "No tardy jobs", fontsize=9.5, color="#2E7D32")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    vector_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(vector_path, bbox_inches="tight")
    svg_path = os.path.splitext(save_path)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    print("Gantt chart saved to %s" % save_path)
    print("Vector version saved to %s" % vector_path)
    print("SVG version saved to %s" % svg_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_gantt(env, records, metrics, validation, candidates, save_path, show=False):
    """Professional Gantt chart: graphical encodings, legend, compact metric table."""
    if not records:
        raise RuntimeError("empty schedule")
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    n_m = env.inst.n_machines
    n_j = env.inst.n_jobs
    cmax = metrics["cmax"]
    loads = np.asarray(metrics.get("loads", np.zeros(n_m)), dtype=float)
    completion = np.asarray(metrics.get("completion", np.zeros(n_j)), dtype=float)
    due = np.asarray(metrics.get("due", np.zeros(n_j)), dtype=float)
    colors = job_palette(n_j)

    width = 36.0 if n_m >= 20 else 21.0
    height = max(13.0, 0.55 * n_m + 4.5)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.0], wspace=0.020,
                          left=0.035, right=0.992, top=0.965, bottom=0.060)
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])
    panel.set_xlim(0.0, 1.0)
    panel.set_ylim(0.0, 1.0)
    panel.axis("off")

    bottleneck = int(np.argmax(loads)) if n_m else 0
    ax.axhspan(bottleneck - 0.5, bottleneck + 0.5, color="#F3E7D3", alpha=0.55, zorder=0)

    last_op = {}
    for job, op, machine, start, end in records:
        last_op[job] = max(last_op.get(job, -1), op)

    for job, op, machine, start, end in records:
        tail = bool(completion[job] > due[job] + 1e-6) and op == last_op[job]
        ax.barh(y=machine, width=end - start, left=start, height=0.82,
                color=colors[job], edgecolor=ACCENT if tail else "#333333",
                linewidth=1.5 if tail else 0.2,
                hatch="//" if tail else None, alpha=0.90, zorder=2)
        if len(records) <= 70 and end - start > 0.06 * cmax:
            ax.text(start + (end - start) / 2.0, machine, "J%dO%d" % (job + 1, op + 1),
                    ha="center", va="center", fontsize=7.0, fontweight="bold", zorder=3)

    ax.axvline(cmax, color=ACCENT, linestyle="--", linewidth=1.8, zorder=1)
    ax.text(cmax * 0.995, -0.45, "Cmax = %.1f" % cmax,
            ha="right", va="center", fontsize=13, fontweight="bold",
            color=ACCENT, clip_on=False, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor=ACCENT, linewidth=0.6, alpha=0.92))
    ax.set_xlim(0.0, cmax * 1.002)
    ax.set_ylim(-0.7, n_m - 0.3)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(["M%d" % (m + 1) for m in range(n_m)])
    ax.set_xlabel("Time", fontsize=19, fontweight="bold")
    ax.set_ylabel("Machine", fontsize=19, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10, direction="in", top=True, width=0.8)
    ax.tick_params(axis="y", labelsize=10, direction="in", right=True, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")
    ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.5, color="#B0B0B0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tardy_amount = np.maximum(0.0, completion - due)

    handles = [
        Patch(facecolor="white", edgecolor=ACCENT, hatch="//",
              label="Tardy job (last operation)"),
        plt.Line2D([0], [0], color=ACCENT, linestyle="--", linewidth=1.8,
                   label="Makespan Cmax"),
        Patch(facecolor="#F3E7D3", edgecolor="#C9A66B", label="Bottleneck machine"),
        Patch(facecolor="#4C72B0", edgecolor="#333333", label="Bar colour = job ID"),
    ]
    legend = panel.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.945),
                          frameon=False, fontsize=16.0, title="Left Gantt chart legend",
                          title_fontsize=19.0, handlelength=2.0,
                          labelspacing=0.50, borderaxespad=0.0, ncol=1)
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    panel.text(0.5, 0.855, "Schedule quality", fontsize=19.0, fontweight="bold",
               ha="center",
               va="top", color="#1A1A1A")
    rows = [
        ["Cmax", "%.1f" % metrics["cmax"]],
        ["LB (load balance)", "%.2f" % metrics["lb"]],
        ["Tardiness", "%.0f" % metrics["tardy"]],
        ["Raw weighted J", "%.1f" % metrics["j"]],
        ["Normalised F", "%.5f" % metrics["f"]],
        ["Machine utilisation", "%.1f %%" % (100.0 * metrics["util"])],
        ["Tardy jobs", "%d / %d" % (metrics["tardy_jobs"], n_j)],
        ["Feasibility", "PASS" if validation["ok"] else "FAIL"],
    ]
    ax_table = panel.inset_axes([0.02, 0.660, 0.96, 0.170])
    ax_table.set_xlim(0.0, 1.0)
    ax_table.set_ylim(0.0, 1.0)
    ax_table.axis("off")
    table = ax_table.table(cellText=rows, colLabels=["Metric", "Value"],
                           colWidths=[0.52, 0.48], cellLoc="center",
                           bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(16.0)
    n_rows = len(rows) + 1
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("white")
        cell.set_facecolor("#F2F2F2" if row == 0 else "white")
        cell.set_text_props(fontweight="bold", color="#1A1A1A", ha="center")
    ax_table.plot([0, 1], [1.0, 1.0], color="#333333", linewidth=1.2, clip_on=False)
    ax_table.plot([0, 1], [1.0 - 1.0 / n_rows, 1.0 - 1.0 / n_rows],
                 color="#333333", linewidth=0.9, clip_on=False)
    ax_table.plot([0, 1], [0.0, 0.0], color="#333333", linewidth=1.2, clip_on=False)

    ax_load = panel.inset_axes([0.04, 0.380, 0.92, 0.245])
    machines = np.arange(1, n_m + 1)
    ax_load.bar(machines, loads, color="#6C8EBF", width=0.75,
                edgecolor="white", linewidth=0.2)
    mean_load = float(np.mean(loads))
    ax_load.axhline(mean_load, color="#C44E52", linestyle="--", linewidth=1.2,
                   label="Mean load")
    ax_load.set_title("Machine loads (LB = %.2f)" % metrics["lb"],
                     fontsize=19.0, fontweight="bold", pad=3)
    step = max(1, n_m // 6)
    ax_load.set_xticks(machines[::step])
    ax_load.set_xticklabels([str(v) for v in machines[::step]])
    ax_load.set_xlabel("Machine", fontsize=14, fontweight="bold")
    ax_load.set_ylabel("Load", fontsize=14, fontweight="bold")
    for label in ax_load.get_xticklabels() + ax_load.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12)
    load_legend = ax_load.legend(fontsize=12, frameon=False, loc="upper right")
    for text in load_legend.get_texts():
        text.set_fontweight("bold")
    ax_load.spines["top"].set_visible(False)
    ax_load.spines["right"].set_visible(False)


    ax_tardy = panel.inset_axes([0.04, 0.075, 0.92, 0.245])
    order = [int(j) for j in np.argsort(-tardy_amount)[:5] if tardy_amount[j] > 0]
    if order:
        positions = np.arange(len(order))
        ax_tardy.barh(positions, tardy_amount[order], color="#C44E52",
                      alpha=0.85, height=0.70)
        ax_tardy.set_yticks(positions)
        ax_tardy.set_yticklabels(["J%d" % (j + 1) for j in order])
        ax_tardy.invert_yaxis()
        ax_tardy.set_xlabel("Tardiness", fontsize=14, fontweight="bold")
        ax_tardy.set_title("Top tardy jobs (total = %.0f)" % metrics["tardy"],
                           fontsize=19.0, fontweight="bold", pad=3)
        for label in ax_tardy.get_xticklabels() + ax_tardy.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(12)
        ax_tardy.spines["top"].set_visible(False)
        ax_tardy.spines["right"].set_visible(False)
    else:
        panel.text(0.03, 0.080, "No tardy jobs", fontsize=11.5, fontweight="bold",
                   color="#2E7D32")



    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")

    def save_vector(path, fmt):
        """Write a vector file; fall back to *_new when a viewer holds the target."""
        try:
            fig.savefig(path, format=fmt, bbox_inches="tight")
            print("%s version saved to %s" % (fmt.upper(), path))
            return path
        except PermissionError:
            alternative = os.path.splitext(path)[0] + "_new." + fmt
            fig.savefig(alternative, format=fmt, bbox_inches="tight")
            print("%s target is locked; saved to %s" % (fmt.upper(), alternative))
            return alternative

    print("Gantt chart saved to %s" % save_path)
    save_vector(os.path.splitext(save_path)[0] + ".pdf", "pdf")
    save_vector(os.path.splitext(save_path)[0] + ".svg", "svg")
    save_vector(os.path.splitext(save_path)[0] + ".eps", "eps")
    if show:
        plt.show()
    plt.close(fig)

def _save_vector_versions(fig, base_path):
    """Write pdf/svg/eps next to base_path, falling back when a file is locked."""
    for fmt in ("pdf", "svg", "eps"):
        path = base_path + "." + fmt
        try:
            fig.savefig(path, format=fmt, bbox_inches="tight")
            print("%s version saved to %s" % (fmt.upper(), path))
        except PermissionError:
            alternative = base_path + "_new." + fmt
            fig.savefig(alternative, format=fmt, bbox_inches="tight")
            print("%s target is locked; saved to %s" % (fmt.upper(), alternative))




def _plot_gantt(records, loads, completion, due, cmax, save_path, show=False):
    """Gantt chart with the legend inside the frame; the makespan stays a dashed line only."""
    records = [tuple(row) for row in records]
    loads = np.asarray(loads, dtype=float)
    completion = np.asarray(completion, dtype=float)
    due = np.asarray(due, dtype=float)
    n_m = len(loads)
    n_j = len(completion)
    colors = job_palette(n_j)
    base = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "font.size": 9,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "axes.linewidth": 0.8,
    })

    width = 9.6 if n_m >= 20 else 8.0
    height = max(6.2, 0.12 * n_m + 2.2)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.075, 0.085, 0.915, 0.900])

    bottleneck = int(np.argmax(loads)) if n_m else 0
    ax.axhspan(bottleneck - 0.5, bottleneck + 0.5, color="#F3E7D3", alpha=0.55, zorder=0)

    parsed = [(int(job), int(op), int(machine), float(start), float(end))
              for job, op, machine, start, end in records]
    last_op = {}
    for job, op, machine, start, end in parsed:
        last_op[job] = max(last_op.get(job, -1), op)

    for job, op, machine, start, end in parsed:
        tail = bool(completion[job] > due[job] + 1e-6) and op == last_op[job]
        ax.barh(y=machine, width=end - start, left=start, height=0.82,
                color=colors[job], edgecolor=ACCENT if tail else "#333333",
                linewidth=1.5 if tail else 0.2,
                hatch="//" if tail else None, alpha=0.90, zorder=2)
        if len(records) <= 70 and end - start > 0.06 * cmax:
            ax.text(start + (end - start) / 2.0, machine, "J%dO%d" % (job + 1, op + 1),
                    ha="center", va="center", fontsize=7.0, fontweight="bold", zorder=3)

    head = 5.5
    ax.axvline(cmax, color=ACCENT, linestyle="--", linewidth=1.4, zorder=1)
    ax.set_xlim(0.0, cmax * 1.05)
    ax.set_ylim(-0.7, n_m - 1 + head)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(["M%d" % (m + 1) for m in range(n_m)])
    ax.set_xticks(np.arange(0, cmax * 0.95, 200))
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.tick_params(axis="x", labelsize=11.5, direction="in", top=True, width=0.8)
    ax.tick_params(axis="y", labelsize=11.5, direction="in", right=True, width=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.5, color="#B0B0B0", zorder=0)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor="white", edgecolor=ACCENT, hatch="//",
              label="Tardy job (last operation)"),
        plt.Line2D([0], [0], color=ACCENT, linestyle="--", linewidth=1.4,
                   label="Makespan Cmax"),
        Patch(facecolor="#F3E7D3", edgecolor="#C9A66B", label="Bottleneck machine"),
        Patch(facecolor="#4C72B0", edgecolor="#333333", label="Bar colour = job ID"),
    ]
    legend = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.998, 0.998),
                       frameon=True, facecolor="white", edgecolor="#999999",
                       framealpha=0.96, fontsize=11, handlelength=1.8,
                       labelspacing=0.45, borderpad=0.5, borderaxespad=0.0, ncol=2,
                       columnspacing=1.2)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    _save_vector_versions(fig, base)
    if show:
        plt.show()
    plt.close(fig)


def _plot_panels(loads, completion, due, lb, tardy, base, show=False):
    """Two diagnostic panels drawn in the style of the Origin figures."""
    loads = np.asarray(loads, dtype=float)
    completion = np.asarray(completion, dtype=float)
    due = np.asarray(due, dtype=float)
    n_m = len(loads)

    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
    })

    fig, (ax_load, ax_tardy) = plt.subplots(
        1, 2, figsize=(8.25, 3.44), dpi=300,
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.26})
    for ax in (ax_load, ax_tardy):
        ax.tick_params(direction="in", top=True, right=True, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.grid(True, which="major", ls=":", lw=0.5, color="#DDDDDD")
        ax.set_axisbelow(True)

    machines = np.arange(1, n_m + 1)
    ax_load.bar(machines, loads, color="#2E6FBA", width=0.72, edgecolor="none")
    ax_load.axhline(float(np.mean(loads)), color="#8C8C8C", linestyle="--",
                    linewidth=1.2, label="Mean load")
    ax_load.set_title("(a) Machine loads (LB = %.2f)" % lb, pad=5)
    step = max(1, n_m // 10)
    ax_load.set_xticks(machines[::step])
    ax_load.set_xticklabels([str(v) for v in machines[::step]])
    ax_load.set_xlabel("Machine")
    ax_load.set_ylabel("Load")
    ax_load.legend(loc="upper right", frameon=False)

    tardy_amount = np.maximum(0.0, completion - due)
    order = [int(j) for j in np.argsort(-tardy_amount)[:5] if tardy_amount[j] > 0]
    if order:
        positions = np.arange(len(order))
        ax_tardy.barh(positions, tardy_amount[order], color="#C44E52",
                      alpha=0.95, height=0.68)
        ax_tardy.set_yticks(positions)
        ax_tardy.set_yticklabels(["J%d" % (j + 1) for j in order])
        ax_tardy.invert_yaxis()
        ax_tardy.set_xlabel("Tardiness")
        ax_tardy.set_xlim(0, 1200)
        ax_tardy.set_xticks([0, 300, 600, 900, 1200])
    else:
        ax_tardy.text(0.5, 0.5, "No tardy jobs", ha="center", va="center",
                      transform=ax_tardy.transAxes)
    ax_tardy.set_title("(b) Top tardy jobs (total = %.0f)" % tardy, pad=5)

    fig.tight_layout()
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    _save_vector_versions(fig, base)
    if show:
        plt.show()
    plt.close(fig)


def plot_gantt_split(env, records, metrics, validation, candidates, save_path, show=False):
    """Gantt chart with its legend only, plus the two diagnostic panels as a separate figure.

    The schedule-quality values are written to a csv file next to the figures, because the
    table is typeset by the manuscript rather than drawn inside the figure.
    """
    if not records:
        raise RuntimeError("empty schedule")
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    n_m = env.inst.n_machines
    n_j = env.inst.n_jobs
    cmax = metrics["cmax"]
    loads = np.asarray(metrics.get("loads", np.zeros(n_m)), dtype=float)
    completion = np.asarray(metrics.get("completion", np.zeros(n_j)), dtype=float)
    due = np.asarray(metrics.get("due", np.zeros(n_j)), dtype=float)
    colors = job_palette(n_j)
    base = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # ---------------- Figure 1: Gantt chart with its own legend ----------------
    _plot_gantt(records, loads, completion, due, cmax, save_path, show=show)

    np.savez(base + "_panels_data.npz", loads=loads, completion=completion, due=due,
             records=np.asarray(records, dtype=float), cmax=cmax, lb=metrics["lb"],
             tardy=metrics["tardy"])
    # ---------------- Figure 2: the two diagnostic panels, side by side ----------------
    _plot_panels(loads, completion, due, metrics["lb"], metrics["tardy"],
                 base + "_panels", show=show)


    rows = [
        ("cmax", "%.1f" % metrics["cmax"]),
        ("lb", "%.2f" % metrics["lb"]),
        ("tardiness", "%.0f" % metrics["tardy"]),
        ("raw_weighted_J", "%.1f" % metrics["j"]),
        ("normalised_F", "%.5f" % metrics["f"]),
        ("machine_utilisation", "%.1f" % (100.0 * metrics["util"])),
        ("tardy_jobs", "%d/%d" % (metrics["tardy_jobs"], n_j)),
        ("feasibility", "PASS" if validation["ok"] else "FAIL"),
    ]
    with open(base + "_metrics.csv", "w", encoding="utf-8") as fh:
        fh.write("metric,value\n")
        for key, value in rows:
            fh.write("%s,%s\n" % (key, value))
    print("Gantt chart saved to %s" % save_path)
    print("diagnostic panels saved to %s_panels.pdf" % base)


if __name__ == "__main__":
    raise SystemExit(main())



