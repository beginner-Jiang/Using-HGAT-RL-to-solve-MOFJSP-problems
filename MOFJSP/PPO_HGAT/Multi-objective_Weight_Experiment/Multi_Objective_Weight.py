# -*- coding: utf-8 -*-
"""Multi-objective weight study: why the objectives have to be coordinated, and which weighting to use.

The study has two levels, as the author asked.

Level 1 - the necessity of coordination.  The three corners of the simplex, (1,0,0), (0,1,0) and (0,0,1),
are trained next to the balanced weighting the method uses.  A corner optimises its own objective and pays
for it on the other two, which is the evidence that the objectives have to be traded against each other at
all.  The table prints, for every point, the three normalised sub-objectives and a column naming the
objectives on which the balanced weighting is better than that point.

Level 2 - the choice of the weighting.  A few weightings around the balanced one are trained and compared on
the weighted objective, with the spread of the per instance values, the confidence interval of the mean, and
the significance of the best against the runner-up.  This is where the weighting the method uses has to be
shown to be an appropriate one.

The protocol is the one of the parameter sensitivity study - the same instances, the same split, the same
budget of episodes, the same update structure and the same evaluation - so that the numbers of the two
studies are comparable.  The parameter study no longer scans the weights themselves; it defers to this
study, which is why the two exist side by side.

Output: tables only, written to Figure_And_File/PPO_HGAT/Multi-objective_Weight_Experiment/.

Usage:  python PPO_HGAT/Multi-objective_Weight_Experiment/Multi_Objective_Weight.py
"""
import os
import json
import random
import sys
import argparse

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (os.path.join(PROJECT_ROOT, "PPO_HGAT", "Param_Experiment"), PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import Param as P  # noqa: E402 - the protocol of the parameter study, reused on purpose

OUT_DIR = os.path.join(PROJECT_ROOT, "Figure_And_File", "PPO_HGAT", "Multi-objective_Weight_Experiment")

# Widths and step size of the recipe, the values the parameter study trains its configurations with.
ENC_WIDTH = 256
HIDDEN = 512
OUT = 256
LR = 2e-4
SEED = 42
# Heartbeat of a point: every 100 episodes the study says where it is, so that the log of the run
# shows the remaining budget of the study instead of staying silent for hours.  The evaluation of a
# single weighting decides how long the whole study takes, and the timing is printed with it.
PROGRESS_EPISODES = 100
# One file per point of the study.  The loop of a point is single-threaded and the host has idle
# cores while the device carries the training, so the points are split across processes to turn
CACHE_DIR = os.path.join(OUT_DIR, "measured_weights.d")
LEGACY_CACHE = os.path.join(OUT_DIR, "measured_weights.json")

BALANCED = (0.4, 0.3, 0.3)
# Level 1: the balanced weighting and the three corners of the simplex.
COORDINATION = [BALANCED, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
# Level 2: the balanced weighting and four points around it.
SELECTION = [BALANCED, (0.34, 0.33, 0.33), (0.5, 0.25, 0.25),
             (0.3, 0.3, 0.4), (0.3, 0.4, 0.3)]
OBJECTIVES = ("Cmax", "LB", "Tardy")
# Budget of the study.  It is the budget of the parameter study, and the environment variable is the
# handle a short validation run needs before the full study is queued.
EPOCHS = int(os.environ.get("MO_WEIGHT_EPOCHS", P.NUM_EPOCHS))


class Config:
    """The configuration of one run of this study: the weights plus the rest of the recipe."""

    def __init__(self, w1, w2, w3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.gamma = 1.0
        self.beta = 0.2
        self.K = 4
        self.reward_scaling = P.REWARD_SCALING
        self.reward_clip = P.REWARD_CLIP


def train_and_measure(weights, train_instances, test_instances, geometry):
    """Train one weighting with the recipe of the parameter study and evaluate it on the test set."""
    max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date = geometry
    # The seed fixes the sequence of training instances and the initialisation of this weighting, and
    # it is reset for every weighting: the tables of this study compare weightings on one recipe and
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cfg = Config(*weights)
    gat = P.HeteroGAT(3, 3, ENC_WIDTH, OUT, num_heads=cfg.K).to(P.DEVICE)
    upper = P.UpperPolicy(OUT, HIDDEN).to(P.DEVICE)
    lower = P.LowerPolicy(OUT, HIDDEN).to(P.DEVICE)
    critic_u = P.Critic(OUT, HIDDEN).to(P.DEVICE)
    critic_l = P.Critic(OUT, HIDDEN).to(P.DEVICE)
    state_norm = P.TorchRunningMeanStd(shape=(OUT,), device=P.DEVICE)
    agent = P.HierarchicalPPOAgent(None, gat, upper, lower, critic_u, critic_l,
                                   gamma=cfg.gamma, beta=cfg.beta, lr=LR, lam=1.0,
                                   state_norm=state_norm)
    # Curriculum of the draw.  The recipe this study measures trains its model on instances that grow
    # with the progress of the run (PPO_HGAT.Config.curriculum_episodes), while this function drew
    small = [item for item in train_instances if getattr(item, "size", "") == "small"] or train_instances
    medium = [item for item in train_instances if getattr(item, "size", "") == "medium"]
    large = [item for item in train_instances if getattr(item, "size", "") == "large"]

    def next_instance(episode_index):
        """Instance of the next episode of the schedule.

        ``episode_index`` counts the episodes the collector has started and starts at zero, so the
        first episode of the run carries the progress zero, which is the value the sequential loop
        of this function gave it.
        """
        progress = episode_index / max(1, EPOCHS)
        if progress < 0.5 or not medium:
            pool = small
        elif progress < 0.75 or not large:
            pool = small + medium
        else:
            pool = small + medium + large
        return random.choice(pool)

    # The collector of the parameter study runs the batch, so that a point of this study and a
    # configuration of that one are collected by the same code and their timings are comparable.
    P.train_collector(agent, cfg, next_instance, EPOCHS, geometry,
                      label=str(weights), report_every=PROGRESS_EPISODES)
    return P.evaluate(agent, test_instances, max_jobs, max_total_ops, max_machines, max_proc_time,
                      max_due_date, cfg, deterministic=True)


def cliffs_delta(a, b):
    """Cliff's delta of two samples: positive means the first tends to be larger."""
    greater = lesser = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                lesser += 1
    total = len(a) * len(b)
    return (greater - lesser) / total if total else 0.0


def normalise(series):
    """Min-max normalisation inside this study, so that the three objectives are one scale in the table."""
    low, high = min(series), max(series)
    span = high - low
    return [(value - low) / span if span > 0 else 0.0 for value in series]


def load_measurements(cache_dir, legacy_path):
    """Read every measurement of the study: one file per weight point, plus a legacy single file."""
    paths = []
    if os.path.isdir(cache_dir):
        paths = [os.path.join(cache_dir, name) for name in sorted(os.listdir(cache_dir))
                 if name.endswith(".json")]
    if legacy_path and os.path.isfile(legacy_path):
        paths.append(legacy_path)
    measured = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
        except (OSError, ValueError):
            continue
        for entry in entries:
            key = tuple(entry["w"])
            measured[key] = {name: entry[name] for name in
                             ("F", "Cmax", "LB", "Tardy", "F_list")}
    return measured


def save_measurement(cache_dir, weights, record):
    """Write one measurement to its own file, so concurrent processes cannot overwrite each other."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "w_%.2f_%.2f_%.2f.json" % weights)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump([{"w": list(weights), "F": record["F"], "Cmax": record["Cmax"],
                    "LB": record["LB"], "Tardy": record["Tardy"],
                    "F_list": record["F_list"]}], handle, indent=1)
    os.replace(tmp, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading instances...")
    all_instances, max_jobs, max_machines, max_total_ops, max_proc_time, max_due_date, _cap = \
        P.load_all_instances(P.INSTANCE_DIR, P.FILE_PATTERN)
    random.shuffle(all_instances)
    total = len(all_instances)
    train_end = int(P.TRAIN_RATIO * total)
    val_end = train_end + int(P.VAL_RATIO * total)
    train_tuples = all_instances[:train_end]
    test_tuples = all_instances[val_end:]
    geometry = (max_jobs, max_total_ops, max_machines, max_proc_time, max_due_date)
    train_instances = [P.MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, _cap)
                       for jobs, caps, dues, fname, size in train_tuples]
    test_instances = [P.MOFJSPInstance(jobs, caps, dues, fname, size, max_proc_time, _cap)
                      for jobs, caps, dues, fname, size in test_tuples]
    print("Training set: %d, test set: %d" % (len(train_instances), len(test_instances)))

    planned = []
    for weights in COORDINATION + SELECTION:
        if weights not in planned:
            planned.append(weights)
    print("Weightings to train: %d (budget %d episodes each)" % (len(planned), EPOCHS))

    measured = {}
    parser = argparse.ArgumentParser(description="Multi-objective weight study of the proposed method.")
    # The points of the plan this process trains, as indices into it, so that the study can be split
    # across processes; an empty list trains every point whose measurement is not on disk yet.
    parser.add_argument("--points", default="",
                        help="comma separated indices into the plan, e.g. 0,1,2 (default: all open)")
    arguments = parser.parse_args()
    wanted = {int(part) for part in arguments.points.split(",") if part.strip()}

    # One weighting costs the training of the budget plus the evaluation of the test set, which is
    # hours under the load of this host; the measurement of a point travels to disk as soon as it is
    measured = load_measurements(CACHE_DIR, LEGACY_CACHE)
    if measured:
        print("Restored %d weighting(s) from %s" % (len(measured), os.path.basename(CACHE_DIR)))
    for index, weights in enumerate(planned):
        if wanted and index not in wanted:
            continue
        if weights in measured:
            print("\n========== The weighting %s is already measured ==========" % (weights,))
            continue
        print("\n========== Training the weighting %s ==========" % (weights,))
        weighted, cmax, lb, tardy, per_instance = train_and_measure(
            weights, train_instances, test_instances, geometry)
        record = {'F': weighted, 'Cmax': cmax, 'LB': lb, 'Tardy': tardy,
                  'F_list': per_instance}
        measured[weights] = record
        print("    F=%.4f  Cmax=%.1f  LB=%.2f  Tardy=%.1f" % (weighted, cmax, lb, tardy))
        save_measurement(CACHE_DIR, weights, record)

    open_points = [w for w in planned if w not in measured]
    if open_points:
        # The tables are normalised over every point of the study, so they are written by the process
        # that finds the last open measurement on disk; a run without --points finishes the study
        # once all of them are measured.
        print("\n%d of %d weightings are still open; the tables wait for them"
              % (len(open_points), len(planned)))
        return 0

    # The normalisation of the table is taken over the points measured here, so that the columns of the
    # three objectives are comparable with each other inside this study.
    cmax_norm = normalise([measured[w]['Cmax'] for w in planned])
    lb_norm = normalise([measured[w]['LB'] for w in planned])
    tardy_norm = normalise([measured[w]['Tardy'] for w in planned])
    normalised = {w: (cmax_norm[i], lb_norm[i], tardy_norm[i]) for i, w in enumerate(planned)}
    balanced_norm = normalised[BALANCED]

    # Level 1: which objectives does the balanced weighting win against each point of the simplex.
    rows = []
    for weights in COORDINATION:
        norm = normalised[weights]
        better_on = [OBJECTIVES[i] for i in range(3) if balanced_norm[i] < norm[i]]
        rows.append({"weights": "(%g,%g,%g)" % weights, "F": measured[weights]['F'],
                     "Cmax": measured[weights]['Cmax'], "LB": measured[weights]['LB'],
                     "Tardy": measured[weights]['Tardy'],
                     "Cmax_normalised": norm[0], "LB_normalised": norm[1],
                     "Tardy_normalised": norm[2],
                     "balanced_better_on": ",".join(better_on) or "none"})
    level1 = pd.DataFrame(rows)
    level1.to_csv(os.path.join(OUT_DIR, "level1_coordination.csv"), index=False)
    print("\n========== Level 1: the necessity of coordination ==========")
    print(level1.to_string(index=False))

    # Level 2: which weighting of the neighbourhood around the balanced one is the appropriate one.
    rows = []
    for weights in SELECTION:
        norm = normalised[weights]
        values = measured[weights]['F_list']
        # Every point of this level is compared with the weighting the method uses, on the per instance
        # values of the same test set: this is the level that has to show that weighting to be the
        # appropriate one, so the comparison belongs in its table instead of in a table of its own.
        reference = measured[BALANCED]['F_list']
        delta = cliffs_delta(values, reference)
        try:
            from scipy.stats import wilcoxon
            _, p_value = wilcoxon(values, reference)
        except Exception as exc:  # noqa: BLE001 - the table survives a missing scipy
            p_value = float('nan')
            print("    wilcoxon unavailable: %s" % exc)
        magnitude = abs(delta)
        effect = ("negligible" if magnitude < 0.147 else
                  "small" if magnitude < 0.33 else
                  "medium" if magnitude < 0.474 else "large")
        rows.append({"weights": "(%g,%g,%g)" % weights, "F_mean": measured[weights]['F'],
                     "F_sd": float(np.std(values)),
                     "F_ci95": 1.96 * float(np.std(values)) / np.sqrt(max(1, len(values))),
                     "Cmax": measured[weights]['Cmax'], "LB": measured[weights]['LB'],
                     "Tardy": measured[weights]['Tardy'],
                     "Cmax_normalised": norm[0], "LB_normalised": norm[1],
                     "Tardy_normalised": norm[2], "epochs": EPOCHS,
                     "delta_vs_current": delta, "p_vs_current": p_value,
                     "effect_vs_current": effect})
    level2 = pd.DataFrame(rows).sort_values("F_mean").reset_index(drop=True)
    level2.to_csv(os.path.join(OUT_DIR, "level2_weight_selection.csv"), index=False)
    print("\n========== Level 2: the choice of the weighting ==========")
    print(level2.to_string(index=False))

    print("\nTables written to %s" % OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())



