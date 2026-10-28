# -*- coding: utf-8 -*-
"""Run the makespan-only public-benchmark validation experiment.

Examples
--------
Prepare the two public families and their manifest::

    python run_public_validation.py --stage prepare

Each family is split locally.  Brandimarte trains on MK01-MK05, selects on
MK06-MK07 and tests on MK08-MK10; Lawrence trains on la01-la20, selects on
la21-la25 and tests on la26-la40.  The adopted checkpoint and the learning
baselines are fine-tuned on one family at a time::

    python run_public_validation.py --stage train --family Brandimarte --episodes 600
    python run_public_validation.py --stage evaluate --family Brandimarte --rollouts 51

Use ``--family Lawrence`` for the other family or ``--family all`` for both.
The whole pipeline for one family is ``--stage all``.  Training is single-seed
(seed 42) and every reported test is paired by instance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
EXPERIMENT3 = HERE.parent / "Experiment3"
if str(EXPERIMENT3) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT3))

import public_benchmark as pb
import baseline_adapter as ba


def _safe_mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.mean(values)) if values else float("nan")


def _relative_gap(cmax: float, bks) -> float:
    if bks is None or not bks:
        return float("nan")
    return (float(cmax) - float(bks)) / float(bks) * 100.0


FAMILIES = ("Combined", "Brandimarte", "Lawrence")
CHECKPOINT_FAMILY = {"Combined": "Lawrence"}


def split_instances(instances: Sequence[pb.BenchInstance], family: str):
    """Return the train/validation/test split of one public protocol.

    ``Combined`` is the single public protocol: Lawrence la01--la20 is the
    adaptation split, la21--la25 is the validation split, and the held-out test
    set contains all ten Brandimarte instances plus Lawrence la26--la40.
    """
    if family == "Combined":
        train = [item for item in instances
                 if item.family == "Lawrence" and item.split == "train"]
        validation = [item for item in instances
                      if item.family == "Lawrence" and item.split == "validation"]
        test = [item for item in instances if item.split == "test"]
        return train, validation, test
    train = [item for item in instances
             if item.family == family and item.split == "train"]
    validation = [item for item in instances
                  if item.family == family and item.split == "validation"]
    test = [item for item in instances
            if item.family == family and item.split == "test"]
    return train, validation, test


def build_next_instance(instances: Sequence[pb.BenchInstance], seed: int = 42):
    order = list(instances)
    rng = random.Random(seed)

    def next_instance(_index: int):
        # A fresh permutation per sweep reduces short-period repetition while
        # remaining deterministic for the single training seed.
        if not order:
            raise RuntimeError("no training instances")
        return rng.choice(order)

    return next_instance


def evaluate_validation(Param, agent, cfg, validation, geometry, rollouts: int):
    policy = pb.evaluate_policy(Param, agent, cfg, validation, geometry,
                                rollouts=rollouts, batch=min(4, rollouts))
    gaps = [_relative_gap(policy[item.name], item.bks) for item in validation]
    known = [gap for gap in gaps if not math.isnan(gap)]
    # The validation score is the mean gap over instances with a published
    # reference.  If none is available, the normalised makespan is used.
    if known:
        score = float(np.mean(known))
    else:
        base = _safe_mean([policy[item.name] for item in validation])
        score = float(np.mean([policy[item.name] / base for item in validation]))
    return policy, float(score)


def save_training_log(path: Path, rows: List[dict], config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "history": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_prepare(args) -> List[pb.BenchInstance]:
    code_data, public_data, _ = pb.default_paths(args.project_root)
    print("== Preparing public data ==", flush=True)
    return pb.prepare_public_data(args.source_root, public_data, code_data)


def run_train(args) -> None:
    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    train, validation, test = split_instances(instances, args.family)
    print("== Public benchmark training split: %s ==" % args.family, flush=True)
    print("train=%d validation=%d test=%d"
          % (len(train), len(validation), len(test)), flush=True)
    if len(train) < 2 or len(validation) < 1:
        raise RuntimeError("train/validation split is empty; run --stage prepare first")

    geometry = pb.benchmark_geometry(instances)
    print("geometry: jobs=%d ops=%d machines=%d max_proc=%.1f max_due=%.2f"
          % geometry, flush=True)

    Param, agent, cfg, n_heads = pb.load_adopted_agent(args.project_root, args.checkpoint)
    if args.freeze_encoder:
        for parameter in agent.gat.parameters():
            parameter.requires_grad = False
        print("encoder frozen: only the policy heads are adapted", flush=True)
    train_param = [pb.to_param_instance(Param, item, geometry[3], 1.0)
                   for item in train]
    checkpoint_dir = code_data / "checkpoints" / CHECKPOINT_FAMILY.get(args.family, args.family)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_" + args.tag) if args.tag else ""
    best_path = checkpoint_dir / ("hghrl_public_best%s.pth" % suffix)
    last_path = checkpoint_dir / ("hghrl_public_last%s.pth" % suffix)
    history_path = code_data / ("training_history_%s%s.json"
                                % (args.family, ("_" + args.tag) if args.tag else ""))

    # Zero-shot baseline is measured before any public-instance gradient step.
    print("== Zero-shot validation and test probe ==", flush=True)
    zero_policy_val, zero_score = evaluate_validation(
        Param, agent, cfg, validation, geometry, args.probe_rollouts)
    zero_policy_test = pb.evaluate_policy(Param, agent, cfg, test, geometry,
                                          rollouts=args.probe_rollouts,
                                          batch=min(4, args.probe_rollouts))
    zero_rows = []
    for item in test:
        zero_rows.append({
            "name": item.name,
            "family": item.family,
            "cmax": zero_policy_test[item.name],
            "bks": item.bks,
            "gap_percent": _relative_gap(zero_policy_test[item.name], item.bks),
        })
    (code_data / ("zero_shot_test_%s.json" % args.family)).write_text(
        json.dumps({"validation_score": zero_score, "rows": zero_rows}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print("zero-shot validation relative gap %.3f%%" % zero_score, flush=True)

    # Fine-tune in bounded chunks so that a validation checkpoint can be
    # selected without changing the optimizer state.
    history: List[dict] = []
    best_score = float("inf")
    best_episode = 0
    next_instance = build_next_instance(train_param, seed=args.seed)
    started = time.time()
    for offset in range(0, args.episodes, args.chunk):
        chunk = min(args.chunk, args.episodes - offset)
        for group in agent.opt.param_groups:
            group["lr"] = args.lr
        print("== fine-tune episodes %d..%d (lr=%.2e) ==" %
              (offset + 1, offset + chunk, agent.opt.param_groups[0]["lr"]), flush=True)
        Param.train_collector(
            agent, cfg, next_instance, chunk, geometry,
            num_envs=args.envs, label="public-finetune", report_every=max(1, chunk // 4),
        )
        episodes_done = offset + chunk
        val_policy, val_score = evaluate_validation(
            Param, agent, cfg, validation, geometry, args.probe_rollouts)
        elapsed = time.time() - started
        row = {
            "episode": episodes_done,
            "validation_gap_percent": val_score,
            "elapsed_seconds": elapsed,
            "validation_cmax": {item.name: val_policy[item.name] for item in validation},
        }
        history.append(row)
        print("validation after %d: mean gap %.3f%% (best %.3f%% at %d)"
              % (episodes_done, val_score, best_score, best_episode), flush=True)
        pb.save_public_checkpoint(last_path, agent, agent.state_norm, cfg)
        if val_score < best_score:
            best_score = val_score
            best_episode = episodes_done
            pb.save_public_checkpoint(best_path, agent, agent.state_norm, cfg)
            print("saved new best checkpoint -> %s" % best_path, flush=True)
        save_training_log(history_path, history, {
            "seed": args.seed,
            "episodes": args.episodes,
            "chunk": args.chunk,
            "lr": args.lr,
            "envs": args.envs,
            "probe_rollouts": args.probe_rollouts,
            "geometry": list(geometry),
            "n_heads": n_heads,
            "best_episode": best_episode,
            "best_validation_gap_percent": best_score,
        })
    print("training finished in %.1f min; best validation gap %.3f%% at episode %d"
          % ((time.time() - started) / 60.0, best_score, best_episode), flush=True)

    if args.proposed_only:
        print("skip baseline fine-tuning (proposed-only run)", flush=True)
        return
    print("== Fine-tuning the three graph-neural-network baselines ==", flush=True)
    baseline_paths = ba.train_all(
        args.project_root, code_data, train, validation, geometry,
        episodes=args.baseline_episodes, envs_count=args.baseline_envs,
        lr=args.baseline_lr, seed=args.seed, family=args.family,
    )
    (code_data / ("baseline_checkpoints_%s.json" % args.family)).write_text(
        json.dumps(baseline_paths, indent=2, ensure_ascii=False), encoding="utf-8")
    print("baseline checkpoints: %s" % baseline_paths, flush=True)


def run_baselines(args) -> None:
    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    train, validation, _ = split_instances(instances, args.family)
    geometry = pb.benchmark_geometry(instances)
    print("== Fine-tuning the learning baselines: %s ==" % args.family, flush=True)
    baseline_paths = ba.train_all(
        args.project_root, code_data, train, validation, geometry,
        episodes=args.baseline_episodes, envs_count=args.baseline_envs,
        lr=args.baseline_lr, seed=args.seed, family=args.family,
    )
    (code_data / ("baseline_checkpoints_%s.json" % args.family)).write_text(
        json.dumps(baseline_paths, indent=2, ensure_ascii=False), encoding="utf-8")
    print("baseline checkpoints: %s" % baseline_paths, flush=True)


def run_select(args) -> None:
    """Choose, on the validation split only, between the zero-shot checkpoint
    and the best fine-tuned checkpoint of every adaptable model."""
    code_data, _, _ = pb.default_paths(args.project_root)
    instances, _ = pb.load_manifest(code_data)
    train, validation, test = split_instances(instances, args.family)
    geometry = pb.benchmark_geometry(instances)
    checkpoint_dir = code_data / "checkpoints" / CHECKPOINT_FAMILY.get(args.family, args.family)
    selected: Dict[str, str] = {}
    scores: Dict[str, Dict[str, float]] = {}

    zero_path = code_data / ("zero_shot_test_%s.json" % args.family)
    history_path = code_data / ("training_history_%s%s.json"
                                % (args.family, ("_" + args.tag) if args.tag else ""))
    fine_best = checkpoint_dir / "hghrl_public_best.pth"
    if zero_path.exists() and fine_best.exists():
        zero_score = float(json.loads(zero_path.read_text(encoding="utf-8"))["validation_score"])
        fine_score = float("inf")
        if history_path.exists():
            fine_score = float(json.loads(history_path.read_text(encoding="utf-8"))
                               ["config"].get("best_validation_gap_percent", float("inf")))
        if zero_score <= fine_score:
            selected["Proposed-policy"] = str(args.checkpoint)
        else:
            selected["Proposed-policy"] = str(fine_best)
        scores["Proposed-policy"] = {"zero_shot": zero_score, "fine_tuned": fine_score}

    internal_paths = pb.materialize_internal_files(list(validation), code_data / "internal")
    for name in ba.TRAINABLE:
        fine_best = checkpoint_dir / ("%s_public_best.pth" % name.lower())
        default = ba.default_checkpoint(args.project_root, name)
        if not fine_best.exists():
            selected[name] = str(default)
            continue
        module = ba.import_baseline(args.project_root, name)
        objects = ba._instance_objects(module, validation, internal_paths, geometry)
        entry = {}
        for label, path in (("zero_shot", default), ("fine_tuned", fine_best)):
            agent, _ = ba.load_agent(module, path, lr=2e-4)
            values = ba.evaluate_agent(module, agent, objects, geometry,
                                       rollouts=max(2, args.probe_rollouts),
                                       batch=min(4, max(2, args.probe_rollouts)))
            gaps = [(values[item.name] - item.bks) / item.bks * 100.0
                    for item in validation
                    if item.bks and item.name in values]
            entry[label] = float(np.mean(gaps)) if gaps else float("nan")
            del agent
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        chosen = default if entry["zero_shot"] <= entry["fine_tuned"] else fine_best
        selected[name] = str(chosen)
        scores[name] = entry
        print("    %s: zero-shot %.3f%% | fine-tuned %.3f%% | selected %s"
              % (name, entry["zero_shot"], entry["fine_tuned"], chosen.name), flush=True)
        del module, objects
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {"family": args.family, "criterion": "mean validation gap (%)",
               "rollouts": max(2, args.probe_rollouts),
               "checkpoints": selected, "scores": scores}
    (checkpoint_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("selection: %s" % json.dumps(selected, indent=2), flush=True)


def run_evaluate(args) -> None:
    code_data, _, figure_dir = pb.default_paths(args.project_root)
    instances, manifest = pb.load_manifest(code_data)
    train, validation, test = split_instances(instances, args.family)
    geometry = pb.benchmark_geometry(instances)

    checkpoint = args.checkpoint
    checkpoint_family = CHECKPOINT_FAMILY.get(args.family, args.family)
    public_best = code_data / "checkpoints" / checkpoint_family / "hghrl_public_best.pth"
    selection = ba.selection_map(code_data, checkpoint_family)
    if selection.get("Proposed-policy") and Path(selection["Proposed-policy"]).exists():
        checkpoint = Path(selection["Proposed-policy"])
    elif public_best.exists() and args.use_finetuned:
        checkpoint = public_best
    print("== Evaluating checkpoint %s ==" % checkpoint, flush=True)
    Param, agent, cfg, _ = pb.load_adopted_agent(args.project_root, checkpoint)
    agent.gat.eval(); agent.upper.eval(); agent.lower.eval()
    agent.critic_u.eval(); agent.critic_l.eval()

    print("== Dispatching rules ==", flush=True)
    rules = pb.evaluate_rules(test)
    print("== Proposed policy (%d rollouts per instance) ==" % args.rollouts, flush=True)
    policy_cache = (code_data / "cache" / args.family
                    / ("Proposed-policy_r%d_s%d.json" % (args.rollouts, args.seed)))
    if policy_cache.exists():
        policy = {key: float(value) for key, value in
                  json.loads(policy_cache.read_text(encoding="utf-8")).items()}
        print("    policy: cached", flush=True)
    else:
        policy = pb.evaluate_policy(Param, agent, cfg, test, geometry,
                                    rollouts=args.rollouts, batch=args.eval_batch)
        policy_cache.parent.mkdir(parents=True, exist_ok=True)
        policy_cache.write_text(json.dumps(policy, indent=1), encoding="utf-8")
    print("== Learning baselines (MLP+DQN, MLP+PPO, GAT+PPO, HGNN, DAN, RESCHED) ==", flush=True)
    baselines = ba.evaluate_all(args.project_root, code_data, test, geometry,
                               rollouts=args.rollouts, seed=args.seed, family=args.family,
                               selection=selection, cache_tag=args.family)
    all_methods = list(pb.RULES) + list(baselines) + [
        "Proposed-policy", "Proposed-hybrid"
    ]
    per_instance = pb.build_per_instance_rows(test, policy, rules, baselines, "finetuned")

    fields = ["name", "family", "jobs", "machines", "operations", "bks",
              "best_rule", "best_rule_name", "ratio_hybrid_best_rule",
              "ratio_policy_best_rule", "candidate_source"] + \
             all_methods + ["gap_%s" % method for method in all_methods]
    result_dir = code_data / "results" / args.family
    result_dir.mkdir(parents=True, exist_ok=True)
    pb.write_csv(result_dir / "per_instance.csv", per_instance, fields)
    summary_methods = pb.summarize_methods(per_instance, all_methods)
    pb.write_csv(result_dir / "method_summary.csv", summary_methods,
                 ["method", "n", "mean_makespan", "median_makespan",
                  "mean_gap_percent", "median_gap_percent", "bks_count"])
    stats = pb.compute_statistics(per_instance)
    (result_dir / "statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    pairwise_rows = []
    for key, reference in (("pairwise_hybrid", "Proposed-hybrid"),
                           ("pairwise_policy", "Proposed-policy")):
        for row in stats[key]:
            row = dict(row)
            row["comparison"] = key
            pairwise_rows.append(row)
    for family, blob in stats["families"].items():
        for key, reference in (("pairwise_hybrid", "Proposed-hybrid"),
                               ("pairwise_policy", "Proposed-policy")):
            for row in blob[key]:
                row = dict(row)
                row["comparison"] = key
                row["family"] = family
                pairwise_rows.append(row)
    pb.write_csv(result_dir / "pairwise_tests.csv", pairwise_rows,
                 ["family", "comparison", "reference", "method", "n",
                  "wilcoxon_statistic", "p_value", "p_holm", "cliffs_delta", "effect"])
    friedman_rows = [{"scope": "overall", **{k: v for k, v in stats["overall"].items()
                                             if k not in ("ranks",)},
                      "ranks": json.dumps(stats["overall"].get("ranks", {}))}]
    for family, blob in stats["families"].items():
        friedman_rows.append({"scope": family,
                              **{k: v for k, v in blob["friedman"].items()
                                 if k not in ("ranks",)},
                              "ranks": json.dumps(blob["friedman"].get("ranks", {}))})
    pb.write_csv(result_dir / "friedman_tests.csv", friedman_rows,
                 ["scope", "n", "statistic", "p", "iman_davenport_F",
                  "iman_davenport_p", "ranks"])
    counts = stats["counts_hybrid_vs_best_rule"]
    count_rows = []
    for family, values in counts.items():
        row = {"family": family}
        row.update(values)
        count_rows.append(row)
    pb.write_csv(result_dir / "win_tie_loss.csv", count_rows,
                 ["family", "Win", "Tie", "Loss"])

    # The public report uses the two protocol-specific figures produced by
    # analyze_public_experiments.py; avoid a third mixed-protocol figure set.
    figures = []
    (result_dir / "figure_manifest.json").write_text(
        json.dumps(figures, indent=2, ensure_ascii=False), encoding="utf-8")
    (result_dir / "experiment_manifest.json").write_text(
        json.dumps({
            "checkpoint": str(checkpoint),
            "seed": 42,
            "rollouts_per_instance": args.rollouts,
            "eval_batch": args.eval_batch,
            "family": args.family,
            "test_instances": [item.name for item in test],
            "methods": all_methods,
            "baseline_checkpoints": {
                name: str((code_data / "checkpoints" / args.family / (name.lower() + "_public_best.pth"))
                          if (code_data / "checkpoints" / args.family / (name.lower() + "_public_best.pth")).exists()
                          else ba.default_checkpoint(args.project_root, name))
                for name in baselines
            },
            "statistics": "Friedman/Iman-Davenport; paired Wilcoxon; Holm; Cliff's delta",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("== Results written ==", flush=True)
    print("data: %s" % result_dir, flush=True)
    print("figures: %s" % figure_dir, flush=True)
    for row in summary_methods:
        print("%-18s mean Cmax %.2f | mean gap %s%% | median gap %s%%"
              % (row["method"], row["mean_makespan"],
                 ("%.2f" % row["mean_gap_percent"]) if not math.isnan(row["mean_gap_percent"]) else "n/a",
                 ("%.2f" % row["median_gap_percent"]) if not math.isnan(row["median_gap_percent"]) else "n/a"),
              flush=True)
    print("overall Friedman p=%s; average ranks=%s"
          % (stats["overall"].get("p"), stats["overall"].get("ranks")), flush=True)
    print("win/tie/loss vs best rule: %s" % counts, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=pb.project_root_from_script())
    parser.add_argument("--source-root", type=Path,
                        default=Path(r"D:\360MoveData\Users\lenovo\Desktop\撰写的论文\实验报告\参考数据集\fjsp-instances-main"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage", choices=("prepare", "train", "baselines", "select",
                                            "evaluate", "all"),
                        default="all")
    parser.add_argument("--family", choices=("Combined", "Brandimarte", "Lawrence", "all"),
                        default="Combined")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-episodes", type=int, default=600)
    parser.add_argument("--baseline-envs", type=int, default=4)
    parser.add_argument("--baseline-lr", type=float, default=2e-4)
    parser.add_argument("--probe-rollouts", type=int, default=5)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--proposed-only", action="store_true")
    parser.add_argument("--rollouts", type=int, default=51)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--use-finetuned", action="store_true", default=True)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.project_root / "model" / "hghrl" / "hghrl_best.pth"

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    if args.stage in ("prepare", "all"):
        run_prepare(args)
    families = ("Combined",) if args.family == "all" else (args.family,)
    for family in families:
        args.family = family
        if args.stage in ("train", "all"):
            run_train(args)
        if args.stage == "baselines":
            run_baselines(args)
        if args.stage in ("select", "all"):
            run_select(args)
        if args.stage in ("evaluate", "all"):
            run_evaluate(args)


if __name__ == "__main__":
    main()
