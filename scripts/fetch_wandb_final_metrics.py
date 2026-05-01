#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

import wandb


DEFAULT_TASKS = [
    "OfflineCarCircle-v0",
    "OfflineDroneRun-v0",
    "OfflineDroneCircle-v0",
    "OfflineCarRun-v0",
    "OfflineAntCircle-v0",
    "OfflineBallCircle-v0",
    "OfflineBallRun-v0",
    "OfflineMetadrive-easysparse-v0",
    "OfflineMetadrive-easydense-v0",
    "OfflineMetadrive-mediummean-v0",
    "OfflineMetadrive-mediumdense-v0",
    "OfflineMetadrive-hardsparse-v0",
    "OfflineMetadrive-hardmean-v0",
    "OfflineMetadrive-harddense-v0",
]
DEFAULT_ALGOS = ["bc-safe"]
DEFAULT_SEEDS = [0]

SUMMARY_KEYS = (
    "eval/NormalizedReward",
    "eval/NormalizedCost",
    "eval/Reward",
    "eval/Cost",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch final logged reward/cost metrics from W&B without re-evaluating checkpoints.")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default="OSRL-safetygym")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--output-json", dest="output_json")
    return parser.parse_args()


def resolve_entity(api: wandb.Api, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env_entity = os.getenv("WANDB_ENTITY")
    if env_entity:
        return env_entity
    for attr in ("default_entity", "entity"):
        value = getattr(api, attr, None)
        if isinstance(value, str) and value:
            return value
    viewer = getattr(api, "viewer", None)
    if viewer is not None:
        for attr in ("entity", "username"):
            value = getattr(viewer, attr, None)
            if isinstance(value, str) and value:
                return value
    raise SystemExit("Could not determine W&B entity. Pass --entity or set WANDB_ENTITY.")


def summary_dict(run) -> Dict[str, object]:
    summary = getattr(run, "summary", None)
    if summary is None:
        return {}
    json_dict = getattr(summary, "_json_dict", None)
    if isinstance(json_dict, dict):
        return json_dict
    try:
        return dict(summary)
    except Exception:
        return {}


def latest_history_values(run) -> Dict[str, float]:
    values: Dict[str, float] = {}
    try:
        for row in run.scan_history(keys=list(SUMMARY_KEYS)):
            for key in SUMMARY_KEYS:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    values[key] = float(value)
    except Exception:
        pass
    return values


def extract_metrics(run) -> Tuple[Optional[Dict[str, float]], str]:
    summary = summary_dict(run)
    metrics = {}
    missing = False
    for key in SUMMARY_KEYS:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
        else:
            missing = True
    if not missing:
        return metrics, "summary"

    history = latest_history_values(run)
    merged = dict(metrics)
    merged.update(history)
    for key in SUMMARY_KEYS:
        if key not in merged:
            return None, "missing"
    return merged, "history"


def candidate_sort_key(run):
    state = getattr(run, "state", "") or ""
    finished = 1 if state == "finished" else 0
    updated_at = getattr(run, "updated_at", "") or ""
    created_at = getattr(run, "created_at", "") or ""
    return (finished, updated_at, created_at)


def choose_best_run(runs: Iterable, task: str, algo: str, seed: int, include_unfinished: bool):
    target_name = f"{algo}-seed{seed}"
    candidates = [run for run in runs if getattr(run, "group", None) == task and getattr(run, "name", None) == target_name]
    if not candidates:
        return None, "run-not-found"
    finished_candidates = [run for run in candidates if getattr(run, "state", None) == "finished"]
    if finished_candidates:
        candidates = finished_candidates
    elif not include_unfinished:
        return None, "only-unfinished-runs-found"
    candidates.sort(key=candidate_sort_key, reverse=True)
    metric_candidates = []
    for run in candidates:
        metrics, source = extract_metrics(run)
        if metrics is not None:
            metric_candidates.append((run, metrics, source))
    if metric_candidates:
        metric_candidates.sort(key=lambda item: candidate_sort_key(item[0]), reverse=True)
        return metric_candidates[0], None
    return (candidates[0], None, "missing"), None


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def main():
    args = parse_args()
    api = wandb.Api()
    entity = resolve_entity(api, args.entity)
    project_path = f"{entity}/{args.project}"
    print(f"[wandb] project={project_path}")
    all_runs = list(api.runs(project_path))

    summaries = []
    missing = []

    for task in args.tasks:
        for algo in args.algorithms:
            algo_results = []
            for seed in args.seeds:
                chosen, missing_reason = choose_best_run(all_runs, task, algo, seed,
                                                         args.include_unfinished)
                if chosen is None:
                    missing.append({
                        "task": task,
                        "algorithm": algo,
                        "seed": seed,
                        "reason": missing_reason,
                    })
                    print(
                        f"[missing] task={task} algo={algo} seed={seed} "
                        f"reason={missing_reason}")
                    continue

                run, metrics, source = chosen
                if metrics is None:
                    missing.append({
                        "task": task,
                        "algorithm": algo,
                        "seed": seed,
                        "reason": "metrics not found",
                        "run_path": run.path,
                    })
                    print(f"[missing] task={task} algo={algo} seed={seed} reason=metrics-not-found run={run.path}")
                    continue

                result = {
                    "task": task,
                    "algorithm": algo,
                    "seed": seed,
                    "run_path": "/".join(run.path),
                    "run_id": run.id,
                    "state": getattr(run, "state", None),
                    "source": source,
                    "reward": metrics["eval/Reward"],
                    "cost": metrics["eval/Cost"],
                    "normalized_reward": metrics["eval/NormalizedReward"],
                    "normalized_cost": metrics["eval/NormalizedCost"],
                }
                algo_results.append(result)
                print(
                    "[wandb-run] "
                    f"task={task} algo={algo} seed={seed} "
                    f"reward={result['reward']:.5f} cost={result['cost']:.5f} "
                    f"normalized_reward={result['normalized_reward']:.5f} "
                    f"normalized_cost={result['normalized_cost']:.5f} "
                    f"source={source} run={result['run_path']}")

            if not algo_results:
                print(
                    "[wandb-summary] "
                    f"task={task} algo={algo} seeds=0/{len(args.seeds)} "
                    "avg_reward=missing avg_cost=missing "
                    "avg_normalized_reward=missing avg_normalized_cost=missing")
                continue

            reward_summary = summarize([x["reward"] for x in algo_results])
            cost_summary = summarize([x["cost"] for x in algo_results])
            normalized_reward_summary = summarize([x["normalized_reward"] for x in algo_results])
            normalized_cost_summary = summarize([x["normalized_cost"] for x in algo_results])
            summary = {
                "task": task,
                "algorithm": algo,
                "seeds": [x["seed"] for x in algo_results],
                "reward_mean": reward_summary["mean"],
                "reward_std": reward_summary["std"],
                "cost_mean": cost_summary["mean"],
                "cost_std": cost_summary["std"],
                "normalized_reward_mean": normalized_reward_summary["mean"],
                "normalized_reward_std": normalized_reward_summary["std"],
                "normalized_cost_mean": normalized_cost_summary["mean"],
                "normalized_cost_std": normalized_cost_summary["std"],
            }
            summaries.append(summary)
            print(
                "[wandb-summary] "
                f"task={task} algo={algo} seeds={len(algo_results)}/{len(args.seeds)} "
                f"avg_reward={summary['reward_mean']:.5f} "
                f"avg_cost={summary['cost_mean']:.5f} "
                f"avg_normalized_reward={summary['normalized_reward_mean']:.5f} "
                f"avg_normalized_cost={summary['normalized_cost_mean']:.5f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "entity": entity,
            "project": args.project,
            "summaries": summaries,
            "missing": missing,
        }, indent=2))
        print(f"[write] {output_path}")


if __name__ == "__main__":
    main()
