#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import bullet_safety_gym  # noqa
import dsrl  # noqa
import numpy as np
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa

from osrl.algorithms import BC, BCTrainer, CPQ, CPQTrainer, COptiDICE, COptiDICETrainer
from osrl.common.exp_util import load_config_and_model, seed_all
from osrl.common.fisor_protocol import (FISOR_PAPER_EVAL_EPISODES,
                                        get_fisor_paper_cost_limit)

DEFAULT_TASKS = [
    "OfflineMetadrive-easymean-v0",
    "OfflineMetadrive-mediumsparse-v0",
    "OfflineAntRun-v0",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OSRL checkpoints with the published FISOR paper protocol.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--algorithms", nargs="+", default=["cpq", "coptidice", "bc-safe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-episodes",
                        dest="eval_episodes",
                        type=int,
                        default=FISOR_PAPER_EVAL_EPISODES)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--output-json", dest="output_json")
    return parser.parse_args()


def make_env(task):
    if "Metadrive" in task:
        import gym as legacy_gym
        return legacy_gym.make(task)
    import gymnasium as gym  # noqa
    return gym.make(task)


def evaluate_cpq(run_dir: Path, eval_episodes: int, eval_cost_limit: int, device: str,
                 threads: int, best: bool):
    cfg, model = load_config_and_model(str(run_dir), best)
    seed_all(cfg["seed"])
    if device == "cpu":
        torch.set_num_threads(threads)

    raw_env = make_env(cfg["task"])
    env = wrap_env(env=raw_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    env.set_target_cost(eval_cost_limit)

    cpq_model = CPQ(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        vae_hidden_sizes=cfg["vae_hidden_sizes"],
        sample_action_num=cfg["sample_action_num"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        beta=cfg["beta"],
        num_q=cfg["num_q"],
        num_qc=cfg["num_qc"],
        qc_scalar=cfg["qc_scalar"],
        cost_limit=cfg["cost_limit"],
        episode_len=cfg["episode_len"],
        device=device,
    )
    cpq_model.load_state_dict(model["model_state"])
    cpq_model.to(device)

    trainer = CPQTrainer(cpq_model,
                         env,
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         device=device)
    try:
        ret, cost, length = trainer.evaluate(eval_episodes)
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        return {
            "reward": float(ret),
            "cost": float(cost),
            "normalized_reward": float(normalized_ret),
            "normalized_cost": float(normalized_cost),
            "length": float(length),
            "train_cost_limit": cfg["cost_limit"],
            "eval_cost_limit": eval_cost_limit,
        }
    finally:
        env.close()


def evaluate_coptidice(run_dir: Path, eval_episodes: int, eval_cost_limit: int, device: str,
                       threads: int, best: bool):
    cfg, model = load_config_and_model(str(run_dir), best)
    seed_all(cfg["seed"])
    if device == "cpu":
        torch.set_num_threads(threads)

    raw_env = make_env(cfg["task"])
    env = wrap_env(env=raw_env, reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    env.set_target_cost(eval_cost_limit)

    coptidice_model = COptiDICE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        f_type=cfg["f_type"],
        init_state_propotion=1.0,
        observations_std=np.array([0]),
        actions_std=np.array([0]),
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        gamma=cfg["gamma"],
        alpha=cfg["alpha"],
        cost_ub_epsilon=cfg["cost_ub_epsilon"],
        num_nu=cfg["num_nu"],
        num_chi=cfg["num_chi"],
        cost_limit=cfg["cost_limit"],
        episode_len=cfg["episode_len"],
        device=device,
    )
    coptidice_model.load_state_dict(model["model_state"])
    coptidice_model.to(device)

    trainer = COptiDICETrainer(coptidice_model,
                               env,
                               reward_scale=cfg["reward_scale"],
                               cost_scale=cfg["cost_scale"],
                               device=device)
    try:
        ret, cost, length = trainer.evaluate(eval_episodes)
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        return {
            "reward": float(ret),
            "cost": float(cost),
            "normalized_reward": float(normalized_ret),
            "normalized_cost": float(normalized_cost),
            "length": float(length),
            "train_cost_limit": cfg["cost_limit"],
            "eval_cost_limit": eval_cost_limit,
        }
    finally:
        env.close()


def evaluate_bc(run_dir: Path, eval_episodes: int, eval_cost_limit: int, device: str, threads: int,
                best: bool):
    cfg, model = load_config_and_model(str(run_dir), best)
    seed_all(cfg["seed"])
    if device == "cpu":
        torch.set_num_threads(threads)

    env = make_env(cfg["task"])
    env.set_target_cost(eval_cost_limit)

    state_dim = env.observation_space.shape[0]
    if cfg["bc_mode"] == "multi-task":
        state_dim += 1
    bc_model = BC(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        episode_len=cfg["episode_len"],
        device=device,
    )
    bc_model.load_state_dict(model["model_state"])
    bc_model.to(device)

    trainer = BCTrainer(bc_model,
                        env,
                        bc_mode=cfg["bc_mode"],
                        cost_limit=cfg["cost_limit"],
                        device=device)
    try:
        if cfg["bc_mode"] == "multi-task":
            trainer.set_target_cost(eval_cost_limit)
        ret, cost, length = trainer.evaluate(eval_episodes)
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        return {
            "reward": float(ret),
            "cost": float(cost),
            "normalized_reward": float(normalized_ret),
            "normalized_cost": float(normalized_cost),
            "length": float(length),
            "train_cost_limit": cfg["cost_limit"],
            "eval_cost_limit": eval_cost_limit,
        }
    finally:
        env.close()


def evaluate_run(algo: str, run_dir: Path, eval_episodes: int, eval_cost_limit: int, device: str,
                 threads: int, best: bool):
    if algo == "cpq":
        return evaluate_cpq(run_dir, eval_episodes, eval_cost_limit, device, threads, best)
    if algo == "coptidice":
        return evaluate_coptidice(run_dir, eval_episodes, eval_cost_limit, device, threads, best)
    if algo == "bc-safe":
        return evaluate_bc(run_dir, eval_episodes, eval_cost_limit, device, threads, best)
    raise ValueError(f"Unsupported algorithm: {algo}")


def summarize(values):
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def main():
    args = parse_args()
    summaries = []
    missing = []

    for task in args.tasks:
        eval_cost_limit = get_fisor_paper_cost_limit(task)
        for algo in args.algorithms:
            algo_results = []
            for seed in args.seeds:
                run_dir = Path(args.logdir) / task / f"{algo}-seed{seed}" / f"{algo}-seed{seed}"
                if not run_dir.exists():
                    missing.append(str(run_dir))
                    print(f"[missing] {run_dir}")
                    continue

                result = evaluate_run(algo, run_dir, args.eval_episodes, eval_cost_limit,
                                      args.device, args.threads, args.best)
                result.update({"task": task, "algorithm": algo, "seed": seed})
                algo_results.append(result)
                print(
                    "[eval] "
                    f"task={task} algo={algo} seed={seed} "
                    f"reward={result['reward']:.5f} normalized_reward={result['normalized_reward']:.5f} "
                    f"cost={result['cost']:.5f} normalized_cost={result['normalized_cost']:.5f} "
                    f"length={result['length']:.2f} eval_cost_limit={result['eval_cost_limit']} "
                    f"train_cost_limit={result['train_cost_limit']}")

            if not algo_results:
                continue

            reward_summary = summarize([x["normalized_reward"] for x in algo_results])
            cost_summary = summarize([x["normalized_cost"] for x in algo_results])
            raw_reward_summary = summarize([x["reward"] for x in algo_results])
            raw_cost_summary = summarize([x["cost"] for x in algo_results])
            summary = {
                "task": task,
                "algorithm": algo,
                "seeds": [x["seed"] for x in algo_results],
                "eval_cost_limit": eval_cost_limit,
                "eval_episodes": args.eval_episodes,
                "reward_mean": raw_reward_summary["mean"],
                "reward_std": raw_reward_summary["std"],
                "cost_mean": raw_cost_summary["mean"],
                "cost_std": raw_cost_summary["std"],
                "normalized_reward_mean": reward_summary["mean"],
                "normalized_reward_std": reward_summary["std"],
                "normalized_cost_mean": cost_summary["mean"],
                "normalized_cost_std": cost_summary["std"],
            }
            summaries.append(summary)
            print(
                "[summary] "
                f"task={task} algo={algo} "
                f"normalized_reward={summary['normalized_reward_mean']:.5f}±{summary['normalized_reward_std']:.5f} "
                f"normalized_cost={summary['normalized_cost_mean']:.5f}±{summary['normalized_cost_std']:.5f} "
                f"reward={summary['reward_mean']:.5f}±{summary['reward_std']:.5f} "
                f"cost={summary['cost_mean']:.5f}±{summary['cost_std']:.5f}")

    if args.output_json:
        payload = {"summaries": summaries, "missing": missing}
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"[write] {output_path}")

    if not summaries:
        raise SystemExit("No completed runs were found to evaluate.")


if __name__ == "__main__":
    main()
