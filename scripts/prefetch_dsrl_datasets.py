#!/usr/bin/env python3

import argparse

import bullet_safety_gym  # noqa: F401
import dsrl  # noqa: F401
import gymnasium as gym
from gymnasium.error import NameNotFound


def prefetch(task: str) -> None:
    try:
        env = gym.make(task)
    except NameNotFound as exc:
        if "OfflineMetadrive" in task:
            raise SystemExit(
                f"{task} is not registered. MetaDrive support is missing in this environment. "
                "Install it with `./scripts/install_metadrive_compat.sh`, "
                "then rerun ./run.sh."
            ) from exc
        raise
    dataset = env.get_dataset()
    dataset_path = getattr(getattr(env, "unwrapped", env), "dataset_filepath", None)
    print(f"[prefetch] task={task} observations={dataset['observations'].shape}")
    if dataset_path is not None:
        print(f"[prefetch] task={task} dataset={dataset_path}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DSRL datasets ahead of training to avoid first-run races."
    )
    parser.add_argument("tasks", nargs="+", help="DSRL task ids to prefetch")
    args = parser.parse_args()

    for task in args.tasks:
        prefetch(task)


if __name__ == "__main__":
    main()
