#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=("OfflineAntRun-v0")
fi

export SEEDS="${SEEDS:-0}"
export ALGORITHMS="${ALGORITHMS:-cpq coptidice bc-safe}"
export UPDATE_STEPS="${UPDATE_STEPS:-1000}"
export EVAL_EPISODES="${EVAL_EPISODES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export THREADS="${THREADS:-1}"

echo "[smoke] tasks=${TASKS[*]}"
echo "[smoke] seeds=$SEEDS"
echo "[smoke] algorithms=$ALGORITHMS"
echo "[smoke] update_steps=$UPDATE_STEPS"
echo "[smoke] eval_episodes=$EVAL_EPISODES"
echo "[smoke] num_workers=$NUM_WORKERS"
echo "[smoke] threads=$THREADS"

"$SCRIPT_DIR/run.sh" "${TASKS[@]}"
