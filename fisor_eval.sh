#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(
    "OfflineMetadrive-easymean-v0"
    "OfflineMetadrive-mediumsparse-v0"
    "OfflineAntRun-v0"
  )
fi

SEEDS=(${SEEDS:-0 1 2})
ALGORITHMS=(${ALGORITHMS:-cpq coptidice bc-safe})
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

echo "[fisor-eval] tasks=${TASKS[*]}"
echo "[fisor-eval] seeds=${SEEDS[*]}"
echo "[fisor-eval] algorithms=${ALGORITHMS[*]}"
echo "[fisor-eval] logdir=$LOGDIR"
echo "[fisor-eval] device=$DEVICE"
echo "[fisor-eval] threads=$THREADS"
echo "[fisor-eval] eval_episodes=$EVAL_EPISODES"

extra_args=()
if [ "${BEST:-0}" = "1" ]; then
  extra_args+=(--best)
fi
if [ -n "$OUTPUT_JSON" ]; then
  extra_args+=(--output-json "$OUTPUT_JSON")
fi

python scripts/eval_fisor_protocol.py \
  --tasks "${TASKS[@]}" \
  --algorithms "${ALGORITHMS[@]}" \
  --seeds "${SEEDS[@]}" \
  --logdir "$LOGDIR" \
  --device "$DEVICE" \
  --threads "$THREADS" \
  --eval-episodes "$EVAL_EPISODES" \
  "${extra_args[@]}"
