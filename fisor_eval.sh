#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(
    "OfflineCarCircle-v0"
    "OfflineDroneRun-v0"
    "OfflineDroneCircle-v0"
    "OfflineCarRun-v0"
    "OfflineAntCircle-v0"
    "OfflineBallCircle-v0"
    "OfflineBallRun-v0"
    "OfflineMetadrive-easysparse-v0"
    "OfflineMetadrive-easydense-v0"
    "OfflineMetadrive-mediummean-v0"
    "OfflineMetadrive-mediumdense-v0"
    "OfflineMetadrive-hardsparse-v0"
    "OfflineMetadrive-hardmean-v0"
    "OfflineMetadrive-harddense-v0"
  )
fi

SEEDS=(${SEEDS:-0})
ALGORITHMS=(${ALGORITHMS:-bc-safe})
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

cmd=(
  python scripts/eval_fisor_protocol.py
  --tasks "${TASKS[@]}"
  --algorithms "${ALGORITHMS[@]}"
  --seeds "${SEEDS[@]}"
  --logdir "$LOGDIR"
  --device "$DEVICE"
  --threads "$THREADS"
  --eval-episodes "$EVAL_EPISODES"
)

if [ "${#extra_args[@]}" -gt 0 ]; then
  cmd+=("${extra_args[@]}")
fi

"${cmd[@]}"
