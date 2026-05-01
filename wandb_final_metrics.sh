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
PROJECT="${PROJECT:-OSRL-safetygym}"
ENTITY="${ENTITY:-${WANDB_ENTITY:-}}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
INCLUDE_UNFINISHED="${INCLUDE_UNFINISHED:-0}"

echo "[wandb-metrics] project=$PROJECT"
echo "[wandb-metrics] entity=${ENTITY:-auto}"
echo "[wandb-metrics] tasks=${TASKS[*]}"
echo "[wandb-metrics] seeds=${SEEDS[*]}"
echo "[wandb-metrics] algorithms=${ALGORITHMS[*]}"
echo "[wandb-metrics] include_unfinished=$INCLUDE_UNFINISHED"

extra_args=()
if [ -n "$ENTITY" ]; then
  extra_args+=(--entity "$ENTITY")
fi
if [ -n "$OUTPUT_JSON" ]; then
  extra_args+=(--output-json "$OUTPUT_JSON")
fi
if [ "$INCLUDE_UNFINISHED" = "1" ]; then
  extra_args+=(--include-unfinished)
fi

cmd=(
  python scripts/fetch_wandb_final_metrics.py
  --project "$PROJECT"
  --tasks "${TASKS[@]}"
  --algorithms "${ALGORITHMS[@]}"
  --seeds "${SEEDS[@]}"
)

if [ "${#extra_args[@]}" -gt 0 ]; then
  cmd+=("${extra_args[@]}")
fi

"${cmd[@]}"
