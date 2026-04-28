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
