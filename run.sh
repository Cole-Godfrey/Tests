#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=("OfflineCarButton1Gymnasium-v0")
fi

SEEDS=(${SEEDS:-0 1 2})
ALGORITHMS=(${ALGORITHMS:-cpq coptidice})
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"

PROJECT="${PROJECT:-OSRL-safetygym}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
NUM_WORKERS="${NUM_WORKERS:-4}"
THREADS="${THREADS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [ "${#GPUS[@]}" -eq 0 ] || [ -z "${GPUS[0]}" ]; then
  echo "No GPUs selected. Set CUDA_DEVICES to a comma-separated list such as 0,1." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

mkdir -p "$LOGDIR/stdout"

echo "[preflight] repo=$SCRIPT_DIR"
echo "[preflight] tasks=${TASKS[*]}"
echo "[preflight] seeds=${SEEDS[*]}"
echo "[preflight] algorithms=${ALGORITHMS[*]}"
echo "[preflight] gpus=${GPUS[*]}"
echo "[preflight] logdir=$LOGDIR"
echo "[preflight] wandb_mode=$WANDB_MODE"
if [ -n "${DSRL_DATASET_DIR:-}" ]; then
  echo "[preflight] dsrl_dataset_dir=$DSRL_DATASET_DIR"
else
  echo "[preflight] dsrl_dataset_dir=~/.dsrl/datasets"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Activate the Linux conda environment on the GPU server first."
    )

print(f"[preflight] torch={torch.__version__} visible_gpus={torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    print(f"[preflight] gpu{idx}={torch.cuda.get_device_name(idx)}")
PY

python scripts/prefetch_dsrl_datasets.py "${TASKS[@]}"

job_dir="$(mktemp -d)"
trap 'rm -rf "$job_dir"' EXIT

job_index=0
for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
      worker_idx=$((job_index % ${#GPUS[@]}))
      printf '%s|%s|%s\n' "$algo" "$task" "$seed" >> "$job_dir/worker_${worker_idx}.jobs"
      job_index=$((job_index + 1))
    done
  done
done

run_single() {
  local gpu="$1"
  local algo="$2"
  local task="$3"
  local seed="$4"
  local script_path=""
  local safe_task="${task//[^A-Za-z0-9._-]/_}"
  local stdout_dir="$LOGDIR/stdout/$safe_task"
  local stdout_file="$stdout_dir/${algo}-seed${seed}.log"

  case "$algo" in
    cpq)
      script_path="examples/train/train_cpq.py"
      ;;
    coptidice)
      script_path="examples/train/train_coptidice.py"
      ;;
    *)
      echo "Unsupported algorithm: $algo" >&2
      return 1
      ;;
  esac

  mkdir -p "$stdout_dir"
  echo "[launch] gpu=$gpu algo=$algo task=$task seed=$seed"
  echo "[launch] stdout=$stdout_file"

  CUDA_VISIBLE_DEVICES="$gpu" python "$script_path" \
    --task "$task" \
    --device cuda:0 \
    --seed "$seed" \
    --threads "$THREADS" \
    --num_workers "$NUM_WORKERS" \
    --eval_episodes "$EVAL_EPISODES" \
    --project "$PROJECT" \
    --group "$task" \
    --name "${algo}-seed${seed}" \
    --logdir "$LOGDIR" \
    >"$stdout_file" 2>&1

  echo "[done] gpu=$gpu algo=$algo task=$task seed=$seed"
}

worker_pids=()
for idx in "${!GPUS[@]}"; do
  job_file="$job_dir/worker_${idx}.jobs"
  if [ ! -s "$job_file" ]; then
    continue
  fi

  gpu="${GPUS[$idx]}"
  (
    while IFS='|' read -r algo task seed; do
      [ -n "$algo" ] || continue
      run_single "$gpu" "$algo" "$task" "$seed"
    done < "$job_file"
  ) &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
