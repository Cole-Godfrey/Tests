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

if [ "${#GPUS[@]}" -eq 0 ] || [ -z "${GPUS[0]}" ]; then
  echo "No GPUs selected. Set CUDA_DEVICES to a comma-separated list such as 0,1." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGDIR/stdout"

echo "[preflight] repo=$SCRIPT_DIR"
echo "[preflight] tasks=${TASKS[*]}"
echo "[preflight] seeds=${SEEDS[*]}"
echo "[preflight] algorithms=${ALGORITHMS[*]}"
echo "[preflight] gpus=${GPUS[*]}"
echo "[preflight] logdir=$LOGDIR"
if [ -n "${WANDB_MODE:-}" ]; then
  export WANDB_MODE
  echo "[preflight] wandb_mode=$WANDB_MODE"
else
  echo "[preflight] wandb_mode=wandb-default"
fi
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

python - <<'PY'
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pkg_resources is missing. Downgrade setuptools in this env, e.g. "
        "`pip install --force-reinstall \"setuptools<82\"`, then rerun ./run.sh."
    ) from exc
PY

python scripts/prefetch_dsrl_datasets.py "${TASKS[@]}"

run_single() {
  local gpu="$1"
  local algo="$2"
  local task="$3"
  local seed="$4"
  local module_path=""
  local safe_task="${task//[^A-Za-z0-9._-]/_}"
  local stdout_dir="$LOGDIR/stdout/$safe_task"
  local stdout_file="$stdout_dir/${algo}-seed${seed}.log"

  case "$algo" in
    cpq)
      module_path="examples.train.train_cpq"
      ;;
    coptidice)
      module_path="examples.train.train_coptidice"
      ;;
    *)
      echo "Unsupported algorithm: $algo" >&2
      return 1
      ;;
  esac

  mkdir -p "$stdout_dir"
  echo "[launch] gpu=$gpu algo=$algo task=$task seed=$seed"
  echo "[launch] stdout=$stdout_file"
  echo "[launch] streaming=terminal+log"

  CUDA_VISIBLE_DEVICES="$gpu" python -m "$module_path" \
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
    2>&1 | tee "$stdout_file"

  echo "[done] gpu=$gpu algo=$algo task=$task seed=$seed"
}

job_index=0
for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
      gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
      run_single "$gpu" "$algo" "$task" "$seed"
      job_index=$((job_index + 1))
    done
  done
done

echo "[done] all requested runs completed"
