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
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"
SKIP_RUNS_RAW="${SKIP_RUNS:-}"
if [ -n "$SKIP_RUNS_RAW" ]; then
  read -r -a SKIP_RUNS <<< "$SKIP_RUNS_RAW"
else
  SKIP_RUNS=()
fi

PROJECT="${PROJECT:-OSRL-safetygym}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
NUM_WORKERS="${NUM_WORKERS:-4}"
THREADS="${THREADS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
UPDATE_STEPS="${UPDATE_STEPS:-1000000}"
SAVE_STDOUT_LOGS="${SAVE_STDOUT_LOGS:-0}"

if [ "${#GPUS[@]}" -eq 0 ] || [ -z "${GPUS[0]}" ]; then
  echo "No GPUs selected. Set CUDA_DEVICES to a comma-separated list such as 0,1." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$SAVE_STDOUT_LOGS" = "1" ]; then
  mkdir -p "$LOGDIR/stdout"
fi

echo "[preflight] repo=$SCRIPT_DIR"
echo "[preflight] tasks=${TASKS[*]}"
echo "[preflight] seeds=${SEEDS[*]}"
echo "[preflight] algorithms=${ALGORITHMS[*]}"
echo "[preflight] gpus=${GPUS[*]}"
echo "[preflight] logdir=$LOGDIR"
echo "[preflight] update_steps=$UPDATE_STEPS"
echo "[preflight] eval_episodes=$EVAL_EPISODES"
echo "[preflight] eval_protocol=fisor-paper"
echo "[preflight] save_stdout_logs=$SAVE_STDOUT_LOGS"
if [ "${#SKIP_RUNS[@]}" -eq 0 ]; then
  echo "[preflight] skip_runs=none"
else
  echo "[preflight] skip_runs=${SKIP_RUNS[*]}"
fi
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

python - <<'PY'
missing = []
for module_name in ("matplotlib", "wandb"):
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)

if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + ". Install them with `pip install -r requirements-server.txt`, then rerun ./run.sh."
    )
PY

python scripts/prefetch_dsrl_datasets.py "${TASKS[@]}"

resolve_fisor_cost_limit() {
  local task="$1"
  python - "$task" <<'PY'
import sys

from fisor_protocol import get_fisor_paper_cost_limit

print(get_fisor_paper_cost_limit(sys.argv[1]))
PY
}

run_single() {
  local gpu="$1"
  local algo="$2"
  local task="$3"
  local seed="$4"
  local module_path=""
  local cost_limit=""
  local -a extra_args=()
  local safe_task="${task//[^A-Za-z0-9._-]/_}"
  local stdout_dir="$LOGDIR/stdout/$safe_task"
  local stdout_file="$stdout_dir/${algo}-seed${seed}.log"
  local run_dir="$LOGDIR/$task/${algo}-seed${seed}"
  local completion_marker="$run_dir/.run_complete"
  cost_limit="$(resolve_fisor_cost_limit "$task")"

  case "$algo" in
    cpq)
      module_path="examples.train.train_cpq"
      ;;
    coptidice)
      module_path="examples.train.train_coptidice"
      ;;
    bc-safe)
      module_path="examples.train.train_bc"
      extra_args+=(--bc_mode safe)
      ;;
    *)
      echo "Unsupported algorithm: $algo" >&2
      return 1
      ;;
  esac

  if [ "$SAVE_STDOUT_LOGS" = "1" ]; then
    mkdir -p "$stdout_dir"
    echo "[launch] stdout=$stdout_file"
    echo "[launch] streaming=terminal+log"
  else
    echo "[launch] stdout=disabled"
    echo "[launch] streaming=terminal-only"
  fi
  mkdir -p "$run_dir"
  rm -f "$completion_marker"
  echo "[launch] gpu=$gpu algo=$algo task=$task seed=$seed"
  echo "[launch] cost_limit=$cost_limit"

  set +e
  if [ "$SAVE_STDOUT_LOGS" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python -m "$module_path" \
      --task "$task" \
      --device cuda:0 \
      --seed "$seed" \
      --threads "$THREADS" \
      --num_workers "$NUM_WORKERS" \
      --eval_episodes "$EVAL_EPISODES" \
      --update_steps "$UPDATE_STEPS" \
      --cost_limit "$cost_limit" \
      --project "$PROJECT" \
      --group "$task" \
      --name "${algo}-seed${seed}" \
      --logdir "$LOGDIR" \
      "${extra_args[@]}" \
      2>&1 | tee "$stdout_file"
    cmd_status=${PIPESTATUS[0]}
  else
    CUDA_VISIBLE_DEVICES="$gpu" python -m "$module_path" \
      --task "$task" \
      --device cuda:0 \
      --seed "$seed" \
      --threads "$THREADS" \
      --num_workers "$NUM_WORKERS" \
      --eval_episodes "$EVAL_EPISODES" \
      --update_steps "$UPDATE_STEPS" \
      --cost_limit "$cost_limit" \
      --project "$PROJECT" \
      --group "$task" \
      --name "${algo}-seed${seed}" \
      --logdir "$LOGDIR" \
      "${extra_args[@]}"
    cmd_status=$?
  fi
  set -e

  if [ "$cmd_status" -eq 139 ] && [[ "$task" == OfflineMetadrive-* ]] && [ -f "$completion_marker" ]; then
    echo "[warn] metadrive run segfaulted during shutdown after completing training; treating as completed"
    cmd_status=0
  fi

  if [ "$cmd_status" -ne 0 ]; then
    return "$cmd_status"
  fi

  echo "[done] gpu=$gpu algo=$algo task=$task seed=$seed"
}

should_skip_run() {
  local task="$1"
  local algo="$2"
  local seed="$3"
  local entry=""

  for entry in "${SKIP_RUNS[@]}"; do
    case "$entry" in
      "$task:$algo:$seed"|"$algo:$seed")
        return 0
        ;;
    esac
  done

  return 1
}

job_index=0
for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
      gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
      if should_skip_run "$task" "$algo" "$seed"; then
        echo "[skip] task=$task algo=$algo seed=$seed"
        job_index=$((job_index + 1))
        continue
      fi
      run_single "$gpu" "$algo" "$task" "$seed"
      job_index=$((job_index + 1))
    done
  done
done

echo "[done] all requested runs completed"
