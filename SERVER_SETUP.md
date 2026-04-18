# Linux Server Setup

This repo now includes a Linux/CUDA conda environment file and a `run.sh` launcher that defaults to:

- `OfflineCarButton1Gymnasium-v0`
- algorithms: `CPQ` and `COptiDICE`
- seeds: `0 1 2`
- one training process per GPU across `CUDA_DEVICES=0,1`

## 1. Create the conda environment

From the repo root on the Linux server:

```bash
conda env create -f environment.yml
conda activate osrl
pip install -e . --no-deps
```

If you update `environment.yml` later, refresh with:

```bash
conda env update -f environment.yml --prune
conda activate osrl
pip install -e . --no-deps
```

## 2. Optional environment variables

Use offline W&B logging by default:

```bash
export WANDB_MODE=offline
```

If you want the DSRL datasets on scratch storage instead of the default `~/.dsrl/datasets`:

```bash
export DSRL_DATASET_DIR=/path/to/shared-or-scratch-storage/dsrl
mkdir -p "$DSRL_DATASET_DIR"
```

## 3. Launch the default 6 runs

```bash
chmod +x run.sh
./run.sh
```

That launches:

- `train_cpq.py` on `OfflineCarButton1Gymnasium-v0` for seeds `0,1,2`
- `train_coptidice.py` on `OfflineCarButton1Gymnasium-v0` for seeds `0,1,2`

The script first verifies CUDA, then downloads the DSRL dataset once, then schedules one job per GPU.

## 4. Useful overrides

Pick different GPUs:

```bash
CUDA_DEVICES=2,3 ./run.sh
```

Reduce CPU pressure on a busy shared machine:

```bash
THREADS=2 NUM_WORKERS=2 ./run.sh
```

Run additional Safety Gymnasium tasks:

```bash
./run.sh OfflineCarButton1Gymnasium-v0 OfflineCarGoal1Gymnasium-v0 OfflinePointButton1Gymnasium-v0
```

Run detached:

```bash
nohup ./run.sh > run.out 2>&1 &
```

## 5. Output locations

- training artifacts: `logs/<task>/<algo>-seed<seed>/`
- stdout/stderr captures: `logs/stdout/<task>/<algo>-seed<seed>.log`

## 6. Host-level note

The conda env handles the Python stack. If the server image is missing standard OpenGL or MuJoCo runtime libraries, `safety-gymnasium` may still fail at import or env creation time. If that happens, you will need the server's base image or modules adjusted; the Python environment here is already set up correctly for OSRL itself.
