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
pip install -r requirements-server.txt
pip install -e . --no-deps
```

This setup intentionally installs PyTorch from the official CUDA 11.7 pip wheels rather than conda. That avoids the `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` MKL issue you hit with conda-based PyTorch.

`swig` is not included in the conda env because it is not required for the current pip-based install path. If a future package install explicitly asks for it on your server image, install it separately with your system package manager or a one-off `conda install -c conda-forge swig`.

If your existing env is broken, rebuild it cleanly:

```bash
conda deactivate
conda env remove -n osrl -y
conda env create -f environment.yml
conda activate osrl
pip install -r requirements-server.txt
pip install -e . --no-deps
```

Validate the install before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

If training fails immediately with `ModuleNotFoundError: No module named 'pkg_resources'`, you likely have `setuptools>=82`, which removed `pkg_resources`. Downgrade it and rerun:

```bash
pip install --force-reinstall "setuptools<82"
```

If you update the environment later, refresh with:

```bash
conda env remove -n osrl -y
conda env create -f environment.yml
conda activate osrl
pip install -r requirements-server.txt
pip install -e . --no-deps
```

## 2. Optional environment variables

If you want the DSRL datasets on scratch storage instead of the default `~/.dsrl/datasets`:

```bash
export DSRL_DATASET_DIR=/path/to/shared-or-scratch-storage/dsrl
mkdir -p "$DSRL_DATASET_DIR"
```

If you ever want to disable online syncing for a specific run, set:

```bash
export WANDB_MODE=offline
```

## 3. Launch the default 6 runs

```bash
chmod +x run.sh
./run.sh
```

That launches:

- `train_cpq.py` on `OfflineCarButton1Gymnasium-v0` for seeds `0,1,2`
- `train_coptidice.py` on `OfflineCarButton1Gymnasium-v0` for seeds `0,1,2`

The script first verifies CUDA, then downloads the DSRL dataset once, then runs one training job at a time in the foreground. Output is streamed to the terminal and written to the matching log file with `tee`. If a run fails, the script stops immediately on that run.

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
