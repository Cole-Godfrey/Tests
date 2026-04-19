# Linux Server Setup

This repo now includes a Linux/CUDA conda environment file and a `run.sh` launcher that defaults to:

- `OfflineMetadrive-easymean-v0`
- `OfflineMetadrive-mediumsparse-v0`
- `OfflineAntRun-v0`
- algorithms: `CPQ`, `COptiDICE`, and `BC-Safe`
- seeds: `0 1 2`
- `1,000,000` update steps per run
- `20` evaluation episodes per checkpoint
- FISOR paper cost limits:
  `10` for `Safety-Gymnasium`, `5` for `Bullet-Safety-Gym` and `MetaDrive`
- one training job at a time in the foreground

## 1. Create the conda environment

From the repo root on the Linux server:

```bash
conda env create -f environment.yml
conda activate osrl
pip install -r requirements-server.txt
./scripts/install_metadrive_compat.sh
pip install -e . --no-deps
```

This setup intentionally installs PyTorch from the official CUDA 11.7 pip wheels rather than conda. That avoids the `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` MKL issue you hit with conda-based PyTorch.

`swig` is not included in the conda env because it is not required for the current pip-based install path. If a future package install explicitly asks for it on your server image, install it separately with your system package manager or a one-off `conda install -c conda-forge swig`.

`OfflineMetadrive-*` tasks such as `easymean` and `mediumsparse` require MetaDrive. The provided installer script patches the upstream `metadrive_clean` dependency pins so it can install on Python 3.10/Linux, where the original `panda3d==1.10.8` pin is no longer available. It also force-reinstalls a compatible `numpy/scipy/numba` stack so MetaDrive does not upgrade OSRL out from under you.

If your existing env is broken, rebuild it cleanly:

```bash
conda deactivate
conda env remove -n osrl -y
conda env create -f environment.yml
conda activate osrl
pip install -r requirements-server.txt
./scripts/install_metadrive_compat.sh
pip install -e . --no-deps
```

Validate the install before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import matplotlib, wandb, pkg_resources; print('python deps ok')"
python -c "import metadrive; print('metadrive ok')"
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
./scripts/install_metadrive_compat.sh
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

## 3. Launch the default 27 runs

```bash
chmod +x run.sh
./run.sh
```

That launches:

- `CPQ`, `COptiDICE`, and `BC-Safe`
- on `OfflineMetadrive-easymean-v0`, `OfflineMetadrive-mediumsparse-v0`, and `OfflineAntRun-v0`
- for seeds `0,1,2`
- with `1,000,000` update steps per run
- with FISOR paper-style evaluation (`20` episodes, task-family-specific cost limits)

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

Override the default update budget:

```bash
UPDATE_STEPS=200000 ./run.sh
```

Skip already-finished runs explicitly:

```bash
SKIP_RUNS="OfflineMetadrive-easymean-v0:cpq:0" ./run.sh
```

Run a custom task list:

```bash
./run.sh OfflineMetadrive-easymean-v0 OfflineAntRun-v0
```

Run detached:

```bash
nohup ./run.sh > run.out 2>&1 &
```

Evaluate finished checkpoints with the FISOR paper protocol:

```bash
chmod +x fisor_eval.sh
./fisor_eval.sh
```

That re-scores completed checkpoints with `20` eval episodes and the published FISOR cost limit for each task family, then prints per-seed and averaged summaries across the available seeds.

## 5. Output locations

- training artifacts: `logs/<task>/<algo>-seed<seed>/`
- stdout/stderr captures: `logs/stdout/<task>/<algo>-seed<seed>.log`

## 6. Host-level note

The conda env handles the Python stack. If the server image is missing standard OpenGL or MuJoCo runtime libraries, `safety-gymnasium` may still fail at import or env creation time. If that happens, you will need the server's base image or modules adjusted; the Python environment here is already set up correctly for OSRL itself.
