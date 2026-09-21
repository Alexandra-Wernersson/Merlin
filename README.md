![Merlin logo](merlin_logo.png)
<sub>*Logo designed by Sofía Franco Oñate*</sub>

Merlin is a simulation-based inference (SBI) package
to perform cosmological + nuisance parameter estimation with 3x2pt angular power spectrum measurements from Stage IV surveys. It is built on top of the [swyft](https://github.com/undark-lab/swyft) code to perform Marginal Neural Ratio Estimation (MNRE) and the [cloelib](https://github.com/cloe-org/cloelib) library for the 3x2pt theory predictions. 

## Installation

The package requires Python ≥ 3.9. All dependencies are declared in `pyproject.toml` and installed automatically, except for `cloelib` (see below). A GPU is recommended for training but not required.

### 1. Install `cloelib` first
`cloelib` must be installed manually from source before installing Merlin:
```bash
git clone https://github.com/cloe-org/cloelib.git
pip install ./cloelib          # or `pip install -e ./cloelib` for an editable install
```

### 2. Install the remaining dependencies and Merlin
The rest of the dependencies (`numpy`, `torch`, `scipy`, `joblib`, `psutil`,
`matplotlib`, `zarr`, `swyft`, `pytorch-lightning`, `euclidlib`, `getdist`,
`pandas`) are installed automatically:
```bash
git clone https://github.com/Alexandra-Wernersson/merlin.git
cd merlin
pip install .          # or `pip install -e .` for an editable install
```

This also registers the `merlin-simulate`, `merlin-train`, and `merlin-plot`
console scripts (see `pyproject.toml`).

## Usage

### Step 1: Setup the config file
Copy `input/config_example.yaml` and edit it — its inline comments explain
what each section does. `RUN.run_dir` is the only path you need to set by
hand; everything else is auto-derived under it (see "Output directory
layout").

### Step 2: Generate simulations
```
merlin-simulate my_config.yaml
```
If `PRIORS.use_Fisher_priors: true`, runs a Fisher analysis first (a no-op
otherwise) and saves the inverse Fisher matrix to `RUN.run_dir/finv.npz`, used
to set the bounds of `uniform`-type priors — Fisher only ever feeds prior
bounds for this simulation step, so it's run here rather than as a separate
command. Then generates simulations and saves them to a Zarr store at
`RUN.run_dir/store`. For a large `N_sims`, submit this as a batch job (see
"Running on a cluster" below) rather than running it interactively. Drops a
copy of the config used at `RUN.run_dir/config.yaml`.

### Step 3: Train the network and run inference
```
merlin-train my_config.yaml
```
Creates the next `RUN.run_dir/train_<N>/` (`train_1`, `train_2`, ... — never
reuses or overwrites one) and drops a copy of the config used at
`train_<N>/config.yaml`. Generates the fiducial observation fresh into it
(per `MOCK_OBS`/`FIDUCIAL`), preprocesses the store (Cholesky
whitening, scale cuts, probe selection, PCA — projection saved to
`train_<N>/aux_files/SVD.npy`), trains the network, and runs inference on
that observation. Since preprocessing/the observation are both regenerated
fresh on every run, you can change the network architecture,
`ANALYSIS_VARIANTS`, or `MOCK_OBS`/`FIDUCIAL` and re-run `merlin-train` to get a
new `train_<N>` against the *same* simulations — no need to re-run
`merlin-simulate`. The best checkpoint (by validation loss) is saved to
`train_<N>/best.ckpt` — no predictions are saved to disk; reload
the checkpoint later to get predictions without retraining.

### Step 4: Plot
```
merlin-plot my_config.yaml --mode corner   --train-id N   # triangle plot of params_to_infer, vs. mock obs
merlin-plot my_config.yaml --mode coverage --train-id N   # coverage/calibration test
merlin-plot my_config.yaml --mode loss     --train-id N   # train/val loss vs. epoch, from the csv logs
```
`--train-id` (always required — a deliberate choice, so you never
accidentally plot a stale run) picks which `train_<N>/` to plot. Each mode
loads that run's checkpoint (`corner`/`coverage`) or csv logs (`loss`) and
saves a PDF to `train_<N>/plots/<mode>.pdf`. Meant for a quick look, not a
cluster job. Set `PLOTTING.mcmc_path` in the config to overlay a
nested-sampling (e.g. Nautilus) chain on the `corner` plot — `null` (the
default) skips it.

For an interactive corner plot with additional observation diagnostics, open
`notebooks/corner_plot.ipynb` instead (set `TRAIN_ID` there the same way).
See `notebooks/pipeline_walkthrough.ipynb` for an interactive, step-by-step
tour of the rest of the pipeline.

## Running on a cluster

`jobs/` has example SLURM batch scripts for Snellius-like clusters, covering
Steps 2 and 3 above (Step 4, `merlin-plot`, is meant for a quick interactive
look, not a batch job):
```
sbatch jobs/submit_simulate.sh [path/to/config.yaml]   # CPU-only, --cpus-per-task should match SIMULATION.n_workers
sbatch jobs/submit_train.sh    [path/to/config.yaml]   # 1 GPU
```
Both default to `input/config_example.yaml` if no config is given. These are
examples tuned for Snellius specifically — on a different cluster you'll
likely need to adapt the `#SBATCH` preamble too (partition names, `--gpus`
syntax, per-node CPU/GPU counts, etc.), not just the environment-activation
lines (`module load`/`conda activate`) near the bottom, which you'll almost
certainly need to change to however you actually activate merlin's
environment. Each script's header comments explain its resource choices in
more detail.

## Output directory layout

Two scopes, populated at different times: `RUN.run_dir` itself holds what's
*shared* across every training run (written once by `merlin-simulate`);
`train_<N>/` holds what's specific to one `merlin-train` run — kept separate
because retraining can vary independently of the simulations (architecture,
scale cuts, even a different observation to evaluate against).

```
<run_dir>/
├── config.yaml                    copy of the config used by merlin-simulate
├── store/                         the Zarr simulation store
├── finv.npz                       inverse Fisher matrix (only if PRIORS.use_Fisher_priors)
├── train_1/                       first merlin-train run against this store
│   ├── config.yaml                 copy of the config used for THIS run
│   ├── best.ckpt                   trained network — see inference.predict_from_checkpoint
│   ├── metrics.csv                 per-step/epoch train_loss/val_loss
│   ├── plots/{corner,coverage,loss}.pdf   written by `merlin-plot --train-id 1`
│   ├── mock_obs/
│   │   ├── obs.npy                 raw fiducial observation
│   │   └── obs_cholesky.npy        Cholesky-whitened observation used for inference
│   └── aux_files/
│       ├── Lfid.npy                Cholesky factor of the fiducial covariance
│       └── SVD.npy                 PCA projection matrix
├── train_2/                       a second run — different architecture, scale
│   └── ...                         cuts, or observation, same simulations
└── train_<N>/
    └── ...
```

Predictions are never saved to disk — they're cheap to recompute from a
saved checkpoint whenever needed (`predict_from_checkpoint`).
