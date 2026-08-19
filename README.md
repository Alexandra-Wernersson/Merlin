# Merlin

Simulation-based inference (SBI) pipeline for Euclid 3x2pt cosmology. Trains a
neural network (via `swyft`'s neural ratio estimation) to infer posteriors on
cosmological + nuisance parameters from weak-lensing / galaxy-clustering
angular power spectra ($C_\ell$), instead of running a traditional MCMC
likelihood analysis.

Built on `swyft`, `cloelib` (Euclid cosmology library), `euclidlib`,
`pytorch-lightning`, `getdist`.

## Installation

### 1. Install `cloelib` first
`cloelib` must be installed manually from source before installing Merlin:
```
git clone https://github.com/cloe-org/cloelib.git
pip install ./cloelib          # or `pip install -e ./cloelib` for an editable install
```

### 2. Install the remaining dependencies and Merlin
The rest of the dependencies (`numpy`, `torch`, `scipy`, `joblib`, `psutil`,
`matplotlib`, `zarr`, `swyft`, `pytorch-lightning`, `euclidlib`, `getdist`,
`pandas`) are installed automatically:
```
pip install .          # or `pip install -e .` for an editable install
```

This also registers the `merlin-simulate`, `merlin-train`, and `merlin-plot`
console scripts (see `pyproject.toml`).

## Repository structure

```
merlin/
├── config.py            load_config; RUN.run_dir / train_<N> path derivation
├── params.py             canonical parameter registry (PARAMS, PARAM_GROUPS,
│                         PARAM_LABELS, MCMC_KEY_MAP)
├── priors.py              PRIORS section parsing/validation, Fisher-bounds overlay
├── tracers.py              cloelib/euclidlib tracer + n(z) loading
├── simulator.py             swyft Simulator, PriorSampler, build_simulator factory,
│                            sample_correlated_noise
├── fisher.py                 Fisher matrix (finite differences)
├── observation.py             fiducial observation generation + Cholesky whitening
├── simulate.py                 fills the Zarr simulation store (parallel joblib)
├── preprocessing.py             Cholesky whitening, scale cuts, probe selection, PCA
├── network.py                    Network architecture (swyft.SwyftModule)
├── train.py                       training loop
├── inference.py                    infer / predict_from_checkpoint
├── coverage.py                      coverage/calibration test
├── plotting.py                       corner/coverage/loss plots (merlin-plot backend)
├── swyft_patches.py                   small monkeypatches to swyft's corner-plot internals
├── io.py                               misc array/pickle IO helpers
└── cli/                                 merlin-simulate / merlin-train / merlin-plot entry points

notebooks/
├── pipeline_walkthrough.ipynb   interactive tour: config → simulator → observation →
│                                 store preprocessing → PCA → training → coverage
└── corner_plot.ipynb             load a trained checkpoint, plot a corner plot
                                    (optionally against an MCMC/nested-sampling chain)

input/
└── config_example.yaml           template config — copy and edit this

aux_files/                        input covariance matrix / n(z) FITS file (not run-generated)

jobs/                              example SLURM batch scripts (see "Running on a cluster" below)
├── submit_simulate.sh              merlin-simulate — CPU-only
└── submit_train.sh                 merlin-train — 1 GPU
```

## Usage

### Step 1: Setup the config file
Copy `input/config_example.yaml` and edit it — see "Config reference" below
for what each section does. `RUN.run_dir` is the only path you need to set by
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
(per `MOCK_OBS`/`FIDUCIAL VALUES`), preprocesses the store (Cholesky
whitening, scale cuts, probe selection, PCA — projection saved to
`train_<N>/aux_files/SVD.npy`), trains the network, and runs inference on
that observation. Since preprocessing/the observation are both regenerated
fresh on every run, you can change the network architecture, `ANALYSIS
VARIANTS`, or `MOCK_OBS`/`FIDUCIAL VALUES` and re-run `merlin-train` to get a
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

## Config reference

See `input/config_example.yaml` for a fully worked example with inline
comments; this is a brief summary of what each section controls.

- **`RUN`** — `run_dir`: the only path you set by hand (see "Output directory
  layout" above).
- **`FIDUCIAL VALUES`** — the 50 cosmological + nuisance parameters, keyed by
  canonical name (`params.PARAMS`) — matched by name, not position, so order
  doesn't matter. Used as the fiducial cosmology for the mock observation and
  Fisher analysis.
- **`PRIORS`** — per-parameter prior specification, one entry per name in
  `FIDUCIAL VALUES`: `{type: fixed}` (held at its fiducial value, never
  sampled), `{type: uniform, lower, upper}`, or `{type: normal, mean, sigma}`.
  `use_Fisher_priors: true` runs a Fisher analysis and overwrites every
  `uniform` entry's bounds with `fiducial ± sigma_scale·σ` (`normal` entries
  are left as configured). `covmat_Fisher`/`nz_Fisher` are used only for that
  Fisher computation, independent of `AUX FILES.covmat`/`nz`.
- **`MOCK_OBS`** — where the fiducial observation comes from.
  `generate_from_fiducial: true` (default) simulates it from the fiducial
  cosmology; `add_noise_fid_obs` controls whether a noise realization is
  folded in. `generate_from_fiducial: false` instead loads an already-noisy
  data vector from the `.npy` array at `path` (e.g. real data).
- **`AUX FILES`** — `covmat`/`nz`/`ell`/`Nbin_z`: the physics inputs used for
  simulation, observation generation, and preprocessing (everything except
  the Fisher computation itself, which uses `PRIORS.covmat_Fisher`/`nz_Fisher`
  instead). `ell` accepts either a file path or an inline list.
- **`SIMULATION`** — `N_sims`, `batch_size`, `chunk_size`, `n_workers` for
  `merlin-simulate`'s parallel store-filling loop.
- **`ANALYSIS VARIANTS`** — modifications applied at *preprocessing* time
  (`merlin-train`/`merlin-plot`), never baked into the store, so different
  `train_<N>` runs can vary them independently without re-simulating:
  - `SCALE CUTS` (`SHE_SHE`/`POS_SHE`/`POS_POS`) — per-probe ℓ_max cutoffs;
    zeroes out ell bins above the threshold.
  - `regenerate_noise_samples` — if `true`, discards the store's own noise
    samples and draws fresh ones from the *current* `AUX FILES.covmat`
    instead (e.g. to retrain against a different noise covariance than the
    one active at simulation time, without re-running the expensive physics).
  - `train_on_data` (`3x2pt`/`2x2pt`/`WL`) — which probe block(s) of the data
    vector to train on; unlike `SCALE CUTS`, this actually shrinks the data
    fed into PCA/the network rather than masking it. `2x2pt` here means
    GGL+GCph (3x2pt minus WL), this project's own convention.
- **`PCA`** — `recompute_pca`, `q` (max components), `variance_cut` (percent
  variance threshold for keeping a component).
- **`NETWORK`** — architecture: `hidden_feat_compress` (compression-MLP
  hidden-layer sizes), `num_feat_param`, `num_blocks_ratios`,
  `hidden_feat_ratios`, `dropout_ratios` (swyft ratio-estimator settings).
- **`TRAINING`** — `learning_rate`, `batch_size`, `num_workers`, `max_epochs`,
  `early_stopping` (patience), `val_fraction`, `accelerator` (`auto`/`gpu`/
  `cpu`), and `params_to_infer` (a `PARAM_GROUPS` name like `COSMO`, or an
  explicit list) — which parameters the network is trained to produce
  posteriors for; must be a subset of the parameters varied in `PRIORS`.
- **`PLOTTING`** — `mcmc_path`: optional Nautilus-format nested-sampling
  chain `.npz` to overlay on `merlin-plot --mode corner` / `corner_plot.ipynb`
  plots. `null` (default) skips it. `smooth_swyft`/`nbins_swyft`: swyft
  smoothing and bin count for Merlin's corner-plot density estimate.
