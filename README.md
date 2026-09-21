![Merlin logo](merlin_logo.png)
<sub>*Logo designed by Sofía Franco Oñate*</sub>

Merlin is a simulation-based inference (SBI) package
for cosmological + nuisance parameter estimation with 3x2pt measurements from Stage IV surveys. It is built on top of the [swyft](https://github.com/undark-lab/swyft) code to perform Marginal Neural Ratio Estimation (MNRE) and the [cloelib](https://github.com/cloe-org/cloelib) library for the 3x2pt theory predictions. 

## Installation

The package requires Python ≥ 3.9. All dependencies are declared in `pyproject.toml` and installed automatically, except for `cloelib` (see below). A GPU is recommended for training but not required.

#### 1. Install cloelib first
`cloelib` must be installed manually from source before installing Merlin:
```bash
git clone https://github.com/cloe-org/cloelib.git
pip install ./cloelib          # or `pip install -e ./cloelib` for an editable install
```

#### 2. Install the remaining dependencies and Merlin
The rest of the dependencies (`numpy`, `torch`, `scipy`, `joblib`, `psutil`,
`matplotlib`, `zarr`, `swyft`, `pytorch-lightning`, `euclidlib`, `getdist`,
`pandas`) are installed automatically:
```bash
git clone https://github.com/Alexandra-Wernersson/merlin.git
cd merlin
pip install .          # or `pip install -e .` for an editable install
```

This also registers the `merlin-simulate`, `merlin-train`, and `merlin-plot`
console scripts.

## Usage

#### Step 1: Setup the config file
Copy `input/config_example.yaml` and edit it, its comments explain what each section does. `RUN.run_dir` is the only path that needs to be set by hand; everything else is derived from it (see "Output directory layout").

#### Step 2: Generate simulations
Generate simulations with the following command:
```
merlin-simulate my_config.yaml
```
If `PRIORS.use_Fisher_priors: true`, this runs a Fisher analysis first and saves the inverse Fisher matrix to `RUN.run_dir/finv.npz`, which is used to restrict the prior bounds. Then it generates simulations and saves them to a Zarr store at `RUN.run_dir/store`. If `CLOELIB_SETTINGS.restrict_prior_for_derived: true`, a rejection sampling is applied to the cosmological parameters in order to zoom around the derived-parameter (e.g. `sigma8`) region. Finally, it saves a copy of the config used for simulation at `RUN.run_dir/config.yaml`.

#### Step 3: Train the network and run inference
Once simulations are generated, you can train via:
```
merlin-train my_config.yaml
```
This creates the next `RUN.run_dir/train_<N>/` (`train_1`, `train_2`) and saves a copy of the config used for training at `train_<N>/config.yaml`. In particular, it generates the mock observation from the fiducial, preprocesses the store (Cholesky whitening, PCA — projection, scale cuts, probe selection, noise regeneration), trains the network, and runs inference on that observation. Because preprocessing happens fresh every time you call `merlin-train`, changing the network architecture or `ANALYSIS_VARIANTS` doesn't require new simulations — just re-run `merlin-train` to get a new `train_<N>` from the *same* store. 

#### Step 4: Plot
If you already have a trained run, you can go straight to plotting:
```
merlin-plot my_config.yaml --mode corner   --train-id N   # triangle plot of params_to_infer
merlin-plot my_config.yaml --mode coverage --train-id N   # coverage test
merlin-plot my_config.yaml --mode loss     --train-id N   # train/val loss vs. epoch, from the csv logs
```
Here `--train-id` picks which `train_<N>/` to plot. Each mode loads that run's checkpoint or csv logs and saves a PDF to `train_<N>/plots/<mode>.pdf` (`corner_eval_fiducial.pdf` instead if `PLOTTING.eval_fiducial` is used). Other useful flags for plotting: `--smooth`/`--bins` (corner), `--cols` (coverage), `--fiducial-override` (corner, evaluate at a different fiducial). Set `PLOTTING.mcmc_path` in the config to overlay a nested-sampling run on the `corner` plot.

For an interactive corner plot with additional observation diagnostics, open `notebooks/corner_plot.ipynb` instead. See `notebooks/pipeline_walkthrough.ipynb` for an interactive, step-by-step tour of the rest of the pipeline.

## Running on a cluster

`jobs/` has example SLURM batch scripts, covering Steps 2 and 3 above (Step 4, `merlin-plot`, is meant for a quick interactive look):
```
sbatch jobs/submit_simulate.sh [path/to/config.yaml]   # CPU-only, --cpus-per-task should match SIMULATION.n_workers
sbatch jobs/submit_train.sh    [path/to/config.yaml]   # 1 GPU
```
You will likely need to adapt the #SBATCH headers and module/environment setup to your cluster.

## Output directory layout

The path `RUN.run_dir` holds what's *shared* across every training run (written once by `merlin-simulate`), while the sub-folders
`train_<N>/` hold what's specific to one `merlin-train` run. These are kept separate because retraining can vary independently of 
the simulations (e.g. different architecture, multiple analysis variants, or even a different observation to evaluate against).

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

Predictions are never saved to disk, they're cheap to recompute from a
saved checkpoint whenever needed (`predict_from_checkpoint`).
