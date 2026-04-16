# Merlin
A Merlin project (Name is subject to change)

## Dependencies
Requires: `swyft`, `euclidlib`, `cloelib`, `pytorch-lightning`, `getdist`

## Step 1: Setup the config file
Go to `examples/new_config.ini` and set your fiducial values, store path, file paths, etc.
Use `examples/example_config.ini` as a reference template.

## Step 2: Generate Observation
Run (in terminal)
```
python generate_observation.py examples/new_config.ini
```
Saves the following to the paths defined in `[OBSERVATION]`:
- `OBS` — raw fiducial observation (C_ells, noise, z)
- `OBS_CHOLESKY_NOISELESS` — Cholesky-whitened noiseless observation (used for inference)
- `LFID` — Cholesky factor of the covariance at the fiducial

If `run_fisher = True` in `[FINV]`, also runs a Fisher analysis to generate the inverse Fisher matrix used for prior bounds.

## Step 3: Generate simulations
Run (in a sbatch script)
```
python simulate.py examples/new_config.ini
```
Generates simulations and saves them to a Zarr store at `[SIMULATION] store_path`.

## Step 4: Train the network and store the predictions
Run
```
python training.py examples/new_config.ini
```
Trains the network on the simulations and runs inference on the observation.
The best model checkpoint is saved to `[STORES] checkpoint_path`.
Predictions are saved to `[STORES] predictions_cosmo`.

## Step 5: Plot the predictions
Open `Merlin_results.ipynb` to visualize the simulator output and the network predictions.

## To Do

* Make training into one file or something you can choose in the config file so one can run `python train.py examples/new_config.ini cosmo` or `nuisance`
* Add the heavy 65 parameter case
* Make the whole pipeline run at once
* and more....
