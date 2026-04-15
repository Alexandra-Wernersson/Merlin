# Merlin
A Melin project (Come up with a nice name)

# Step 1: Setup the config file
Go to ```examples/example_config.ini``` and set your fiducial values, store path etc.

# Step 2: Generate Observation
Run (in terminal)
```python generate_observation.py examples/example_config.ini```
to generate obs, Lfid, Fisher if specified.

# Step 3: Generate simulations
Run (in a sbatch script)
```python simulate.py examples/example_config.ini```
to generate simulations.

# Step 4: Train the network and store the predictions
Run
```python train_cosmo.py examples/example_config.ini```
To train the network and store the predictions for the cosmological parameters.


# Step 5: Plot the predictions
Look through ```merlin_results.ipynb``` to vizualize the simulator output and the predictions.

# To Do

* Make training into one file or something you can choose in the config file so one can run ```python train.py examples/example_config.ini 'cosmo'/'nuisance'``` or similar.
* Create plotting notebook and plotting script.
* Add the heavy 65 parameter case
* and more....
