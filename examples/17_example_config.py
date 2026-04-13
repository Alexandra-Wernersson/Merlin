[FIDUCIAL VALUES]
H0 = 67.0
omega_b = 2.2445
omega_cdm = 0.1206
n_s = 0.96
ln10^{10}a_s = 3.0568 
a_ia = 1.72
eta_ia = -0.41
b_1 = 1.09977
b_2 = 1.22025
b_3 = 1.2724
b_4 = 1.31662
b_5 = 1.35812
b_6 = 1.39982
b_7 = 1.44465
b_8 = 1.4965
b_9 = 1.56525
b_10 = 1.74299

[PRIORS SCALES]
H0 = 60, 15
#omega_b = 2.0, 2.5 
omega_b = 0.8, 3.5
omega_cdm = 0.1, 0.04
n_s = 0.91, 0.12
ln10^{10}a_s = 2.8, 0.5
a_ia = 1.55, 0.34
eta_ia = -0.45, 0.08
b_1 = 0.9, 0.3
b_2 = 1.1, 0.3
b_3 = 1.15, 0.3
b_4 = 1.2, 0.3
b_5 = 1.25, 0.3
b_6 = 1.3, 0.3
b_7 = 1.35, 0.3
b_8 = 1.4, 0.3
b_9 = 1.4, 0.3
b_10 = 1.6, 0.3 

[FINV]
N_pars = 17
N_nuisance = 12
finv_file = /home/awernersson/projects/Euclid_Swyft/Fisher/Finv_LCDM.npy
#Make fisher automatically, the first line, 

[AUX FILES]
SVD = ./Aux_files/SVD_LCDM.npy
OBS = ./Aux_files/obs_LCDM.npy
LFID = ./Aux_files/cholesky_LCDM.npy

[STORES]
checkpoint_path = /gpfs/scratch1/shared/awernersson/Euclid_Swyft/checkpoint_paths/train_cosmo_test_2work
predictions = /gpfs/scratch1/shared/awernersson/Euclid_Swyft/predictions/predictions_cosmo_test_2work.pkl
predictions_nuisance = /gpfs/scratch1/shared/awernersson/Euclid_Swyft/predictions/predictions_nuisance_test.pkl

[ZARR PARAMS]
run_id = Euclid_Swyft_1
use_zarr = True
n_sims = 50_000
chunk_size = 1000
run_parallel = True
njobs = 16 #72
#targets = z_int,z_ext,z_total,d_t,d_f,d_f_w,n_t,n_f,n_f_w
store_path = /gpfs/scratch1/shared/awernersson/Euclid_Swyft/store_euclid_pipeline9Aug_work
#run_description = 15D test for precessing BBH with high spin and low S/N

[HYPERPARAMS]
min_epochs = 1
max_epochs = 100
early_stopping = 7
learning_rate = 5e-4
num_workers = 8
training_batch_size = 64
validation_batch_size = 64
simulation_batch_size = 8
max_sims = 8
train_data = 0.9
val_data = 0.1

[DEVICE PARAMS]
device = gpu
n_devices = 1
