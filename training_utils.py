import os
import pickle
import swyft  #the version used is 0.4.6.
import torch
import numpy as np
import pytorch_lightning as pl
from scipy import stats
from scipy.linalg import solve_triangular

#V_proj = np.load('./SVD_3x2pt_LCDM_100k_merlin.npy',allow_pickle=True)
#lenV = V_proj.shape[1]  # number of PCA components
#print(f"Kept {lenV} PCA components")

class Network(swyft.AdamWReduceLROnPlateau,swyft.SwyftModule):

    def __init__(self, V_proj):
        super().__init__()
        self.learning_rate = 1e-3
        self.batch_size = 256
        self.num_params_show = 5 # number of parameters of interest, obviously cannot be larger than N_pars
        self.num_feat_param = 4 # number of features per parameter of interest (2 seem to work well)
        self.marginals  = self.get_marginals(self.num_params_show)
        #self.norm =  swyft.networks.OnlineStandardizingLayer(torch.Size([lenV]), epsilon=1e-50)
#        self.V_proj = V_proj
#        self.V_proj = torch.tensor(V_proj, dtype=torch.float32)
        #self.register_buffer("V_proj", torch.tensor(V_proj, dtype=torch.float64))
        # ---- load PCA ----
        lenV = V_proj.shape[1]

        print(f"Using {lenV} PCA components")

        self.norm = swyft.networks.OnlineStandardizingLayer(

            torch.Size([lenV]),

            epsilon = 1e-50
        )

        # ensure writable tensor
        V_proj = np.array(V_proj, copy=True)

        self.register_buffer(

            "V_proj",

            torch.tensor(
                V_proj,
                dtype=torch.float32
            )
        )   

        #compression network, here one could play with number of layers and neurons
        # self.sequential = torch.nn.Sequential(
        #    torch.nn.LazyLinear(256),
        #    torch.nn.ReLU(),
        #    torch.nn.LazyLinear(self.num_params_show*self.num_feat_param),
        #    torch.nn.LazyBatchNorm1d()
        # )
        self.sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(self.num_params_show * self.num_feat_param),
            torch.nn.LazyBatchNorm1d()
        )
        #pre-defined MLP to get 1-dim marginals
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = self.num_feat_param,
                                                       num_params = self.num_params_show, varnames = 'z',
                                                       num_blocks = 4, dropout = 0.1)
        #pre-defined MLP to get 2-dim marginals
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 2*self.num_feat_param,
                                                       marginals = self.marginals, varnames = 'z',
                                                       num_blocks = 4, dropout = 0.1)
  
    @staticmethod
    def get_marginals(n_params): #get all possible marginal combinations given number of parameters
        marginals = []
        for i in range(n_params):
            for j in range(n_params):
                if j>i: marginals.append((i, j))
        return tuple(marginals)

    def forward(self, A, B):
        Cells_noisy = A['C_ells'] + A['noise'] #add noise to the spectra
#        s = self.V_proj.T@Cells_noisy
        s = Cells_noisy @ self.V_proj
        s = self.norm(s)
        s = self.sequential(s)

        z = B['z'][..., :self.num_params_show]
        s1 = s.reshape(-1, self.num_params_show, self.num_feat_param)
        s2 = torch.stack([torch.cat([s1[:, i, :], s1[:, j, :]], dim=-1) for i, j in self.marginals], dim=1)

        logratios1 = self.logratios1(s1, z)
        logratios2 = self.logratios2(s2, z)
        
        return logratios1, logratios2

def make_resampler_new(store_samples, N_sims, N_spectra, Nbin_ell):
    def resampler(x):
        i = np.random.randint(N_sims, size = N_spectra) # we mix the noise of different spectra across different realizations
        blocks = [store_samples['noise'][i[k], k*Nbin_ell:(k+1)*Nbin_ell] for k in range(N_spectra)]
        x['noise'] = np.concatenate(blocks)
        return x
    return resampler

def make_resampler(store_samples, N_sims, N_spectra, Nbin_ell):

    noise_blocks = store_samples['noise'].reshape(N_sims, N_spectra, Nbin_ell)

    def resampler(x):
        i = np.random.randint(0, N_sims, size=N_spectra)
        sampled = noise_blocks[i, np.arange(N_spectra), :]
        x['noise'] = sampled.reshape(-1)
        return x

    return resampler

def load_or_compute_pca(
    Cells_chol,
    config
):
    pca_cfg = config["PCA"]
    store_PCA = pca_cfg["store_pca"]
    pca_file = pca_cfg["SVD"]
    q = int(pca_cfg["q"])
    var_cut = float(pca_cfg["variance_cut"])


    # --------------------------------------------------

    if store_PCA:

        print("\nRunning PCA compression...")

        fCells_chol = torch.from_numpy(
            Cells_chol
        )

        U, S, V = torch.pca_lowrank(
            fCells_chol,
            q = q,
            center = True
        )

        Stot = torch.sum(S)

        Sn = (S / Stot) * 100

        # keep only modes above variance threshold
        ind = Sn > var_cut

        V_proj = V[:, ind]

        np.save(
            pca_file,
            V_proj.numpy(),
            allow_pickle = True
        )

        print(f"PCA stored to: {pca_file}")


    # --------------------------------------------------

    V_proj = np.load(
        pca_file,
        allow_pickle = True
    )

    lenV = V_proj.shape[1]
    print(f"Kept {lenV} PCA components")

    return V_proj

def compute_cholesky_projections(store, Lfid, ell_lims):
     """
     Apply inverse Cholesky rotation to C_ells and noise.

     Parameters:
         store: Swyft SampleStore or ZarrStore with ['C_ells', 'noise']
         Lfid: dict of Cholesky matrices, indexed by ell (as strings)
         ell_lims: np.array of ell bin edges, shape (N_bins + 1,)

     Returns:
         Cells_chol, noise_chol: shape (N_sims, N_spectra, N_bins)
     """
     N_sims = len(store['z'])
     N_spectra, N_bins = store['C_ells'].shape[1], len(ell_lims) - 1

     ells = 0.5 * (ell_lims[:-1] + ell_lims[1:])

     Cells_chol = np.zeros((N_sims, N_spectra, N_bins))
     noise_chol = np.zeros((N_sims, N_spectra, N_bins))
     #Lfid = Lfid.item()
     #store = store.item()
     for ind, ell in enumerate(ells):
         ell_key = str(int(round(float(ell))))
         Linv = np.linalg.inv(Lfid[ell_key])
         Cells_chol[..., ind] = np.matmul(Linv, store['C_ells'][..., ind].T).T
         noise_chol[..., ind] = np.matmul(Linv, store['noise'][..., ind].T).T

     return Cells_chol, noise_chol

# utils.py

class old_Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    def __init__(self, Vh_proj):
        super().__init__()
        self.learning_rate = 1e-3
        self.batch_size = 256
        self.num_params_show = 5
        self.num_feat_param = 2
        self.marginals = self.get_marginals(self.num_params_show)
        self.flatten = torch.nn.Flatten()
        self.register_buffer('Vh_proj', torch.as_tensor(Vh_proj, dtype=torch.float32))
        self.lenV = Vh_proj.shape[0]
        self.norm = swyft.networks.OnlineStandardizingLayer(torch.Size([self.lenV]), epsilon=1e-50)
        self.sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(self.num_params_show * self.num_feat_param),
            torch.nn.LazyBatchNorm1d()
        )
        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=self.num_feat_param,
            num_params=self.num_params_show,
            varnames='z',
            num_blocks=4,
            dropout=0.1
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=2 * self.num_feat_param,
            marginals=self.marginals,
            varnames='z',
            num_blocks=4,
            dropout=0.1
        )

    @staticmethod
    def get_marginals(n_params):
        marginals = []
        for i in range(n_params):
            for j in range(n_params):
                if j > i:
                    marginals.append((i, j))
        return tuple(marginals)

    def forward(self, A, B):
        C_ells_noisy_chol = A['C_ells'] + A['noise']
        fC_ells_noisy_chol = self.flatten(C_ells_noisy_chol)
        if not torch.is_tensor(fC_ells_noisy_chol):
            fC_ells_noisy_chol = torch.as_tensor(fC_ells_noisy_chol)
    
        fC_ells_noisy_chol = fC_ells_noisy_chol.to(dtype=self.Vh_proj.dtype, device=self.Vh_proj.device)
    
        # Project onto PCA basis
        s = (self.Vh_proj @ fC_ells_noisy_chol.T).T
        s = self.norm(s)
        s = self.sequential(s)
    
        # Get batch size from processed features
        batch_size = s.shape[0]
    
        # Handle the z tensor - FIXED: Create a proper clone/copy
        z_raw = B['z']
    
        # Debug prints to understand what's happening
#        print(f"z_raw shape: {z_raw.shape}, batch_size: {batch_size}")
    
        # Handle the case where z_raw has an extra batch dimension
        if z_raw.dim() == 3:
            # z_raw is [batch, n_samples, n_params] 
            # For inference, we typically want the first batch element
            z_raw = z_raw[0]  # Now [n_samples, n_params]
 #           print(f"After removing batch dim z_raw shape: {z_raw.shape}")
    
        # KEY FIX: Create a proper clone that can be used in autograd
        # This ensures the tensor is not an inference tensor
        z = z_raw[:batch_size, :self.num_params_show].clone().detach().requires_grad_(z_raw.requires_grad)
    
  #      print(f"Final z shape: {z.shape}")
    
        s1 = s.reshape(batch_size, self.num_params_show, self.num_feat_param)
        s2 = torch.stack(
            [torch.cat([s1[:, i, :], s1[:, j, :]], dim=-1) for i, j in self.marginals],
            dim=1
        )
    
        logratios1 = self.logratios1(s1, z)
        logratios2 = self.logratios2(s2, z)
    
        return logratios1, logratios2
    
def preprocess_cholesky(store, Lfid, lmin=10, lmax=2000, N_sims=75000, Nbin_z=10, Nbin_ell=30):
    N_spectra = Nbin_z * (2 * Nbin_z + 1)
    ell_lims = np.logspace(np.log10(lmin), np.log10(lmax), Nbin_ell)
    ells = 0.5 * (ell_lims[:-1] + ell_lims[1:])
    #ell_keys = sorted(map(int, Lfid.keys()))
    #ells = ell_keys  # use only available ones

    Cells_chol = np.zeros((N_sims, N_spectra, Nbin_ell - 1))
    noise_chol = np.zeros((N_sims, N_spectra, Nbin_ell - 1))
    #Lfid = Lfid.item() 
    #store_pc = store.item()
    for ind, ell in enumerate(ells):
        #ell_key = str(int(round(float(ell))))
        #if ell_key not in Lfid:
        #    print(f"Skipping ell={ell_key} (not in Lfid)")
        #    continue
        # Linv = np.linalg.inv(Lfid[ell_key])
        Linv = np.linalg.inv(Lfid[str(int(ell))])
        Cells_chol[..., ind] = np.matmul(Linv, store['C_ells'][..., ind].T).T
        noise_chol[..., ind] = np.matmul(Linv, store['noise'][..., ind].T).T

    return Cells_chol, noise_chol
#    return swyft.Samples(z=store['z'], C_ells=Cells_chol, noise=noise_chol)

def apply_cholesky_to_obs_batch(obs, Lfid, lmin=10, lmax=2000, Nbin_ell=30):
    print(obs['C_ells'].shape)
    N_spectra, N_bins = obs['C_ells'].shape

    ell_lims = np.logspace(np.log10(lmin), np.log10(lmax), Nbin_ell)
    ells = 0.5 * (ell_lims[:-1] + ell_lims[1:])

    assert N_bins == len(ells)

    #oCells_chol = np.zeros_like(obs['C_ells'])
    #onoise_chol = np.zeros_like(obs['noise'])
    oCells_chol = np.zeros((N_spectra,Nbin_ell-1))
    onoise_chol = np.zeros((N_spectra,Nbin_ell-1))
    for sample_idx in range(N_samples):
        for ind, ell in enumerate(ells):
            ell_key = str(int(round(float(ell))))
            Linv = np.linalg.inv(Lfid[ell_key])
            oCells_chol[sample_idx, :, ind] = np.matmul(Linv, obs['C_ells'][sample_idx, :, ind].T).T
            onoise_chol[sample_idx, :, ind] = np.matmul(Linv, obs['noise'][sample_idx, :, ind].T).T

    return oCells_chol, onoise_chol

def apply_cholesky_to_obs(obs, Lfid, lmin=10, lmax=2000, Nbin_ell=30):
    """
    Apply inverse Cholesky rotation to a single observed C_ell and noise vector.

    Parameters:
        obs: dict with keys ['C_ells', 'noise'], each of shape (N_spectra, N_bins)
        Lfid: dict of Cholesky matrices, indexed by ell (as strings)
        lmin: Minimum ell (default 10)
        lmax: Maximum ell (default 2000)
        Nbin_ell: Number of ell bins (default 30)

    Returns:
        oCells_chol, onoise_chol: arrays of shape (N_spectra, N_bins)
    """
    #N_spectra, N_bins = 210, 29 #obs['C_ells'].shape
    N_spectra, N_bins = obs['C_ells'].shape
    # Compute ell bin centers
    ell_lims = np.logspace(np.log10(lmin), np.log10(lmax), Nbin_ell)
    ells = 0.5 * (ell_lims[:-1] + ell_lims[1:])

    oCells_chol = np.zeros((N_spectra, N_bins))
    onoise_chol = np.zeros((N_spectra, N_bins))

    for ind, ell in enumerate(ells):
        ell_key = str(int(round(float(ell))))
        Linv = np.linalg.inv(Lfid[str(int(ell))])
        oCells_chol[..., ind] = np.matmul(Linv, obs['C_ells'][..., ind].T).T
        onoise_chol[..., ind] = np.matmul(Linv, obs['noise'][..., ind].T).T

    return oCells_chol, onoise_chol

def lowerbounds_scales_bounds_sigmas(fiducial, finv_file, N_pars):
    Finv = np.load(finv_file)
    sigmas=[]
    for i in range(N_pars):
        sigmas.append(np.sqrt(Finv[i,i]))
    lower_bounds = [fiducial[i] - 5 * sigmas[i] for i in range(N_pars)]
    scales = [10 * sigmas[i] for i in range(N_pars)]
    bounds = []
    for i in range(N_pars):
        bounds.append([fiducial[i]-5*sigmas[i],fiducial[i]+5*sigmas[i]])
    bounds = np.array(bounds)
    return lower_bounds, scales, bounds, sigmas

def make_prior_samples(fiducial, sigmas, N_pars, num_params_show):
    lower = []
    width = []
    for i in range(N_pars - num_params_show, N_pars):
        lower.append([fiducial[i] - 5 * sigmas[i]])
        width.append([10 * sigmas[i]])
    samples = stats.uniform(np.array(lower).flatten(), np.array(width).flatten()).rvs(size=(500_000, num_params_show))
    return swyft.Samples(z=samples)

def load_or_precompute_cholesky(store, Lfid, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    cells_path = os.path.join(cache_dir, "Cells_chol.npy")
    noise_path = os.path.join(cache_dir, "noise_chol.npy")

    if os.path.exists(cells_path) and os.path.exists(noise_path):
        print("Loading cached Cholesky data")
        Cells_chol = np.load(cells_path, mmap_mode="r")
        noise_chol = np.load(noise_path, mmap_mode="r")
    else:
        print("Computing Cholesky transform (this is slow, done once)")
        assert store['C_ells'].shape[1] == Lfid.shape[0], \
            "Mismatch between simulation dimension and Lfid"

        Cells_chol = solve_triangular(
            Lfid, store['C_ells'].T, lower=True, check_finite=False
        ).T

        noise_chol = solve_triangular(
            Lfid, store['noise'].T, lower=True, check_finite=False
        ).T

        np.save(cells_path, Cells_chol)
        np.save(noise_path, noise_chol)
        print("Saved Cholesky-transformed simulations to cache")

    return Cells_chol, noise_chol


def save_predictions(predictions, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved to {save_path}")

def load_file(file_path):
    return np.load(file_path, allow_pickle=True)

