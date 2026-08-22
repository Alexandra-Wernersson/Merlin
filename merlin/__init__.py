import os
import logging
import warnings

# Suppress noisy third-party warnings/prints that drown out merlin's progress output.
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 128))
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*UninitializedParameter.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*os\\.fork\\(\\) was called.*")
warnings.filterwarnings("ignore", message=".*torch\\.set_default_tensor_type\\(\\) is deprecated.*")

# pandas calls numexpr.set_num_threads() with the full core count regardless
# of NUMEXPR_MAX_THREADS, printing an unfilterable C-level error — disable it.
import pandas as _pd
_pd.set_option("compute.use_numexpr", False)
del _pd

from .config import load_config, populate_train_dir
from .params import (
    PARAMS, DERIVED_PARAMS, COSMO_PARAMS, NUISANCE_KEYS, N_COSMO, ZS,
    PARAM_GROUPS, PARAM_LABELS, MCMC_KEY_MAP, resolve_params, resolve_derived_names,
)
from .tracers import load_dndz
from .priors import (
    PriorSpec, resolve_priors, apply_fisher_bounds, load_fisher_sigmas,
    resolve_inference_params,
)
from .simulator import Simulator, build_simulator, sample_correlated_noise
from .preprocessing import (
    preprocess,
    preprocess_obs,
    apply_cholesky,
    load_or_precompute_cholesky,
    load_or_compute_pca,
    make_resampler,
    make_scale_cut_mask,
    select_data_probes,
)
from .io import save_predictions, load_predictions, load_file
from .network import Network
from .train import train
from .inference import infer, predict_from_checkpoint, load_network_from_checkpoint
from .simulate import simulate
from .observation import generate_observation
from .fisher import run_fisher
from .coverage import run_coverage_test
from .plotting import (
    setup_latex_style, plot_corner_mode, plot_coverage_mode, plot_loss_mode,
    load_mcmc_overlay, mcmc_samples_for,
)
from .swyft_patches import apply_patches, set_swyft_filled

# Suppress jax/pytorch_lightning/lightning_fabric startup log chatter.
# Must come AFTER the imports above: importing pytorch_lightning resets its logger to INFO.
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
