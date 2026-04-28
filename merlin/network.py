"""
Neural network for likelihood-to-evidence ratio estimation.

Corresponds to the ``Network`` class in the original ``training_utils.py``.
"""

import numpy as np
import torch
import swyft


class Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    """
    Compression + ratio-estimation network for 1-d and 2-d marginals.

    Parameters
    ----------
    V_proj:
        PCA projection matrix of shape ``(n_data, n_components)``.
    """

    def __init__(self, V_proj):
        super().__init__()
        self.learning_rate = 1e-3
        self.batch_size = 256
        self.num_params_show = 5
        self.num_feat_param = 4
        self.marginals = self._get_marginals(self.num_params_show)

        lenV = V_proj.shape[1]
        self.norm = swyft.networks.OnlineStandardizingLayer(
            torch.Size([lenV]), epsilon=1e-50
        )

        V_proj = np.array(V_proj, copy=True)
        self.register_buffer("V_proj", torch.tensor(V_proj, dtype=torch.float32))

        self.sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(self.num_params_show * self.num_feat_param),
            torch.nn.LazyBatchNorm1d(),
        )

        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=self.num_feat_param,
            num_params=self.num_params_show,
            varnames="z",
            num_blocks=4,
            dropout=0.1,
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=2 * self.num_feat_param,
            marginals=self.marginals,
            varnames="z",
            num_blocks=4,
            dropout=0.1,
        )

    @staticmethod
    def _get_marginals(n_params):
        return tuple(
            (i, j) for i in range(n_params) for j in range(n_params) if j > i
        )

    def forward(self, A, B):
        # Add signal + noise, project onto PCA basis
        s = (A["C_ells"] + A["noise"]) @ self.V_proj
        s = self.norm(s)
        s = self.sequential(s)

        z = B["z"][..., : self.num_params_show]
        s1 = s.reshape(-1, self.num_params_show, self.num_feat_param)
        s2 = torch.stack(
            [
                torch.cat([s1[:, i, :], s1[:, j, :]], dim=-1)
                for i, j in self.marginals
            ],
            dim=1,
        )

        return self.logratios1(s1, z), self.logratios2(s2, z)
