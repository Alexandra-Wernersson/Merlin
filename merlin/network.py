import torch
import numpy as np
import swyft


class Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):

    def __init__(self, V_proj, param_indices, num_feat_param=1, num_blocks=6):
        super().__init__()
        self.learning_rate   = 1e-3
        self.batch_size      = 256
        self.num_params_show = len(param_indices)
        self.num_feat_param  = num_feat_param
        self.marginals       = self._get_marginals(self.num_params_show)

        self.register_buffer(
            "param_indices",
            torch.tensor(param_indices, dtype=torch.long)
        )

        lenV = V_proj.shape[1]
        self.norm = swyft.networks.OnlineStandardizingLayer(
            torch.Size([lenV]), epsilon=1e-50
        )
        self.register_buffer(
            "V_proj", torch.tensor(np.array(V_proj, copy=True), dtype=torch.float32)
        )

        self.sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(2048),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(self.num_params_show * num_feat_param),
            torch.nn.LazyBatchNorm1d(),
        )

        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=num_feat_param,
            num_params=self.num_params_show,
            varnames="z",
            num_blocks=num_blocks,
            dropout=0.1,
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=2 * num_feat_param,
            marginals=self.marginals,
            varnames="z",
            num_blocks=num_blocks,
            dropout=0.1,
        )

    @staticmethod
    def _get_marginals(n):
        return tuple((i, j) for i in range(n) for j in range(n) if j > i)

    def forward(self, A, B):
        s = (A["C_ells"] + A["noise"]) @ self.V_proj
        s = self.norm(s)
        s = self.sequential(s)

        z_full = B["z"]
        # training: z_full has all N_pars columns → select by index
        # inference: z_full already has only num_params_show columns
        if z_full.shape[-1] > self.num_params_show:
            z = z_full[..., self.param_indices]
        else:
            z = z_full

        s1 = s.reshape(-1, self.num_params_show, self.num_feat_param)
        s2 = torch.stack(
            [torch.cat([s1[:, i, :], s1[:, j, :]], dim=-1) for i, j in self.marginals],
            dim=1,
        )

        return self.logratios1(s1, z), self.logratios2(s2, z)
