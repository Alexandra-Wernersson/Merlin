import torch
import numpy as np
import swyft
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):

    def __init__(self, V_proj, param_indices, num_feat_param=1,
                 num_blocks_ratios=6, hidden_feat_compress=(2048, 512),
                 dropout_ratios=0.1, hidden_feat_ratios=64,
                 checkpoint_dir=None, early_stopping_patience=5,
                 learning_rate=1e-3, batch_size=256):
        super().__init__()
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.num_params_show = len(param_indices)
        self.num_feat_param  = num_feat_param
        self.marginals       = self._get_marginals(self.num_params_show)
        self.checkpoint_dir  = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience

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

        layers = []
        for n_units in hidden_feat_compress:
            layers.append(torch.nn.LazyLinear(n_units))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.LazyLinear(self.num_params_show * num_feat_param))
        layers.append(torch.nn.LazyBatchNorm1d())
        self.sequential = torch.nn.Sequential(*layers)

        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=num_feat_param,
            num_params=self.num_params_show,
            varnames="z",
            num_blocks=num_blocks_ratios,
            dropout=dropout_ratios,
            hidden_features=hidden_feat_ratios,
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=2 * num_feat_param,
            marginals=self.marginals,
            varnames="z",
            num_blocks=num_blocks_ratios,
            dropout=dropout_ratios,
            hidden_features=hidden_feat_ratios,
        )

    @staticmethod
    def _get_marginals(n):
        return tuple((i, j) for i in range(n) for j in range(n) if j > i)

    def configure_callbacks(self):
        # Overrides swyft's AdamWReduceLROnPlateau.configure_callbacks default
        # (which saves to Trainer's default_root_dir) to save to checkpoint_dir
        # instead, so a trained network can be reloaded later without retraining
        # (see inference.predict_from_checkpoint).
        early_stop = EarlyStopping(monitor="val_loss", patience=self.early_stopping_patience)
        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_dir, filename="best",
            monitor="val_loss", mode="min", save_top_k=1,
        )
        return [early_stop, checkpoint]

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
