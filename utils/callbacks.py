import pytorch_lightning as pl
import torch

import numpy as np

import wandb

import pytorch_lightning as pl
import torch

import numpy as np

import wandb


class WandbPCLogCallback(pl.Callback):
    """Logs the input images and output predictions of a module.

    Predictions and labels are logged as class indices."""

    def __init__(self, n):
        super().__init__()
        self.n = n

        self._iter = 0

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        z = torch.randn(self.n, pl_module.config.generator.latent_channels, device=pl_module.device)

        G_out = pl_module(z)
        pcs = G_out.pos.view(self.n, -1, 3).split(1, dim=0)

        trainer.logger.experiment.log({
            "val/examples": [
                wandb.Object3D(pc.squeeze().cpu().numpy(), caption=f"pc") for i, pc in enumerate(pcs)
            ],
            "global_step": self._iter
        })
        self._iter += 1


class ConditionalWandbPCLogCallback(pl.Callback):
    """Logs the input images and output predictions of a module.

    Predictions and labels are logged as class indices."""

    def __init__(self, n):
        super().__init__()
        self.n = n

        self._iter = 0

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        pcs = []
        for y in range(10):
            z = torch.randn(self.n, pl_module.config.generator.latent_channels, device=pl_module.device)
            cond = torch.ones(self.n, device=pl_module.device, dtype=torch.long) * y

            G_out = pl_module(z, cond)
            pos = torch.cat([G_out.pos, cond.view(-1, 1)[G_out.batch]], dim=1)
            pcs += pos.view(self.n, -1, 4).split(1, dim=0)

        trainer.logger.experiment.log({
            "val/examples": [
                wandb.Object3D(pc.squeeze().cpu().numpy(), caption=f"pc") for i, pc in enumerate(pcs)
            ],
            "global_step": self._iter
        })
        self._iter += 1
