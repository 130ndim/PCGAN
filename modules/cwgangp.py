import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric import transforms as T

import pytorch_lightning as pl

from modules.generator import ConditionalGenerator
from modules.discriminator import ConditionalDiscriminator

from config import Config


class CWGANGP(pl.LightningModule):
    opt_g: Adam = None
    opt_d: Adam = None
    sched_g: MultiStepLR = None
    sched_d: MultiStepLR = None

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.generator = ConditionalGenerator(config.generator)
        self.discriminator = ConditionalDiscriminator(config.discriminator)

    def forward(self, x=None, c=None):
        if x is None:
            x = torch.randn(1, self.config.generator.latent_channels, device=self.device)
        if c is None:
            c = torch.randint(0, 10, (1, ), device=self.device)
        return self.generator(x, c)

    def configure_optimizers(self):
        self.opt_g = Adam(self.generator.parameters(), lr=self.config.generator.lr, betas=(0., 0.9))
        self.opt_d = Adam(self.discriminator.parameters(), lr=self.config.discriminator.lr, betas=(0., 0.9))
        self.sched_g = MultiStepLR(
            self.opt_g, self.config.generator.multistep_milestones, gamma=self.config.generator.lr_decay
        )
        self.sched_d = MultiStepLR(
            self.opt_d, self.config.discriminator.multistep_milestones, gamma=self.config.discriminator.lr_decay
        )
        return [{'optimizer': self.opt_g,
                 'frequency': self.config.generator.freq,
                 'scheduler': self.sched_g},
                {'optimizer': self.opt_d,
                 'frequency': self.config.discriminator.freq,
                 'scheduler': self.sched_d}]

    def _generator_loss(self, data: Batch):
        z = torch.randn(size=(data.batch.max() + 1, self.config.generator.latent_channels), device=self.device)

        loss = -torch.mean(self.discriminator(self(z, data.y)))

        return loss

    def _gradient_penalty(self, real: Batch, fake: Batch, center: float = 1.):
        batch = real.batch

        alpha = torch.rand(batch.max() + 1, 1, device=self.device)[batch]
        interp = Data(pos=(alpha * real.pos + (1 - alpha) * fake.pos).requires_grad_(), y=real.y, batch=batch)
        d_interp = self.discriminator(interp)
        grad = torch.autograd.grad(outputs=d_interp,
                                   inputs=interp.pos,
                                   grad_outputs=interp.pos.new_ones(size=(batch.max() + 1, 1), requires_grad=True),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]
        gp = (torch.sqrt((grad ** 2).sum(1) + 1e-7) - center).pow(2).mean()
        return gp

    def _discriminator_loss(self, data: Batch):
        D_real_output = self.discriminator(data)

        z = torch.randn(size=(data.batch.max() + 1, self.config.generator.latent_channels), device=self.device)
        G_output = self(z, data.y)
        D_fake_output = self.discriminator(G_output)

        loss = -torch.mean(D_real_output) + torch.mean(D_fake_output)
        self.log('d_loss', loss.clone(), on_epoch=True, prog_bar=True)

        gp = self._gradient_penalty(data, G_output) * self.config.gradient_penalty
        self.log('gp', gp, on_step=True, on_epoch=False, prog_bar=True)

        loss += gp

        if self.config.regulate_difference:
            loss += self.config.lambda2 * (D_real_output - D_fake_output).pow(2).mean()

        return loss

    def _generator_step(self, data):
        g_loss = self._generator_loss(data)

        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def _discriminator_step(self, data):
        d_loss = self._discriminator_loss(data)

        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss = self._generator_step(batch)
        elif optimizer_idx == 1:
            loss = self._discriminator_step(batch)
        else:
            raise ValueError(f'Wrong optimizer_idx: {optimizer_idx}')

        self.log('g_lr', self.sched_g.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('d_lr', self.sched_d.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def train_dataloader(self):
        ds = ModelNet(self.config.data_path,
                      name='10',
                      transform=T.Compose([T.NormalizeScale(), T.SamplePoints(self.config.n_points)]))
        dl = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True, num_workers=8, drop_last=True)
        return dl
