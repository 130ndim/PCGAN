from typing import List

import torch
from torch.nn import Module, ModuleList, LeakyReLU as LReLU, ELU, Tanh, functional as F

from torch_geometric.data import Data

from modules.layers import UpsampleLinear

from config import GeneratorConfig
from .utils import _make_conv
from .layers import ECConv


class Generator(Module):
    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        super().__init__()
        lc = config.latent_channels
        hc = config.hidden_channels
        ups = config.upsamplings
        k = config.k
        aggr = config.aggr

        self.lin = UpsampleLinear(lc, hc[0], ups[0])
        self.convs = ModuleList([ECConv(hc[0], hc[0] * 2, hc[1], k, aggr)])
        self.upsamples = ModuleList()

        for in_c, out_c, n_p in zip(hc[1:-1], hc[2:], ups[1:]):
            self.upsamples.append(ECConv(in_c, in_c * 2, in_c * (n_p - 1), k, aggr))
            self.convs.append(ECConv(in_c, in_c * 2, out_c, k, aggr))

        self.act = LReLU(0.2)

        self.tanh = Tanh()

    def forward(self, latent):
        h, batch = self.lin(latent)
        for upsample, conv in zip(self.upsamples, self.convs[:-1]):
            # conv
            h = conv(h, batch)
            h = self.act(h)

            # generate new points
            h_u = upsample(h, batch)
            h_u = self.act(h_u)

            # append them to existing ones
            batch = batch.repeat_interleave(1 + h_u.size(-1) // h.size(-1))
            h = torch.cat([h, h_u], dim=1).view(-1, h.size(-1))

        h = self.convs[-1](h, batch)
        h = self.tanh(h)
        data = Data(pos=h, batch=batch)
        return data


class ConditionalGenerator(Module):
    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        super().__init__()
        lc = config.latent_channels
        hc = config.hidden_channels
        ups = config.upsamplings
        k = config.k
        aggr = config.aggr

        self.lin = UpsampleLinear(lc + 10, hc[0], ups[0])
        self.convs = ModuleList([ECConv(hc[0] + 10, hc[0] * 2, hc[1], k, aggr, nonlin=ELU())])
        self.upsamples = ModuleList()

        for in_c, out_c, n_p in zip(hc[1:-1], hc[2:], ups[1:]):
            self.upsamples.append(ECConv(in_c + 10, in_c * 2, in_c * (n_p - 1), k, aggr, nonlin=ELU()))
            self.convs.append(ECConv(in_c + 10, in_c * 2, out_c, k, aggr, nonlin=ELU()))

        self.act = ELU()

        self.tanh = Tanh()

    def forward(self, latent, cond):
        c = F.one_hot(cond, 10)
        h, batch = self.lin(torch.cat([latent, c], dim=1))
        for upsample, conv in zip(self.upsamples, self.convs[:-1]):
            # conv
            h = conv(torch.cat([h, c[batch]], dim=1), batch)
            h = self.act(h)

            # generate new points
            h_u = upsample(torch.cat([h, c[batch]], dim=1), batch)
            h_u = self.act(h_u)

            # append them to existing ones
            batch = batch.repeat_interleave(1 + h_u.size(-1) // h.size(-1))
            h = torch.cat([h, h_u], dim=1).view(-1, h.size(-1))

        h = self.convs[-1](torch.cat([h, c[batch]], dim=1), batch)
        h = self.tanh(h)
        data = Data(pos=h, batch=batch, y=cond)
        return data

