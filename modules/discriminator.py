from typing import List, Optional, Union

import numpy as np

import torch
from torch import Tensor, LongTensor
from torch import nn
from torch.nn import Module, ModuleList, LeakyReLU as LReLU, Sequential as Seq, Linear as Lin, \
    Embedding as Emb, Dropout, ELU, functional as F

from torch_geometric.nn import LayerNorm, global_mean_pool
from torch_geometric.data import Data, Batch

from .utils import _make_conv
from .layers import ECConv
from config import DiscriminatorConfig


class Discriminator(Module):
    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig()):
        super().__init__()
        hc = config.hidden_channels
        k = config.k
        aggr = config.aggr

        self.convs = ModuleList([ECConv(hc[0], hc[0] * 2, hc[1], k, aggr)])

        for in_c, out_c in zip(hc[1:-1], hc[2:]):
            self.convs.append(ECConv(in_c, in_c * 2, out_c, k, aggr))

        self.act = LReLU(0.2)

        self.ffn = Seq(
            Lin(hc[-1], hc[-1] * 2),
            Dropout(0.5, inplace=True),
            LReLU(0.2),
            Lin(hc[-1] * 2, hc[-1] * 2),
            LReLU(0.2),
            Lin(hc[-1] * 2, 1)
        )

    def forward(self, data: Union[Batch, Data]) -> Tensor:

        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)

        x = data.pos
        for conv in self.convs:
            x = conv(x, batch)
            x = self.act(x)
        x = global_mean_pool(x, batch)

        x = self.ffn(x)
        return x


class ConditionalDiscriminator(Module):
    def __init__(self, config: DiscriminatorConfig = DiscriminatorConfig()):
        super().__init__()
        hc = config.hidden_channels
        k = config.k
        aggr = config.aggr

        self.n_classes = config.n_classes

        self.cond_encoder = Seq(Emb(config.n_classes, 64), ELU(), Lin(64, 64), ELU(), Lin(64, 128))
        self.convs = ModuleList()
        self.norms = ModuleList()

        for in_c, out_c in zip(hc[:-1], hc[1:]):
            self.convs.append(ECConv(in_c + 128, in_c * 2, out_c, k, aggr, nonlin=ELU()))
            self.norms.append(LayerNorm(out_c))

        self.act = ELU()

        self.ffn = Seq(
            Lin(hc[-1] + 128, hc[-1] * 2),
            nn.LayerNorm(hc[-1] * 2),
            ELU(),
            Dropout(0.5, inplace=True),
            Lin(hc[-1] * 2, hc[-1] * 2),
            nn.LayerNorm(hc[-1] * 2),
            ELU(),
            Lin(hc[-1] * 2, 1)
        )

    def forward(self, data: Union[Batch, Data]) -> Tensor:

        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)

        c = self.cond_encoder(data.y)
        x = data.pos
        for conv, norm in zip(self.convs, self.norms):
            x = conv(torch.cat([x, c[batch]], dim=1), batch)
            x = norm(x, batch)
            x = self.act(x)

        x = global_mean_pool(x, batch)

        x = torch.cat([x, c], dim=1)
        x = self.ffn(x)
        return x
