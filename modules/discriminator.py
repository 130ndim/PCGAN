from typing import List, Optional, Union

import numpy as np

import torch
from torch import Tensor, LongTensor
from torch.nn import Module, ModuleList, LeakyReLU as LReLU, Sequential as Seq, Linear as Lin, \
    Dropout, ELU, functional as F

from torch_geometric.nn import global_mean_pool
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

        self.convs = ModuleList([ECConv(hc[0] + 10, hc[0] * 2, hc[1], k, aggr, nonlin=ELU())])

        for in_c, out_c in zip(hc[1:-1], hc[2:]):
            self.convs.append(ECConv(in_c + 10, in_c * 2, out_c, k, aggr, nonlin=ELU()))

        self.act = ELU()

        self.ffn = Seq(
            Lin(hc[-1] + 10, hc[-1] * 2),
            Dropout(0.5, inplace=True),
            ELU(),
            Lin(hc[-1] * 2, hc[-1] * 2),
            ELU(),
            Lin(hc[-1] * 2, 1)
        )

    def forward(self, data: Union[Batch, Data]) -> Tensor:

        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)

        c = F.one_hot(data.y, 10)

        x = data.pos
        for conv in self.convs:
            x = conv(torch.cat([x, c[batch]], dim=1), batch)
            x = self.act(x)
        x = global_mean_pool(x, batch)

        x = torch.cat([x, c], dim=1)
        x = self.ffn(x)
        return x
