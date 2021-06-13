from torch import Tensor
from torch.nn import Parameter, Sequential as Seq, Linear as Lin, LeakyReLU as LReLU, init, Module

from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn.inits import reset


class ECConv(DynamicEdgeConv):
    def __init__(self, in_channels, hidden_channels, out_channels, k=10, aggr='mean', nonlin=LReLU(0.2), **kwargs):
        nn = Seq(Lin(in_channels, hidden_channels), nonlin, Lin(hidden_channels, out_channels))
        super().__init__(nn=nn, k=k, aggr=aggr, **kwargs)
        self.W = Parameter(Tensor(in_channels, out_channels))
        self.V = Parameter(Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        if hasattr(self, 'W'):
            init.xavier_uniform_(self.W)
        if hasattr(self, 'V'):
            init.xavier_uniform_(self.V)

    def forward(self, x, batch) -> Tensor:
        h = super().forward(x, batch)
        h += x @ self.W
        return h

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(x_j - x_i) * (x_i @ self.V)
