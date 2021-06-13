from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU as LReLU
from torch_geometric.nn import DynamicEdgeConv


def _make_conv(in_c, h_c, out_c, k, aggr):
    return DynamicEdgeConv(
        Seq(Lin(in_c, h_c), LReLU(0.2), Lin(h_c, out_c)),
        k=k,
        aggr=aggr
    )