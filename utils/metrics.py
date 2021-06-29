from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, LongTensor

from torch_geometric.data import Data, Batch

from torch_cluster import knn
from torch_scatter import scatter

from tqdm.auto import tqdm


def chamfer_distance(
        pc1: Tensor,
        pc2: Tensor,
        batch1: Optional[LongTensor] = None,
        batch2: Optional[LongTensor] = None
) -> Tensor:
    if batch1 is None:
        batch1 = pc1.new_zeros(pc1.size(0), dtype=torch.long)
    if batch2 is None:
        batch2 = pc1.new_zeros(pc1.size(0), dtype=torch.long)

    row, col = knn(pc1, pc2, 1, batch1, batch2)
    left = scatter((pc2[row] - pc1[col]).pow(2).sum(-1), batch1, 0, reduce='sum')

    row, col = knn(pc2, pc1, 1, batch2, batch1)
    right = scatter((pc1[row] - pc2[col]).pow(2).sum(-1), batch2, 0, reduce='sum')

    out = left + right
    return out


def pairwise_distance(gen_pcs: Sequence[Data], real_pcs: Sequence[Data], verbose: bool = True) -> Tensor:
    m = []
    bar = tqdm(total=len(gen_pcs) * len(real_pcs))
    for i in real_pcs:
        r = []
        for j in gen_pcs:
            r.append(chamfer_distance(i.pos, j.pos))
            bar.update(1)
        m.append(r)
    return torch.tensor(m)


def compute_metrics(gen_pcs: Sequence[Data], real_pcs: Sequence[Data]) -> Dict[str, Tensor]:
    res = {}
    pd = pairwise_distance(gen_pcs, real_pcs)
    res['MMD-CD'] = pd.min(0)[0].mean().item()
    argsorted = pd.argsort(0)
    for k in [1, 5, 10, 15]:
        res[f'COV-CD@{k}'] = len(torch.unique(argsorted[:k]).view(-1)) / len(real_pcs)
    return res
