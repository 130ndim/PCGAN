import torch


class SamplePoints:
    def __init__(self, n_points=1024):
        self.n_points = n_points

    def __call__(self, data):
        idx = torch.randperm(data.pos.size(0))[:self.n_points]
        data.pos = data.pos[idx]
        data.x = data.x[idx]
        data.y = data.y[idx]
        return data