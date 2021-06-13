import torch
from torch.nn import Module, Linear


class UpsampleLinear(Module):
    def __init__(self, input_channels, output_channels, num_points):
        super().__init__()
        self.output_channels = output_channels
        self.num_points = num_points
        self.lin = Linear(input_channels, output_channels * num_points)
    #
    # def reset_parameters(self):
    #     self.lin.weight.data.normal_()
    #     self.lin.bias.data.fill_(0)

    def forward(self, x):
        B = x.size(0)
        h = self.lin(x).view(-1, self.output_channels)
        batch = torch.arange(B, device=x.device).repeat_interleave(self.num_points)
        return h, batch


class UpsampleConv(Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
            else:
                param.data.normal_()

    def forward(self, x, batch=None):
        C = x.size(-1)
        h = super().forward(x, batch=batch)
        h = torch.cat([x, h], dim=1).view(-1, C)
        return h
