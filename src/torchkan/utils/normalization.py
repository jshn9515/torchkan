import torch.nn as nn
from torch import Tensor


class SelfSpatialNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 16, affine: bool = True):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_groups,
            num_channels,
            eps=1e-6,
            affine=affine,
        )
        self.conv_y = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_b = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, f: Tensor) -> Tensor:
        f = self.norm_layer(f)
        f = f * self.conv_y(f) + self.conv_b(f)
        return f


class SpatialNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_channels_cond: int,
        num_groups: int = 32,
        affine: bool = True,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_groups,
            num_channels,
            eps=1e-6,
            affine=affine,
        )
        self.conv_y = nn.Conv2d(
            num_channels_cond,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_b = nn.Conv2d(
            num_channels_cond,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, f: Tensor, c: Tensor) -> Tensor:
        f = self.norm_layer(f)
        f = f * self.conv_y(c) + self.conv_b(c)
        return f
