import torch.nn as nn
from torch import Tensor

from torchkan.conv import FastKANConv2DLayer
from torchkan.linear import FastKAN
from torchkan.utils import L1


class SimpleFastConvKAN(nn.Module):
    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        num_classes: int = 10,
        input_channels: int = 1,
        grid_size: int = 8,
        spline_order_out: int = 2,
        groups: int = 1,
        dropout: float = 0.0,
        dropout_linear: float = 0.0,
        l1_penalty: float = 0.0,
        affine: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            FastKANConv2DLayer(
                input_channels,
                layer_sizes[0],
                grid_size=grid_size,
                kernel_size=3,
                groups=1,
                padding=1,
                stride=1,
                dilation=1,
                affine=affine,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[0],
                    layer_sizes[1],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=2,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[1],
                    layer_sizes[2],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=2,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[2],
                    layer_sizes[3],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=1,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        if spline_order_out < 2:
            self.output = nn.Sequential(
                nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes)
            )
        else:
            self.output = FastKAN(
                (layer_sizes[3], num_classes),
                dropout=dropout_linear,
                first_dropout=True,
                grid_size=grid_size,
            )
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.flatten(x)
        x = self.output(x)
        return x


class EightSimpleFastConvKAN(nn.Module):
    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        num_classes: int = 10,
        input_channels: int = 1,
        grid_size: int = 8,
        spline_order_out: int = 2,
        groups: int = 1,
        dropout: float = 0.0,
        dropout_linear: float = 0.0,
        l1_penalty: float = 0.0,
        affine: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            FastKANConv2DLayer(
                input_channels,
                layer_sizes[0],
                grid_size=grid_size,
                kernel_size=3,
                groups=1,
                padding=1,
                stride=1,
                dilation=1,
                affine=affine,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[0],
                    layer_sizes[1],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=2,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[1],
                    layer_sizes[2],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=2,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[2],
                    layer_sizes[3],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=1,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[3],
                    layer_sizes[4],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=1,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[4],
                    layer_sizes[5],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=2,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[5],
                    layer_sizes[6],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=1,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            L1(
                FastKANConv2DLayer(
                    layer_sizes[6],
                    layer_sizes[7],
                    grid_size=grid_size,
                    kernel_size=3,
                    groups=groups,
                    padding=1,
                    stride=1,
                    dilation=1,
                    dropout=dropout,
                    affine=affine,
                ),
                l1_penalty,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        if spline_order_out < 2:
            self.output = nn.Sequential(
                nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes)
            )
        else:
            self.output = FastKAN(
                (layer_sizes[7], num_classes),
                dropout=dropout_linear,
                first_dropout=True,
                grid_size=grid_size,
            )
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.flatten(x)
        x = self.output(x)
        return x
