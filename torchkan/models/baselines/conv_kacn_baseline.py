import torch.nn as nn
from torch import Tensor

from torchkan.conv import KACNConv2DLayer
from torchkan.linear import KACN
from torchkan.utils import L1


class SimpleConvKACN(nn.Module):
    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        num_classes: int = 10,
        input_channels: int = 1,
        spline_order: int = 3,
        spline_order_out: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        dropout_linear: float = 0.0,
        l1_penalty: float = 0.0,
        affine: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            KACNConv2DLayer(
                input_channels,
                layer_sizes[0],
                kernel_size=3,
                spline_order=spline_order,
                groups=1,
                padding=1,
                stride=1,
                dilation=1,
                affine=affine,
            ),
            L1(
                KACNConv2DLayer(
                    layer_sizes[0],
                    layer_sizes[1],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[1],
                    layer_sizes[2],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[2],
                    layer_sizes[3],
                    kernel_size=3,
                    spline_order=spline_order,
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
            self.output = KACN(
                (layer_sizes[3], num_classes),
                dropout=dropout_linear,
                first_dropout=True,
                spline_order=spline_order_out,
            )
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.flatten(x)
        x = self.output(x)
        return x


class EightSimpleConvKACN(nn.Module):
    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        num_classes: int = 10,
        input_channels: int = 1,
        spline_order: int = 3,
        spline_order_out: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        dropout_linear: float = 0.0,
        l1_penalty: float = 0.0,
        affine: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            KACNConv2DLayer(
                input_channels,
                layer_sizes[0],
                kernel_size=3,
                spline_order=spline_order,
                groups=1,
                padding=1,
                stride=1,
                dilation=1,
                dropout=dropout,
                affine=affine,
            ),
            L1(
                KACNConv2DLayer(
                    layer_sizes[0],
                    layer_sizes[1],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[1],
                    layer_sizes[2],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[2],
                    layer_sizes[3],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[3],
                    layer_sizes[4],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[4],
                    layer_sizes[5],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[5],
                    layer_sizes[6],
                    kernel_size=3,
                    spline_order=spline_order,
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
                KACNConv2DLayer(
                    layer_sizes[6],
                    layer_sizes[7],
                    kernel_size=3,
                    spline_order=spline_order,
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
            self.output = KACN(
                (layer_sizes[7], num_classes),
                dropout=dropout_linear,
                first_dropout=True,
                spline_order=spline_order_out,
            )
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.flatten(x)
        x = self.output(x)
        return x
