# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv1d, conv2d, conv3d

from torchkan.utils.typing import (
    Activation,
    Padding1D,
    Padding2D,
    Padding3D,
    PaddingND,
    Size2D,
    Size3D,
    SizeND,
)


class ReLUConvNDLayer(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        conv_w_fun: Callable[..., Tensor],
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        base_activation: Activation = nn.SiLU(),
        dropout: float = 0.0,
        train_ab: bool = True,
        **norm_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g = g
        self.k = k
        self.r = 4 * g**2 / (k + 1) ** 2
        self.train_ab = train_ab
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = base_activation
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs

        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        base_conv = conv_class(
            in_channels=in_channels // groups,
            out_channels=out_channels // groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.base_conv = nn.ModuleList([base_conv] * groups)

        relukan_conv = conv_class(
            in_channels=(g + k) * in_channels // groups,
            out_channels=out_channels // groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.relukan_conv = nn.ModuleList([relukan_conv] * groups)

        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g

        phase_low = phase_low[None, :].repeat(in_channels // groups, 1)
        phase_high = phase_high[None, :].repeat(in_channels // groups, 1)

        phase_dims = (1, in_channels // groups, k + g) + (1,) * ndim

        self.phase_low = nn.Parameter(
            phase_low.view(phase_dims),
            requires_grad=train_ab,
        )

        self.phase_high = nn.Parameter(
            phase_high.view(phase_dims),
            requires_grad=train_ab,
        )

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )
        for conv_layer in self.relukan_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

    def forward_relukan(self, x: Tensor, group_index: int) -> Tensor:
        if self.dropout:
            x = self.dropout(x)
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        x = torch.unsqueeze(x, dim=2)
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = (x1 * x2 * self.r) ** 2
        x = torch.flatten(x, start_dim=1, end_dim=2)

        y = self.relukan_conv[group_index](x)
        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_relukan(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class ReLUKANConv3DLayer(ReLUConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Padding3D = 0,
        dilation: Size3D = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv3d,
            norm_class=nn.InstanceNorm3d,
            conv_w_fun=conv3d,
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            g=g,
            k=k,
            train_ab=train_ab,
            **norm_kwargs,
        )


class ReLUKANConv2DLayer(ReLUConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D,
        stride: Size2D = 1,
        padding: Padding2D = 0,
        dilation: Size2D = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv2d,
            norm_class=nn.InstanceNorm2d,
            conv_w_fun=conv2d,
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            g=g,
            k=k,
            train_ab=train_ab,
            **norm_kwargs,
        )


class ReLUKANConv1DLayer(ReLUConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Padding1D = 0,
        dilation: int = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv1d,
            norm_class=nn.InstanceNorm1d,
            conv_w_fun=conv1d,
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            g=g,
            k=k,
            train_ab=train_ab,
            **norm_kwargs,
        )
