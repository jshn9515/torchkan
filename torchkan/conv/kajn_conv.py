# based on this implementation: https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py

from functools import lru_cache
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


class KAJNConvNDLayer(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        conv_w_fun: Callable[..., Tensor],
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        a: float = 1.0,
        b: float = 1.0,
        base_activation: Activation = nn.SiLU(),
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = base_activation
        self.conv_w_fun = conv_w_fun
        self.a = a
        self.b = b
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

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim

        poly_shape = (
            groups,
            out_channels // groups,
            (in_channels // groups) * (spline_order + 1),
            *kernel_size,
        )

        self.poly_weights = nn.Parameter(torch.randn(poly_shape))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

        nn.init.normal_(
            self.poly_weights,
            mean=0.0,
            std=1
            / (
                in_channels
                * (spline_order + 1)
                * sum(kernel_size)
                / len(kernel_size) ** ndim
            ),
        )

    @lru_cache(maxsize=128)
    def compute_jacobi_polynomials(self, x: Tensor, order: int) -> Tensor:
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return torch.unsqueeze(P0, dim=-1)

        P1 = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        jacobi_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for i in range(2, order + 1):
            theta_k = (
                (2 * i + self.a + self.b)
                * (2 * i + self.a + self.b - 1)
                / (2 * i * (i + self.a + self.b))
            )
            theta_k1 = (
                (2 * i + self.a + self.b - 1)
                * (self.a * self.a - self.b * self.b)
                / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            theta_k2 = (
                (i + self.a - 1)
                * (i + self.b - 1)
                * (2 * i + self.a + self.b)
                / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            pn = (theta_k * x + theta_k1) * jacobi_polys[
                i - 1
            ].clone() - theta_k2 * jacobi_polys[i - 2].clone()
            jacobi_polys.append(pn)

        return torch.concat(jacobi_polys, dim=1)

    def forward_kaj(self, x: Tensor, group_index: int) -> Tensor:
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](x)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = torch.tanh(x)

        # Compute Legendre polynomials for the normalized x
        jacobi_basis = self.compute_jacobi_polynomials(x_normalized, self.spline_order)

        if self.dropout:
            jacobi_basis = self.dropout(jacobi_basis)

        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        # Compute polynomial output using polynomial weights
        poly_output = self.conv_w_fun(
            jacobi_basis,
            self.poly_weights[group_index],
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=1,
        )

        # Combine base and polynomial outputs, normalize, and activate
        x = base_output + poly_output
        if isinstance(self.layer_norm[group_index], nn.LayerNorm):
            original_shape = x.shape
            x = x.view(original_shape[0], -1)
            x = self.layer_norm[group_index](x)
            x = x.view(original_shape)
        else:
            x = self.layer_norm[group_index](x)
        x = self.base_activation(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_kaj(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class KAJNConv3DLayer(KAJNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Padding3D = 0,
        dilation: Size3D = 1,
        groups: int = 1,
        spline_order: int = 3,
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
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            **norm_kwargs,
        )


class KAJNConv2DLayer(KAJNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D,
        stride: Size2D = 1,
        padding: Padding2D = 0,
        dilation: Size2D = 1,
        groups: int = 1,
        spline_order: int = 3,
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
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            **norm_kwargs,
        )


class KAJNConv1DLayer(KAJNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Padding1D = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
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
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            **norm_kwargs,
        )
