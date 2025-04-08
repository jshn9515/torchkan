from functools import lru_cache
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv1d, conv2d, conv3d

PaddingType = Literal['valid', 'same']


class KABNConvNDLayer(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        conv_w_fun: Callable,
        ndim: int,
        in_channels: int,
        out_channels: int,
        degree: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: PaddingType | int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        groups: int = 1,
        base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KABNConvNDLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
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

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim

        poly_shape = (
            groups,
            out_channels // groups,
            (in_channels // groups) * (degree + 1),
            *kernel_size,
        )

        self.poly_weights = nn.Parameter(torch.randn(poly_shape))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')

    @lru_cache(maxsize=128)
    def bernstein_poly(self, x: Tensor, order: int) -> Tensor:
        bernsteins = torch.ones(
            x.shape + (self.degree + 1,), dtype=x.dtype, device=x.device
        )
        for j in range(1, order + 1):
            for k in range(order + 1 - j):
                bernsteins[..., k] = (
                    bernsteins[..., k] * (1 - x) + bernsteins[..., k + 1] * x
                )

        bernsteins = bernsteins.moveaxis(-1, 2).flatten(1, 2)

        return bernsteins

    def forward_kab(self, x: Tensor, group_index: int) -> Tensor:
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](x)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = torch.sigmoid(x)

        if self.dropout:
            x_normalized = self.dropout(x_normalized)

        # Compute Legendre polynomials for the normalized x
        bernstein_basis = self.bernstein_poly(x_normalized, self.degree)
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        # Compute polynomial output using polynomial weights
        poly_output = self.conv_w_fun(
            bernstein_basis,
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
            y = self.forward_kab(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class KABNConv3DLayer(KABNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        degree: int = 3,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KABNConv3DLayer, self).__init__(
            conv_class=nn.Conv3d,
            norm_class=nn.InstanceNorm3d,
            conv_w_fun=conv3d,
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            degree=degree,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            **norm_kwargs,
        )


class KABNConv2DLayer(KABNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        degree: int = 3,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KABNConv2DLayer, self).__init__(
            conv_class=nn.Conv2d,
            norm_class=nn.InstanceNorm2d,
            conv_w_fun=conv2d,
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            degree=degree,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            **norm_kwargs,
        )


class KABNConv1DLayer(KABNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        degree: int = 3,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KABNConv1DLayer, self).__init__(
            conv_class=nn.Conv1d,
            norm_class=nn.InstanceNorm1d,
            conv_w_fun=conv1d,
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            degree=degree,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            **norm_kwargs,
        )
