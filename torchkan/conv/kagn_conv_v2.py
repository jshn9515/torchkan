# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py

from functools import lru_cache
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv1d, conv2d, conv3d

PaddingType = Literal['valid', 'same']


class KAGNConvNDLayerV2(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        conv_w_fun: Callable,
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: PaddingType | int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        groups: int = 1,
        base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KAGNConvNDLayerV2, self).__init__()
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

        self.base_conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )

        self.poly_conv = conv_class(
            in_channels=in_channels * (spline_order + 1),
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )

        self.layer_norm = norm_class(out_channels, **norm_kwargs)
        self.beta_weights = nn.Parameter(
            torch.zeros(spline_order + 1, dtype=torch.float32)
        )

        # Initialize weights using Kaiming uniform distribution for better training start
        # for conv_layer in self.base_conv:
        nn.init.kaiming_uniform_(
            torch.as_tensor(self.base_conv.weight), nonlinearity='linear'
        )
        nn.init.kaiming_uniform_(
            torch.as_tensor(self.poly_conv.weight), nonlinearity='linear'
        )

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)

        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0
            / (
                (sum(kernel_size) / len(kernel_size)) ** ndim
                * self.in_channels
                * (self.spline_order + 1.0)
            ),
        )

    def beta(self, n: int, m: int) -> Tensor:
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Gram polynomials
    def gram_poly(self, x: Tensor, spline_order: int) -> Tensor:
        P0 = x.new_ones(x.size())

        if spline_order == 0:
            return torch.unsqueeze(P0, dim=-1)

        P1 = x
        grams_basis = [P0, P1]

        for i in range(2, spline_order + 1):
            P2 = x * P1 - self.beta(i - 1, i) * P0
            grams_basis.append(P2)
            P0, P1 = P1, P2

        indexes = [
            i * (spline_order + 1) + j
            for i in range(x.shape[1])
            for j in range(spline_order + 1)
        ]

        grams_basis = torch.concat(grams_basis, dim=1)
        grams_basis = grams_basis[:, indexes]
        return grams_basis

    def forward_kag(self, x: Tensor) -> Tensor:
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv(self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.spline_order))

        y = self.poly_conv(grams_basis)
        y = self.base_activation(self.layer_norm(y + basis))

        return y

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_kag(x)


class KAGNConv3DLayerV2(KAGNConvNDLayerV2):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KAGNConv3DLayerV2, self).__init__(
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


class KAGNConv2DLayerV2(KAGNConvNDLayerV2):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KAGNConv2DLayerV2, self).__init__(
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


class KAGNConv1DLayerV2(KAGNConvNDLayerV2):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super(KAGNConv1DLayerV2, self).__init__(
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
