import torch
import torch.nn as nn
from torch import Tensor

from torchkan.utils.typing import (
    Padding1D,
    Padding2D,
    Padding3D,
    PaddingND,
    Size2D,
    Size3D,
    SizeND,
)


class KACNConvNDLayer(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
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
        self.ndim = ndim
        self.epsilon = 1e-7
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

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        poly_conv = conv_class(
            in_channels=(spline_order + 1) * in_channels // groups,
            out_channels=out_channels // groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.poly_conv = nn.ModuleList([poly_conv] * groups)

        arange_buffer_size = (1, 1, -1) + (1,) * ndim
        self.register_buffer(
            'arange', torch.arange(0, spline_order + 1, 1).view(arange_buffer_size)
        )

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.poly_conv:
            nn.init.normal_(
                torch.as_tensor(conv_layer.weight),
                mean=0.0,
                std=1
                / (
                    in_channels
                    * (spline_order + 1)
                    * (sum(kernel_size) / len(kernel_size)) ** ndim
                ),
            )

    def forward_kacn(self, x: Tensor, group_index: int) -> Tensor:
        # Apply base activation to input and then linear transform with base weights
        x = torch.tanh(x)
        x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        x = torch.unsqueeze(x, dim=2)
        x = (x * self.arange).flatten(1, 2)  # type: ignore[assignment]
        x = torch.cos(x)
        x = self.poly_conv[group_index](x)
        x = self.layer_norm[group_index](x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_kacn(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class KACNConv3DLayer(KACNConvNDLayer):
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


class KACNConv2DLayer(KACNConvNDLayer):
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


class KACNConv1DLayer(KACNConvNDLayer):
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
