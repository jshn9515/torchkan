import torch
import torch.nn as nn
from torch import Tensor

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


class KANConvNDLayer(nn.Module):
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
        grid_size: int = 5,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: Activation = nn.GELU(),
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
        self.grid_size = grid_size
        self.base_activation = base_activation
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs
        self.dropout = None

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
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

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

        spline_conv = conv_class(
            in_channels=(grid_size + spline_order) * in_channels // groups,
            out_channels=out_channels // groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.spline_conv = nn.ModuleList([spline_conv] * groups)

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        self.prelu = nn.ModuleList([nn.PReLU()] * groups)

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32,
        )
        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

    def forward_kan(self, x: Tensor, group_index: int) -> Tensor:
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))

        # Compute the basis for the spline using intervals and input values.
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view([1] * (self.ndim + 1) + [-1])
        grid = grid.expand(target).contiguous().to(x.device)

        x_uns = torch.unsqueeze(x, dim=-1)  # Expand dimensions for spline operations.
        bases = torch.logical_and(x_uns >= grid[..., :-1], x_uns < grid[..., 1:])
        bases = bases.to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., : -(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(
                right_intervals == left_intervals,
                torch.ones_like(right_intervals),
                right_intervals - left_intervals,
            )
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + (
                (grid[..., k + 1 :] - x_uns)
                / (grid[..., k + 1 :] - grid[..., 1:-k])
                * bases[..., 1:]
            )
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelu[group_index](
            self.layer_norm[group_index](base_output + spline_output)
        )

        if self.dropout:
            x = self.dropout(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_kan(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class KANConv3DLayer(KANConvNDLayer):
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
        grid_size: int = 5,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: Activation = nn.GELU(),
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
            grid_size=grid_size,
            grid_range=grid_range,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs,
        )


class KANConv2DLayer(KANConvNDLayer):
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
        grid_size: int = 5,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: Activation = nn.GELU(),
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
            grid_size=grid_size,
            grid_range=grid_range,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs,
        )


class KANConv1DLayer(KANConvNDLayer):
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
        grid_size: int = 5,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: Activation = nn.GELU(),
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
            grid_size=grid_size,
            grid_range=grid_range,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs,
        )
