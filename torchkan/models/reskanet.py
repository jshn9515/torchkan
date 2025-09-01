from functools import partial
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torchkan.conv import (
    BottleNeckKAGNConv2DLayer,
    FastKANConv2DLayer,
    KACNConv2DLayer,
    KAGNConv2DLayer,
    KALNConv2DLayer,
    KANConv2DLayer,
)
from torchkan.models.utils.conv_utils import (
    bottleneck_kagn_conv1x1,
    bottleneck_kagn_conv3x3,
    fast_kan_conv1x1,
    fast_kan_conv3x3,
    kacn_conv1x1,
    kacn_conv3x3,
    kagn_conv1x1,
    kagn_conv3x3,
    kaln_conv1x1,
    kaln_conv3x3,
    kan_conv1x1,
    kan_conv3x3,
    moe_bottleneck_kagn_conv3x3,
    moe_kaln_conv3x3,
)

from torchkan.utils.typing import Activation, ConvFunc


class BlockProtocol(Protocol):
    expansion: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ): ...


class BasicBlockTemplate(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        conv1x1x1_fun: ConvFunc,
        conv3x3x3_fun: ConvFunc,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
    ):
        super().__init__()
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock.')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3_fun(
            in_channels,
            out_channels,
            stride=stride,
            groups=groups,
        )
        self.conv2 = conv1x1x1_fun(out_channels, out_channels)
        self.downsample = downsample
        self.stride = stride
        self.base_width = base_width

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class KANBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        grid_size: int = 5,
        base_activation: Activation = nn.GELU(),
        grid_range: tuple[float, float] = (-1, 1),
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kan_conv1x1,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kan_conv3x3,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class FastKANBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        grid_size: int = 5,
        base_activation: Activation = nn.SiLU(),
        grid_range: tuple[float, float] = (-1.0, 1.0),
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            fast_kan_conv1x1,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            l1_decay=l1_decay,
            dropout=dropout,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            fast_kan_conv3x3,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            l1_decay=l1_decay,
            dropout=dropout,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class KALNBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kaln_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kaln_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class KAGNBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kagn_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kagn_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class BottleneckKAGNBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            bottleneck_kagn_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            bottleneck_kagn_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class KACNBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kacn_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kacn_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
# while original implementation places the stride at the first 1x1 convolution(self.conv1)
# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
# This variant is also known as ResNet V1.5 and improves accuracy according to
# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.


class BottleneckTemplate(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        conv1x1x1_fun: ConvFunc,
        conv3x3x3_fun: ConvFunc,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
    ):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1_fun(in_channels, width)
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3_fun(
            width,
            width,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1_fun(width, out_channels * self.expansion)
        # self.bn3 = norm_layer(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class KANBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        grid_size: int = 5,
        base_activation: Activation = nn.GELU(),
        grid_range: tuple[float, float] = (-1.0, 1.0),
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kan_conv1x1,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kan_conv3x3,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class FastKANBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        grid_size: int = 5,
        base_activation: Activation = nn.GELU(),
        grid_range: tuple[float, float] = (-1.0, 1.0),
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            fast_kan_conv1x1,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            fast_kan_conv3x3,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class KALNBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kaln_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kaln_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class KAGNBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        **norm_kwargs,
    ):
        conv1x1_fun = partial(
            kagn_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        conv3x3_fun = partial(
            kagn_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1_fun,
            conv3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class BottleneckKAGNBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
        **norm_kwargs,
    ):
        conv1x1_fun = partial(
            bottleneck_kagn_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        conv3x3_fun = partial(
            bottleneck_kagn_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1_fun,
            conv3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class MoEKALNBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        num_experts: int = 8,
        noisy_gating: bool = True,
        k: int = 2,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kaln_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            moe_kaln_conv3x3,
            spline_order=spline_order,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )

    def forward(self, x: Tensor, train: bool = True) -> Tensor | tuple[Tensor, Tensor]:
        identity = x
        out = self.conv1(x)
        out, moe_loss = self.conv2(out, train=train)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out, moe_loss


class MoEKALNBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        num_experts: int = 8,
        noisy_gating: bool = True,
        k: int = 2,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kaln_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            moe_kaln_conv3x3,
            spline_order=spline_order,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )

    def forward(self, x: Tensor, train: bool = True) -> Tensor | tuple[Tensor, Tensor]:
        identity = x
        out, moe_loss = self.conv1(x, train=train)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out, moe_loss


class KACNBottleneck(BottleneckTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kacn_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kacn_conv3x3,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )


class MoEBottleneckKAGNBasicBlock(BasicBlockTemplate):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spline_order: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        downsample: nn.Module | None = None,
        base_width: int = 64,
        num_experts: int = 8,
        noisy_gating: bool = True,
        k: int = 2,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kaln_conv1x1,
            spline_order=spline_order,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            moe_bottleneck_kagn_conv3x3,
            spline_order=spline_order,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            dropout=dropout,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            downsample=downsample,
            base_width=base_width,
        )

    def forward(self, x: Tensor, train: bool = True) -> Tensor | tuple[Tensor, Tensor]:
        identity = x
        out, moe_loss = self.conv1(x, train=train)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out, moe_loss


class ResKANet(nn.Module):
    def __init__(
        self,
        block: type[BlockProtocol],
        layers: list[int],
        input_channels: int = 3,
        num_classes: int = 1000,
        hidden_layer_dim: int | None = None,
        use_first_maxpool: bool = True,
        mp_kernel_size: int = 3,
        mp_stride: int = 2,
        mp_padding: int = 1,
        fcnv_kernel_size: int = 7,
        fcnv_stride: int = 2,
        fcnv_padding: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        width_scale: int = 1,
        replace_stride_with_dilation: list[bool] | None = None,
        dropout_linear: float = 0.25,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        **kan_kwargs,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.in_channels = 8 * width_scale
        self.hidden_layer_dim = hidden_layer_dim
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f'replace_stride_with_dilation should be None '
                f'or a 3-element tuple, got {replace_stride_with_dilation}'
            )

        self.groups = groups
        self.base_width = width_per_group
        self.use_first_maxpool = use_first_maxpool
        self.hidden_layer = None

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)

        kan_kwargs_fc = kan_kwargs.copy()
        kan_kwargs_fc.pop('groups', None)
        kan_kwargs_fc.pop('dropout', None)
        kan_kwargs_fc['dropout'] = dropout_linear

        match block.__name__:
            case 'KANBasicBlock' | 'KANBottleneck':
                self.conv1 = KANConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'FastKANBasicBlock' | 'FastKANBottleneck':
                self.conv1 = FastKANConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KALNBasicBlock' | 'KALNBottleneck':
                self.conv1 = KALNConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KAGNBasicBlock' | 'KAGNBottleneck':
                self.conv1 = KAGNConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    norm_layer=norm_layer,
                    **kan_kwargs_clean,
                )
            case 'BottleneckKAGNBottleneck' | 'BottleneckKAGNBasicBlock':
                self.conv1 = BottleNeckKAGNConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    norm_layer=norm_layer,
                    **kan_kwargs_clean,
                )
            case 'KACNBasicBlock' | 'KACNBottleneck':
                self.conv1 = KACNConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case _:
                raise ValueError(f'Block {block.__name__} is not supported.')

        self.maxpool = nn.Identity()
        if self.use_first_maxpool:
            self.maxpool = nn.MaxPool2d(
                kernel_size=mp_kernel_size,
                stride=mp_stride,
                padding=mp_padding,
            )

        self.layer1 = self._make_layer(
            block,
            8 * width_scale,
            layers[0],
            norm_layer=norm_layer,
            **kan_kwargs,
        )
        self.layer2 = self._make_layer(
            block,
            16 * width_scale,
            layers[1],
            stride=2,
            norm_layer=norm_layer,
            dilate=replace_stride_with_dilation[0],
            **kan_kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            32 * width_scale,
            layers[2],
            stride=2,
            norm_layer=norm_layer,
            dilate=replace_stride_with_dilation[1],
            **kan_kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            64 * width_scale,
            layers[3],
            stride=2,
            norm_layer=norm_layer,
            dilate=replace_stride_with_dilation[2],
            **kan_kwargs,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout_linear)
        self.fc = nn.Linear(
            64 * width_scale * block.expansion
            if hidden_layer_dim is None
            else hidden_layer_dim,
            num_classes,
        )

    def _make_layer(
        self,
        block: type[BlockProtocol],
        out_channels: int,
        num_block: int,
        stride: int = 1,
        dilate: bool = False,
        **kan_kwargs,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            match block.__name__:
                case 'KANBasicBlock' | 'KANBottleneck':
                    conv1x1 = partial(kan_conv1x1, **kan_kwargs)
                case 'FastKANBasicBlock' | 'FastKANBottleneck':
                    conv1x1 = partial(fast_kan_conv1x1, **kan_kwargs)
                case 'KALNBasicBlock' | 'KALNBottleneck':
                    conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
                case 'KAGNBasicBlock' | 'KAGNBottleneck':
                    conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
                case 'KAGNBasicBlock' | 'KAGNBottleneck':
                    conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
                case 'KACNBasicBlock' | 'KACNBottleneck':
                    conv1x1 = partial(kacn_conv1x1, **kan_kwargs)
                case 'BottleneckKAGNBasicBlock' | 'BottleneckKAGNBottleneck':
                    conv1x1 = partial(bottleneck_kagn_conv1x1, **kan_kwargs)
                case _:
                    raise ValueError(f'Block {block.__name__} is not supported.')

            downsample = conv1x1(
                self.in_channels,
                out_channels * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                dilation=previous_dilation,
                groups=self.groups,
                downsample=downsample,
                base_width=self.base_width,
                **kan_kwargs,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **kan_kwargs,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        if self.use_first_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)

        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class MoEResKANet(nn.Module):
    def __init__(
        self,
        block: type[BlockProtocol],
        layers: list[int],
        input_channels: int = 3,
        num_classes: int = 1000,
        hidden_layer_dim: int | None = None,
        use_first_maxpool: bool = True,
        mp_kernel_size: int = 3,
        mp_stride: int = 2,
        mp_padding: int = 1,
        fcnv_kernel_size: int = 7,
        fcnv_stride: int = 2,
        fcnv_padding: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        width_scale: int = 1,
        replace_stride_with_dilation: list[bool] | None = None,
        num_experts: int = 8,
        noisy_gating: bool = True,
        k: int = 2,
        dropout_linear: float = 0.0,
        **kan_kwargs,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.in_channels = 16 * width_scale
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None '
                f'or a 3-element tuple, got {replace_stride_with_dilation}'
            )

        self.groups = groups
        self.base_width = width_per_group
        self.use_first_maxpool = use_first_maxpool
        self.hidden_layer = None

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)

        match block.__name__:
            case 'MoEKALNBottleneck' | 'MoEKALNBasicBlock':
                self.conv1 = KALNConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
                if hidden_layer_dim is not None:
                    self.hidden_layer = kaln_conv1x1(
                        64 * width_scale * block.expansion,
                        hidden_layer_dim,
                        **kan_kwargs,
                    )
            case 'MoEBottleneckKAGNBasicBlock':
                self.conv1 = BottleNeckKAGNConv2DLayer(
                    input_channels,
                    self.in_channels,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
                if hidden_layer_dim is not None:
                    self.hidden_layer = bottleneck_kagn_conv1x1(
                        64 * width_scale * block.expansion,
                        hidden_layer_dim,
                        **kan_kwargs,
                    )
            case _:
                raise ValueError(f'Block {block.__name__} is not supported.')

        self.maxpool = nn.Identity()
        if use_first_maxpool:
            self.maxpool = nn.MaxPool2d(
                kernel_size=mp_kernel_size,
                stride=mp_stride,
                padding=mp_padding,
            )

        self.layer1 = self._make_layer(
            block,
            8 * width_scale,
            layers[0],
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kan_kwargs,
        )
        self.layer2 = self._make_layer(
            block,
            16 * width_scale,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kan_kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            32 * width_scale,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kan_kwargs,
        )
        self.layer4 = self._make_layer(
            block,
            64 * width_scale,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kan_kwargs,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            64 * width_scale * block.expansion
            if hidden_layer_dim is None
            else hidden_layer_dim,
            num_classes,
        )
        self.drop = nn.Dropout(p=dropout_linear)

    def _make_layer(
        self,
        block: type[BlockProtocol],
        out_channels: int,
        num_block: int,
        stride: int = 1,
        dilate: bool = False,
        num_experts: int = 8,
        noisy_gating: bool = True,
        k: int = 2,
        **kan_kwargs,
    ) -> nn.ModuleList:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            if block in (
                MoEKALNBottleneck,
                MoEKALNBasicBlock,
                MoEBottleneckKAGNBasicBlock,
            ):
                kan_kwargs.pop('num_experts', None)
                kan_kwargs.pop('noisy_gating', None)
                kan_kwargs.pop('k', None)
                conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
            else:
                raise ValueError(f'Block {block.__name__} is not supported.')

            downsample = conv1x1(
                self.in_channels,
                out_channels * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                num_experts=num_experts,
                noisy_gating=noisy_gating,
                k=k,
                **kan_kwargs,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    num_experts=num_experts,
                    noisy_gating=noisy_gating,
                    k=k,
                    **kan_kwargs,
                )
            )
        return nn.ModuleList(layers)

    def _forward_layer(
        self, layer: nn.ModuleList, x: Tensor, train: bool
    ) -> tuple[Tensor, float]:
        moe_loss = 0
        for block in layer:
            x, loss = block(x, train)
            moe_loss += loss
        return x, moe_loss

    def _forward_impl(
        self, x: Tensor, train: bool = True
    ) -> Tensor | tuple[Tensor, float]:
        x = self.conv1(x)
        if self.use_first_maxpool:
            x = self.maxpool(x)

        x, moe_loss1 = self._forward_layer(self.layer1, x, train)
        x, moe_loss2 = self._forward_layer(self.layer2, x, train)
        x, moe_loss3 = self._forward_layer(self.layer3, x, train)
        x, moe_loss4 = self._forward_layer(self.layer4, x, train)

        x = self.avgpool(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)

        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.fc(x)

        return x, (moe_loss1 + moe_loss2 + moe_loss3 + moe_loss4) / 4

    def forward(self, x: Tensor, train: bool = True) -> Tensor | tuple[Tensor, float]:
        return self._forward_impl(x, train)
