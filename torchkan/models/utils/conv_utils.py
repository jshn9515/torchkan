import torch.nn as nn

from torchkan.conv import (
    BottleNeckKAGNConv2DLayer,
    BottleNeckSelfKAGNtention2D,
    FastKANConv2DLayer,
    KACNConv2DLayer,
    KAGNConv2DLayer,
    KALNConv2DLayer,
    KANConv2DLayer,
    MoEBottleNeckKAGNConv2DLayer,
    MoEKAGNConv2DLayer,
    MoEKALNConv2DLayer,
    SelfKAGNtention2D,
    WavKANConv2DLayer,
)
from torchkan.utils import L1

from torchkan.utils.typing import Padding2D, Size2D, WaveletType, WaveletVersion


def kan_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    grid_size: int = 5,
    base_activation: nn.Module = nn.GELU(),
    grid_range: tuple[float, float] = (-1.0, 1.0),
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, weight_decay=l1_decay)
    return conv


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: Size2D = 1,
    padding: Size2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    base_activation: nn.Module = nn.GELU(),
    l1_decay: float = 0.0,
    dropout: float = 0.0,
) -> nn.Module:
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    norm = nn.BatchNorm2d(out_channels)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), conv, norm, base_activation)

    return nn.Sequential(conv, norm, base_activation())


def kan_conv1x1(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    grid_size: int = 5,
    base_activation: nn.Module = nn.GELU(),
    grid_range: tuple[float, float] = (-1.0, 1.0),
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kaln_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    l1_decay: float = 0.0,
    norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    **norm_kwargs,
) -> nn.Module:
    conv = KALNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kagn_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer=nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def self_kagn_conv3x3(
    in_channels: int,
    inner_projection: int | None = None,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    **norm_kwargs,
) -> nn.Module:
    conv = SelfKAGNtention2D(
        in_channels=in_channels,
        inner_projection=inner_projection
        if inner_projection is None
        else in_channels // inner_projection,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    return conv


def bottleneck_kagn_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer: type[nn.Module] = nn.BatchNorm2d,
    l1_decay: float = 0.0,
    dim_reduction: float = 8,
    **norm_kwargs,
) -> nn.Module:
    conv = BottleNeckKAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        sstride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        dim_reduction=dim_reduction,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def moe_bottleneck_kagn_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer: type[nn.Module] = nn.BatchNorm2d,
    l1_decay: float = 0.0,
    dim_reduction: float = 8,
    num_experts: int = 8,
    noisy_gating: bool = True,
    k: int = 2,
    **norm_kwargs,
) -> nn.Module:
    conv = MoEBottleNeckKAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        dim_reduction=dim_reduction,
        norm_layer=norm_layer,
        num_experts=num_experts,
        noisy_gating=noisy_gating,
        k=k,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def bottleneck_kagn_conv1x1(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer: type[nn.Module] = nn.BatchNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = BottleNeckKAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def self_bottleneck_kagn_conv3x3(
    in_channels: int,
    inner_projection: int | None = None,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer=nn.InstanceNorm2d,
    dim_reduction: float = 8,
    **norm_kwargs,
) -> nn.Module:
    conv = BottleNeckSelfKAGNtention2D(
        in_channels,
        inner_projection=inner_projection
        if inner_projection is None
        else in_channels // inner_projection,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        dim_reduction=dim_reduction,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    return conv


def moe_kaln_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    num_experts: int = 8,
    noisy_gating: bool = True,
    k: int = 2,
    dropout: float = 0.0,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = MoEKALNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        num_experts=num_experts,
        noisy_gating=noisy_gating,
        k=k,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def moe_kagn_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    num_experts: int = 8,
    noisy_gating: bool = True,
    k: int = 2,
    dropout: float = 0.0,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = MoEKAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        num_experts=num_experts,
        noisy_gating=noisy_gating,
        k=k,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kaln_conv1x1(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KALNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kagn_conv1x1(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    dropout: float = 0.0,
    norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        spline_order=spline_order,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kacn_conv3x3(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KACNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        spline_order=spline_order,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kacn_conv1x1(
    in_channels: int,
    out_channels: int,
    spline_order: int = 3,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = KACNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        spline_order=spline_order,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def fast_kan_conv3x3(
    in_channels: int,
    out_channels: int,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    grid_size: int = 8,
    base_activation: nn.Module = nn.SiLU(),
    grid_range: tuple[float, float] = (-2, 2),
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = FastKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def fast_kan_conv1x1(
    in_channels: int,
    out_channels: int,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    grid_size: int = 8,
    base_activation: nn.Module = nn.SiLU(),
    grid_range: tuple[float, float] = (-2, 2),
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = FastKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def wav_kan_conv3x3(
    in_channels: int,
    out_channels: int,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    wavelet_type: WaveletType = 'mexican_hat',
    wavelet_version: WaveletVersion = 'fast',
    **norm_kwargs,
) -> nn.Module:
    conv = WavKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        wavelet_type=wavelet_type,
        wavelet_version=wavelet_version,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def wav_kan_conv1x1(
    in_channels: int,
    out_channels: int,
    stride: Size2D = 1,
    padding: Padding2D = 1,
    dilation: Size2D = 1,
    groups: int = 1,
    wavelet_type: WaveletType = 'mexican_hat',
    wavelet_version: WaveletVersion = 'fast',
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    conv = WavKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        wavelet_type=wavelet_type,
        wav_version=wavelet_version,
        dropout=dropout,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv
