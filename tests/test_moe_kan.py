import itertools

import pytest
import torch
import torch.nn as nn

from torchkan.conv import (
    MoEFastKANConv1DLayer,
    MoEFastKANConv2DLayer,
    MoEFastKANConv3DLayer,
    MoEKACNConv1DLayer,
    MoEKACNConv2DLayer,
    MoEKACNConv3DLayer,
    MoEKAGNConv1DLayer,
    MoEKAGNConv2DLayer,
    MoEKAGNConv3DLayer,
    MoEKALNConv1DLayer,
    MoEKALNConv2DLayer,
    MoEKALNConv3DLayer,
    MoEKANConv1DLayer,
    MoEKANConv2DLayer,
    MoEKANConv3DLayer,
    MoEWavKANConv1DLayer,
    MoEWavKANConv2DLayer,
    MoEWavKANConv3DLayer,
)


@pytest.mark.parametrize(
    'groups, noisy_gating',
    itertools.product([1, 4], [True, False]),
)
def test_moekan_conv_1d(groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = MoEKANConv1DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'groups, noisy_gating',
    itertools.product([1, 4], [True, False]),
)
def test_moekan_conv_2d(groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = MoEKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'groups, noisy_gating',
    itertools.product([1, 4], [True, False]),
)
def test_moekan_conv_3d(groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = MoEKANConv3DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
    out, _ = net(input_tensor, False)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating',
    itertools.product([0.0, 0.5], [1, 4], [True, False]),
)
def test_moefastkan_conv_1d(dropout: float, groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = MoEFastKANConv1DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        dropout=dropout,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating',
    itertools.product([0.0, 0.5], [1, 4], [True, False]),
)
def test_moefastkan_conv_2d(dropout: float, groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = MoEFastKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        dropout=dropout,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating',
    itertools.product([0.0, 0.5], [1, 4], [True, False]),
)
def test_moekan_fastconv_3d(dropout: float, groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = MoEFastKANConv3DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        dropout=dropout,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
    out, _ = net(input_tensor, False)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating, net_class',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        [True, False],
        [MoEKALNConv1DLayer, MoEKAGNConv1DLayer, MoEKACNConv1DLayer],
    ),
)
def test_moekalgcn_conv_1d(
    dropout: float, groups: int, noisy_gating: bool, net_class: nn.Module
):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = net_class(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        spline_order=3,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating, net_class',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        [True, False],
        [MoEKALNConv2DLayer, MoEKAGNConv2DLayer, MoEKACNConv2DLayer],
    ),
)
def test_moekalgcn_conv_2d(
    dropout: float, groups: int, noisy_gating: bool, net_class: nn.Module
):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = net_class(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        spline_order=3,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating, net_class',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        [True, False],
        [MoEKALNConv3DLayer, MoEKAGNConv3DLayer, MoEKACNConv3DLayer],
    ),
)
def test_moekalgcn_conv_3d(
    dropout: float, groups: int, noisy_gating: bool, net_class: nn.Module
):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = net_class(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        spline_order=3,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
    out, _ = net(input_tensor, False)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating',
    itertools.product([0.0, 0.5], [1, 4], [True, False]),
)
def test_moewavkan_conv_1d(dropout: float, groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = MoEWavKANConv1DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        wavelet_type='mexican_hat',
        dropout=dropout,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating',
    itertools.product([0.0, 0.5], [1, 4], [True, False]),
)
def test_moewavkan_conv_2d(dropout: float, groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = MoEWavKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        wavelet_type='mexican_hat',
        dropout=dropout,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)
    out, _ = net(input_tensor, False)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, noisy_gating',
    itertools.product([0.0, 0.5], [1, 4], [True, False]),
)
def test_moewavkan_conv_3d(dropout: float, groups: int, noisy_gating: bool):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 12
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = MoEWavKANConv3DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        wavelet_type='mexican_hat',
        dropout=dropout,
        num_experts=8,
        noisy_gating=noisy_gating,
        k=2,
    )
    out, _ = net(input_tensor, True)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
    out, _ = net(input_tensor, False)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
