import itertools

import pytest
import torch

from torchkan.conv import (
    BottleNeckKAGNConv1DLayer,
    BottleNeckKAGNConv2DLayer,
    BottleNeckKAGNConv3DLayer,
    MoEBottleNeckKAGNConv1DLayer,
    MoEBottleNeckKAGNConv2DLayer,
    MoEBottleNeckKAGNConv3DLayer,
)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_kagn_conv_1d(dropout, groups):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    conv = BottleNeckKAGNConv1DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        degree=3,
    )
    out = conv(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_kagn_conv_2d(dropout, groups):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    conv = BottleNeckKAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        degree=3,
    )
    out = conv(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_kagn_conv_3d(dropout, groups):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim)
    conv = BottleNeckKAGNConv3DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        degree=3,
    )
    out = conv(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_moe_kagn_conv_1d(dropout, groups):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    conv = MoEBottleNeckKAGNConv1DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        degree=3,
    )
    out, _ = conv(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, pregate', itertools.product([0.0, 0.5], [1, 4], [True, False])
)
def test_moe_kagn_conv_2d(dropout, groups, pregate):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    conv = MoEBottleNeckKAGNConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        degree=3,
        pregate=pregate,
    )
    out, _ = conv(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_moe_kagn_conv_3d(dropout, groups):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim)
    conv = MoEBottleNeckKAGNConv3DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        dropout=dropout,
        degree=3,
    )
    out, _ = conv(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim, spatial_dim)
