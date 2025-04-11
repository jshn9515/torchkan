import itertools

import pytest
import torch
import torch.nn as nn

from torchkan.conv import (
    FastKANConv1DLayer,
    FastKANConv2DLayer,
    FastKANConv3DLayer,
    KABNConv1DLayer,
    KABNConv2DLayer,
    KABNConv3DLayer,
    KACNConv1DLayer,
    KACNConv2DLayer,
    KACNConv3DLayer,
    KAGNConv1DLayer,
    KAGNConv1DLayerV2,
    KAGNConv2DLayer,
    KAGNConv2DLayerV2,
    KAGNConv3DLayer,
    KAGNConv3DLayerV2,
    KAJNConv1DLayer,
    KAJNConv2DLayer,
    KAJNConv3DLayer,
    KALNConv1DLayer,
    KALNConv2DLayer,
    KALNConv3DLayer,
    KANConv1DLayer,
    KANConv2DLayer,
    KANConv3DLayer,
    ReLUKANConv1DLayer,
    ReLUKANConv2DLayer,
    ReLUKANConv3DLayer,
    WavKANConv1DLayer,
    WavKANConv2DLayer,
    WavKANConv3DLayer,
)
from torchkan.conv.wavkan_conv import WaveletType, WaveletVersion


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_kan_conv_1d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = KANConv1DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        spline_order=3,
        grid_size=5,
        grid_range=(-1, 1),
        base_activation=nn.GELU(),
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_kan_conv_2d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = KANConv2DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        spline_order=3,
        grid_size=5,
        grid_range=(-1, 1),
        base_activation=nn.GELU(),
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_kan_conv_3d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = KANConv3DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        spline_order=3,
        grid_size=5,
        grid_range=(-1, 1),
        base_activation=nn.GELU(),
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_fastkan_conv_1d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = FastKANConv1DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        grid_size=5,
        grid_range=(-1, 1),
        base_activation=nn.GELU(),
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_fastkan_conv_2d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = FastKANConv2DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        grid_size=5,
        grid_range=(-1, 1),
        base_activation=nn.GELU(),
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_fastkan_conv_3d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = FastKANConv3DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        grid_size=5,
        grid_range=(-1, 1),
        base_activation=nn.GELU(),
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize(
    'dropout, groups, wavelets, implementation',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
        ['base', 'fast', 'fast_plus_one'],
    ),
)
def test_wavkan_conv_1d(
    dropout: float, groups: int, wavelets: WaveletType, implementation: WaveletVersion
):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = WavKANConv1DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        wavelet_type=wavelets,
        dropout=dropout,
        wavlet_version=implementation,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, wavelets, implementation',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
        ['base', 'fast', 'fast_plus_one'],
    ),
)
def test_wavkan_conv_2d(
    dropout: float, groups: int, wavelets: WaveletType, implementation: WaveletVersion
):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = WavKANConv2DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        wavelet_type=wavelets,
        dropout=dropout,
        wavlet_version=implementation,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, wavelets, implementation',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
        ['base', 'fast'],
    ),
)
def test_wavkan_conv_3d(
    dropout: float, groups: int, wavelets: WaveletType, implementation: WaveletVersion
):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = WavKANConv3DLayer(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        groups=groups,
        padding=padding,
        stride=1,
        dilation=1,
        wavelet_type=wavelets,
        dropout=dropout,
        wavlet_version=implementation,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize(
    'dropout, groups, conv_class',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        [
            KALNConv1DLayer,
            KAGNConv1DLayer,
            KAGNConv1DLayerV2,
            KACNConv1DLayer,
            KAJNConv1DLayer,
            KABNConv1DLayer,
        ],
    ),
)
def test_kalgcn_conv_1d(dropout: float, groups: int, conv_class: nn.Module):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = conv_class(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        spline_order=3,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, conv_class',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        [
            KALNConv2DLayer,
            KAGNConv2DLayer,
            KAGNConv2DLayerV2,
            KACNConv2DLayer,
            KAJNConv2DLayer,
            KABNConv2DLayer,
        ],
    ),
)
def test_kalgcn_conv_2d(dropout: float, groups: int, conv_class: nn.Module):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = conv_class(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        spline_order=3,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, conv_class',
    itertools.product(
        [0.0, 0.5],
        [1, 4],
        [
            KALNConv3DLayer,
            KAGNConv3DLayer,
            KAGNConv3DLayerV2,
            KACNConv3DLayer,
            KAJNConv3DLayer,
            KABNConv3DLayer,
        ],
    ),
)
def test_kalgcn_conv_3d(dropout: float, groups: int, conv_class: nn.Module):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = conv_class(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        spline_order=3,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_relukan_conv_1d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim)
    net = ReLUKANConv1DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        g=5,
        k=3,
        train_ab=True,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_relukan_conv_2d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, in_channels, spatial_dim, spatial_dim)
    net = ReLUKANConv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        g=5,
        k=3,
        train_ab=True,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, out_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout, groups', itertools.product([0.0, 0.5], [1, 4]))
def test_relukan_conv_3d(dropout: float, groups: int):
    batch_size = 6
    in_channels = 4
    out_channels = 16
    spatial_dim = 32
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, in_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = ReLUKANConv3DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        g=5,
        k=3,
        train_ab=True,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        out_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
