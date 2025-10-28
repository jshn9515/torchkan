import itertools

import pytest
import torch

from torchkan.conv import (
    BottleNeckSelfKAGNtention1D,
    BottleNeckSelfKAGNtention2D,
    BottleNeckSelfKAGNtention3D,
    SelfKAGNtention1D,
    SelfKAGNtention2D,
    SelfKAGNtention3D,
)


@pytest.mark.parametrize(
    'dropout, groups, inner_projection',
    itertools.product([0.0, 0.5], [1, 4], [None, 8]),
)
def test_sa_kagn_conv_1d(dropout: float, groups: int, inner_projection: int):
    batch_size = 6
    num_features = 16
    spatial_dim = 8
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, num_features, spatial_dim)
    net = SelfKAGNtention1D(
        input_dim=num_features,
        inner_projection=inner_projection,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        dropout=dropout,
        spline_order=3,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_features, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, inner_projection',
    itertools.product([0.0, 0.5], [1, 4], [None, 8]),
)
def test_sa_kagn_conv_2d(dropout: float, groups: int, inner_projection: int):
    batch_size = 6
    num_features = 16
    spatial_dim = 8
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, num_features, spatial_dim, spatial_dim)
    net = SelfKAGNtention2D(
        input_dim=num_features,
        inner_projection=inner_projection,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        dropout=dropout,
        spline_order=3,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_features, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, inner_projection',
    itertools.product([0.0, 0.5], [1, 4], [None, 8]),
)
def test_sa_kagn_conv_3d(dropout: float, groups: int, inner_projection: int):
    batch_size = 6
    num_features = 16
    spatial_dim = 16
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, num_features, spatial_dim, spatial_dim, spatial_dim
    )
    net = SelfKAGNtention3D(
        input_dim=num_features,
        inner_projection=inner_projection,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        dropout=dropout,
        spline_order=3,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        num_features,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )


@pytest.mark.parametrize(
    'dropout, groups, inner_projection',
    itertools.product([0.0, 0.5], [1, 4], [None, 8]),
)
def test_sa_bn_kagn_conv_1d(dropout: float, groups: int, inner_projection: int):
    batch_size = 6
    num_features = 16
    spatial_dim = 8
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, num_features, spatial_dim)
    net = BottleNeckSelfKAGNtention1D(
        input_dim=num_features,
        inner_projection=inner_projection,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        dropout=dropout,
        spline_order=3,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_features, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, inner_projection',
    itertools.product([0.0, 0.5], [1, 4], [None, 8]),
)
def test_sa_bn_kagn_conv_2d(dropout: float, groups: int, inner_projection: int):
    batch_size = 6
    num_features = 16
    spatial_dim = 8
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(batch_size, num_features, spatial_dim, spatial_dim)
    net = BottleNeckSelfKAGNtention2D(
        input_dim=num_features,
        inner_projection=inner_projection,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        dropout=dropout,
        spline_order=3,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_features, spatial_dim, spatial_dim)


@pytest.mark.parametrize(
    'dropout, groups, inner_projection',
    itertools.product([0.0, 0.5], [1, 4], [None, 8]),
)
def test_sa_bn_kagn_conv_3d(dropout: float, groups: int, inner_projection: int):
    batch_size = 6
    num_features = 16
    spatial_dim = 8
    kernel_size = 3
    padding = 'same'

    input_tensor = torch.rand(
        batch_size, num_features, spatial_dim, spatial_dim, spatial_dim
    )
    net = BottleNeckSelfKAGNtention3D(
        input_dim=num_features,
        inner_projection=inner_projection,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
        dropout=dropout,
        spline_order=3,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        num_features,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
