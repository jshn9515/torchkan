import pytest
import torch

from torchkan.conv import (
    BottleNeckKAGNFocalModulation1D,
    BottleNeckKAGNFocalModulation2D,
    BottleNeckKAGNFocalModulation3D,
)


@pytest.mark.parametrize('dropout', [0.0, 0.5])
def test_sa_bn_kagn_conv_1d(dropout: float):
    batch_size = 6
    num_channels = 16
    spatial_dim = 8
    focal_level = 2
    focal_window = 3
    focal_factor = 2
    use_postln_in_modulation = True
    normalize_modulator = True
    full_kan = True

    input_tensor = torch.rand(batch_size, num_channels, spatial_dim)
    net = BottleNeckKAGNFocalModulation1D(
        num_channels=num_channels,
        focal_window=focal_window,
        focal_level=focal_level,
        focal_factor=focal_factor,
        use_postln_in_modulation=use_postln_in_modulation,
        normalize_modulator=normalize_modulator,
        full_kan=full_kan,
        spline_order=3,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_channels, spatial_dim)


@pytest.mark.parametrize('dropout', [0.0, 0.5])
def test_sa_bn_kagn_conv_2d(dropout: float):
    batch_size = 6
    num_channels = 16
    spatial_dim = 8
    focal_level = 2
    focal_window = 3
    focal_factor = 2
    use_postln_in_modulation = True
    normalize_modulator = True
    full_kan = True

    input_tensor = torch.rand(batch_size, num_channels, spatial_dim, spatial_dim)
    net = BottleNeckKAGNFocalModulation2D(
        num_channels=num_channels,
        focal_window=focal_window,
        focal_level=focal_level,
        focal_factor=focal_factor,
        use_postln_in_modulation=use_postln_in_modulation,
        normalize_modulator=normalize_modulator,
        full_kan=full_kan,
        spline_order=3,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_channels, spatial_dim, spatial_dim)


@pytest.mark.parametrize('dropout', [0.0, 0.5])
def test_sa_bn_kagn_conv_3d(dropout: float):
    batch_size = 6
    num_channels = 16
    spatial_dim = 8
    focal_level = 2
    focal_window = 3
    focal_factor = 2
    use_postln_in_modulation = True
    normalize_modulator = True
    full_kan = True

    input_tensor = torch.rand(
        batch_size, num_channels, spatial_dim, spatial_dim, spatial_dim
    )
    net = BottleNeckKAGNFocalModulation3D(
        num_channels=num_channels,
        focal_window=focal_window,
        focal_level=focal_level,
        focal_factor=focal_factor,
        use_postln_in_modulation=use_postln_in_modulation,
        normalize_modulator=normalize_modulator,
        full_kan=full_kan,
        spline_order=3,
        dropout=dropout,
    )
    out = net(input_tensor)
    assert out.shape == (
        batch_size,
        num_channels,
        spatial_dim,
        spatial_dim,
        spatial_dim,
    )
