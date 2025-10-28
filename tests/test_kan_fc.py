import pytest
import torch
import torch.nn as nn

from torchkan.linear import (
    BernsteinKANLayer,
    BottleNeckGRAMLayer,
    ChebyKANLayer,
    FastKANLayer,
    GRAMLayer,
    JacobiKANLayer,
    KALNLayer,
    KANLayer,
    ReLUKANLayer,
    WavKANLayer,
)
from torchkan.linear.kan import WaveletType


def test_kan_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16

    input_tensor = torch.rand(batch_size, input_dim)
    net = KANLayer(
        input_dim,
        output_dim,
        spline_order=3,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


@pytest.mark.parametrize('use_base_update', [True, False])
def test_fastkan_fc(use_base_update: bool):
    batch_size = 6
    input_dim = 4
    output_dim = 16

    input_tensor = torch.rand(batch_size, input_dim)
    net = FastKANLayer(
        input_dim,
        output_dim,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        use_base_update=use_base_update,
        base_activation=nn.SiLU(),
        spline_weight_init_scale=0.1,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


@pytest.mark.parametrize(
    'wavelet_type', ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']
)
def test_wavkan_fc(wavelet_type: WaveletType):
    batch_size = 6
    input_dim = 4
    output_dim = 16

    input_tensor = torch.rand(batch_size, input_dim)
    net = WavKANLayer(input_dim, output_dim, wavelet_type=wavelet_type)
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_kacn_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16
    spline_order = 3

    input_tensor = torch.rand(batch_size, input_dim)
    net = ChebyKANLayer(input_dim, output_dim, spline_order=spline_order)
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_kagn_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16
    spline_order = 3

    input_tensor = torch.rand(batch_size, input_dim)
    net = GRAMLayer(
        input_dim, output_dim, spline_order=spline_order, base_activation=nn.SiLU()
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_bn_kagn_fc():
    batch_size = 6
    input_dim = 64
    output_dim = 128
    spline_order = 3

    input_tensor = torch.rand(batch_size, input_dim)
    net = BottleNeckGRAMLayer(
        input_dim,
        output_dim,
        spline_order,
        base_activation=nn.SiLU(),
        dim_reduction=4,
        min_internal=8,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_kajn_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16
    spline_order = 3

    input_tensor = torch.rand(batch_size, input_dim)
    net = JacobiKANLayer(input_dim, output_dim, spline_order=spline_order)
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_kaln_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16
    spline_order = 3

    input_tensor = torch.rand(batch_size, input_dim)
    net = KALNLayer(
        input_dim, output_dim, spline_order=spline_order, base_activation=nn.SiLU()
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_kabn_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16
    spline_order = 3

    input_tensor = torch.rand(batch_size, input_dim)
    net = BernsteinKANLayer(
        input_dim, output_dim, spline_order=spline_order, base_activation=nn.SiLU()
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)


def test_relukan_fc():
    batch_size = 6
    input_dim = 4
    output_dim = 16
    g = 5
    k = 3
    train_ab = True

    input_tensor = torch.rand(batch_size, input_dim)
    net = ReLUKANLayer(input_dim, output_dim, g=g, k=k, train_ab=train_ab)
    out = net(input_tensor)
    assert out.shape == (batch_size, output_dim)
