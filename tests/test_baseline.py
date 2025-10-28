import itertools

import pytest
import torch

from torchkan.models import (
    EightSimpleConvKACN,
    EightSimpleConvKAGN,
    EightSimpleConvKALN,
    EightSimpleConvKAN,
    EightSimpleConvWavKAN,
    EightSimpleFastConvKAN,
    SimpleConvKACN,
    SimpleConvKAGN,
    SimpleConvKALN,
    SimpleConvKAN,
    SimpleConvWavKAN,
    SimpleFastConvKAN,
)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kan(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = SimpleConvKAN(
        layer_sizes=(2, 4, 8, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kan8(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = EightSimpleConvKAN(
        layer_sizes=(2, 2, 4, 4, 8, 8, 16, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_fast_conv_kan(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand((batch_size, input_dim, spatial_dim, spatial_dim))
    net = SimpleFastConvKAN(
        layer_sizes=(2, 4, 8, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        grid_size=8,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_fast_conv_kan8(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = EightSimpleFastConvKAN(
        layer_sizes=(2, 2, 4, 4, 8, 8, 16, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        grid_size=8,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kaln(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = SimpleConvKALN(
        layer_sizes=(2, 4, 8, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kaln8(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = EightSimpleConvKALN(
        layer_sizes=(2, 2, 4, 4, 8, 8, 16, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kacn(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = SimpleConvKACN(
        layer_sizes=(2, 4, 8, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kacn8(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = EightSimpleConvKACN(
        layer_sizes=(2, 2, 4, 4, 8, 8, 16, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kagn(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = SimpleConvKAGN(
        layer_sizes=(2, 4, 8, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty, affine',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]),
)
def test_simple_conv_kagn8(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float, affine: bool
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = EightSimpleConvKAGN(
        layer_sizes=(2, 2, 4, 4, 8, 8, 16, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5]),
)
def test_simple_wav_conv_kan(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = SimpleConvWavKAN(
        layer_sizes=(2, 4, 8, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'groups, dropout, dropout_linear, l1_penalty',
    itertools.product([1, 2], [0, 0.5], [0, 0.5], [0, 0.5]),
)
def test_simple_wav_conv_kan8(
    groups: int, dropout: float, dropout_linear: float, l1_penalty: float
):
    batch_size = 6
    input_dim = 4
    spatial_dim = 16
    num_classes = 24

    input_tensor = torch.rand(batch_size, input_dim, spatial_dim, spatial_dim)
    net = EightSimpleConvWavKAN(
        layer_sizes=(2, 2, 4, 4, 8, 8, 16, 16),
        num_classes=num_classes,
        input_channels=input_dim,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)
