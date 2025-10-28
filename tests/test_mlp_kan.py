import itertools

import pytest
import torch
import torch.nn as nn

from torchkan.linear import (
    KABN,
    KACN,
    KAGN,
    KAJN,
    KALN,
    KAN,
    BottleNeckKAGN,
    FastKAN,
    ReLUKAN,
    WavKAN,
)
from torchkan.linear.kan import WaveletType


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_kan(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = KAN(
        hidden_layers,
        spline_order=3,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_fast_kan(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = FastKAN(
        hidden_layers,
        grid_size=5,
        base_activation=nn.GELU(),
        grid_range=(-1, 1),
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay, wavelet_type',
    itertools.product(
        [0.0, 0.5],
        [True, False],
        [0, 0.1],
        ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
    ),
)
def test_wav_kan(
    dropout: float, first_dropout: bool, l1_decay: float, wavelet_type: WaveletType
):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = WavKAN(
        hidden_layers,
        wavelet_type=wavelet_type,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_kaln(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    spline_order = 3
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = KALN(
        hidden_layers,
        base_activation=nn.GELU(),
        spline_order=spline_order,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_kajn(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    spline_order = 3
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = KAJN(
        hidden_layers,
        base_activation=nn.GELU(),
        spline_order=spline_order,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_kagn(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    spline_order = 3
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = KAGN(
        hidden_layers,
        base_activation=nn.GELU(),
        spline_order=spline_order,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay, dim_reduction, min_internal',
    itertools.product(
        [0.0, 0.5], [True, False], [0, 0.1], [2, 4, 8, 16], [4, 8, 16, 32]
    ),
)
def test_bn_kagn(
    dropout: float,
    first_dropout: bool,
    l1_decay: float,
    dim_reduction: float,
    min_internal: int,
):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    spline_order = 3
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = BottleNeckKAGN(
        hidden_layers,
        base_activation=nn.SiLU(),
        spline_order=spline_order,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
        dim_reduction=dim_reduction,
        min_internal=min_internal,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_kacn(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    spline_order = 3
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = KACN(
        hidden_layers,
        spline_order=spline_order,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_kabn(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    spline_order = 3
    num_classes = 128

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = KABN(
        hidden_layers,
        spline_order=spline_order,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    'dropout, first_dropout, l1_decay',
    itertools.product([0.0, 0.5], [True, False], [0, 0.1]),
)
def test_relukan(dropout: float, first_dropout: bool, l1_decay: float):
    batch_size = 6
    input_dim = 32
    hidden_dim = 64
    num_classes = 128

    g = 5
    k = 3
    train_ab = True

    input_tensor = torch.rand(batch_size, input_dim)
    hidden_layers = (input_dim, hidden_dim, num_classes)

    net = ReLUKAN(
        hidden_layers,
        g=g,
        k=k,
        train_ab=train_ab,
        dropout=dropout,
        l1_decay=l1_decay,
        first_dropout=first_dropout,
    )
    out = net(input_tensor)
    assert out.shape == (batch_size, num_classes)
