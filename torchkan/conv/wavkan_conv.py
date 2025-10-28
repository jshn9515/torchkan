"""
Based on https://github.com/zavareh1/Wav-KAN
This is a sample code for the simulations of the paper:
Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

https://arxiv.org/abs/2405.12832
and also available at:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
We used efficient KAN notation and some part of the code: https://github.com/Blealtan/efficient-kan
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from torchkan.utils.typing import (
    Activation,
    Padding1D,
    Padding2D,
    Padding3D,
    PaddingND,
    Size2D,
    Size3D,
    SizeND,
    WaveletType,
    WaveletVersion,
)


class WaveletConvND(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        wavelet_type: WaveletType = 'mexican_hat',
    ):
        super().__init__()

        shapes = (1, out_channels, in_channels) + (1,) * ndim

        self.scale = nn.Parameter(torch.ones(shapes))
        self.translation = nn.Parameter(torch.zeros(shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.in_channels = in_channels
        self.out_channels = out_channels

        wavelet_weights = conv_class(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.wavelet_weights = nn.ModuleList([wavelet_weights] * out_channels)

        self.wavelet_out = conv_class(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=False,
        )

        for conv_layer in self.wavelet_weights:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )
        nn.init.kaiming_uniform_(
            torch.as_tensor(self.wavelet_out.weight), nonlinearity='linear'
        )

    @staticmethod
    def _forward_mexican_hat(x: Tensor) -> Tensor:
        term1 = (x**2) - 1
        term2 = torch.exp(-0.5 * x**2)
        wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
        return wavelet

    @staticmethod
    def _forward_morlet(x: Tensor) -> Tensor:
        omega0 = 5.0  # Central frequency
        real = torch.cos(omega0 * x)
        envelope = torch.exp(-0.5 * x**2)
        wavelet = envelope * real
        return wavelet

    @staticmethod
    def _forward_dog(x: Tensor) -> Tensor:
        return -x * torch.exp(-0.5 * x**2)

    @staticmethod
    def _forward_meyer(x: Tensor) -> Tensor:
        v = torch.abs(x)

        def meyer_aux(v: Tensor) -> Tensor:
            return torch.where(
                v <= 1 / 2,
                torch.ones_like(v),
                torch.where(
                    v >= 1,
                    torch.zeros_like(v),
                    torch.cos(math.pi / 2 * nu(2 * v - 1)),
                ),
            )

        def nu(t: Tensor) -> Tensor:
            return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)

        # Meyer wavelet calculation using the auxiliary function
        wavelet = torch.sin(math.pi * v) * meyer_aux(v)
        return wavelet

    def _forward_shannon(self, x: Tensor) -> Tensor:
        sinc = torch.sinc(x / math.pi)  # sinc(x) = sin(pi*x) / (pi*x)

        shape = (1, 1, x.size(2)) + (1,) * self.ndim
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = torch.hamming_window(
            x.size(2), periodic=False, dtype=x.dtype, device=x.device
        ).view(shape)
        # Shannon wavelet is the product of the sinc function and the window
        wavelet = sinc * window
        return wavelet

    def forward(self, x: Tensor) -> Tensor:
        x_expanded = torch.unsqueeze(x, dim=1)
        x_scaled = (x_expanded - self.translation) / self.scale

        match self.wavelet_type:
            case 'mexican_hat':
                wavelet = self._forward_mexican_hat(x_scaled)
            case 'morlet':
                wavelet = self._forward_morlet(x_scaled)
            case 'dog':
                wavelet = self._forward_dog(x_scaled)
            case 'meyer':
                wavelet = self._forward_meyer(x_scaled)
            case 'shannon':
                wavelet = self._forward_shannon(x_scaled)
            case _:
                raise ValueError(f'Unsupported wavelet type: {self.wavelet_type}')

        wavelet_x = torch.split(wavelet, 1, dim=1)
        output = []
        for group_index, x in enumerate(wavelet_x):
            x = torch.squeeze(x, dim=1)
            y = self.wavelet_weights[group_index](x)
            output.append(y)
        y = torch.concat(output, dim=1)
        y = self.wavelet_out(y)
        return y


class WaveletConvNDFastPlusOne(WaveletConvND):
    def __init__(
        self,
        conv_class: type[nn.Module],
        conv_class_d_plus_one: type[nn.Module],
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        wavelet_type: WaveletType = 'mexican_hat',
    ):
        super(WaveletConvND, self).__init__()
        assert ndim < 3, 'fast_plus_one version supports only 1D and 2D convs.'

        shapes = (1, out_channels, in_channels) + (1,) * ndim

        self.scale = nn.Parameter(torch.ones(shapes))
        self.translation = nn.Parameter(torch.zeros(shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size_plus = (
            (in_channels,) + kernel_size
            if isinstance(kernel_size, tuple)
            else (in_channels,) + (kernel_size,) * ndim
        )
        stride_plus = (
            (1,) + stride if isinstance(stride, tuple) else (1,) + (stride,) * ndim
        )
        padding_plus = (
            (0,) + padding if isinstance(padding, tuple) else (0,) + (padding,) * ndim
        )
        dilation_plus = (
            (1,) + dilation
            if isinstance(dilation, tuple)
            else (1,) + (dilation,) * ndim
        )

        self.wavelet_weights = conv_class_d_plus_one(
            out_channels,
            out_channels,
            kernel_size=kernel_size_plus,
            stride=stride_plus,
            padding=padding_plus,
            dilation=dilation_plus,
            groups=out_channels,
            bias=False,
        )

        self.wavelet_out = conv_class(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(
            torch.as_tensor(self.wavelet_weights.weight), nonlinearity='linear'
        )
        nn.init.kaiming_uniform_(
            torch.as_tensor(self.wavelet_out.weight), nonlinearity='linear'
        )

    def forward(self, x: Tensor) -> Tensor:
        x_expanded = torch.unsqueeze(x, dim=1)
        x_scaled = (x_expanded - self.translation) / self.scale

        match self.wavelet_type:
            case 'mexican_hat':
                wavelet = self._forward_mexican_hat(x_scaled)
            case 'morlet':
                wavelet = self._forward_morlet(x_scaled)
            case 'dog':
                wavelet = self._forward_dog(x_scaled)
            case 'meyer':
                wavelet = self._forward_meyer(x_scaled)
            case 'shannon':
                wavelet = self._forward_shannon(x_scaled)
            case _:
                raise ValueError(f'Unsupported wavelet type: {self.wavelet_type}')

        y = self.wavelet_weights(wavelet)
        y = torch.squeeze(y, dim=2)
        y = self.wavelet_out(y)
        return y


class WaveletConvNDFast(WaveletConvND):
    def __init__(
        self,
        conv_class: type[nn.Module],
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        wavelet_type: WaveletType = 'mexican_hat',
    ):
        super(WaveletConvND, self).__init__()
        shapes = (1, out_channels, in_channels) + (1,) * ndim

        self.scale = nn.Parameter(torch.ones(shapes))
        self.translation = nn.Parameter(torch.zeros(shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.wavelet_weights = conv_class(
            in_channels * out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=False,
        )

        self.wavelet_out = conv_class(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(
            torch.as_tensor(self.wavelet_weights.weight), nonlinearity='linear'
        )
        nn.init.kaiming_uniform_(
            torch.as_tensor(self.wavelet_out.weight), nonlinearity='linear'
        )

    def forward(self, x: Tensor) -> Tensor:
        x_expanded = torch.unsqueeze(x, dim=1)
        x_scaled = (x_expanded - self.translation) / self.scale

        match self.wavelet_type:
            case 'mexican_hat':
                wavelet = self._forward_mexican_hat(x_scaled)
            case 'morlet':
                wavelet = self._forward_morlet(x_scaled)
            case 'dog':
                wavelet = self._forward_dog(x_scaled)
            case 'meyer':
                wavelet = self._forward_meyer(x_scaled)
            case 'shannon':
                wavelet = self._forward_shannon(x_scaled)
            case _:
                raise ValueError(f'Unsupported wavelet type: {self.wavelet_type}')

        y = self.wavelet_weights(wavelet.flatten(1, 2))
        y = self.wavelet_out(y)
        return y


class WavKANConvNDLayer(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        conv_class_plus_one: type[nn.Module],
        norm_class: type[nn.Module],
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        base_activation: Activation = nn.SiLU(),
        dropout: float = 0.0,
        wavelet_version: WaveletVersion = 'base',
        wavelet_type: WaveletType = 'mexican_hat',
        **norm_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.norm_kwargs = norm_kwargs
        self.wavelet_type = wavelet_type

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        base_conv = conv_class(
            in_channels=in_channels // groups,
            out_channels=out_channels // groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.base_conv = nn.ModuleList([base_conv] * groups)

        match wavelet_version:
            case 'base':
                wavelet_conv = WaveletConvND(
                    conv_class=conv_class,
                    ndim=ndim,
                    in_channels=in_channels // groups,
                    out_channels=out_channels // groups,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    wavelet_type=wavelet_type,
                )
            case 'fast':
                wavelet_conv = WaveletConvNDFast(
                    conv_class=conv_class,
                    ndim=ndim,
                    in_channels=in_channels // groups,
                    out_channels=out_channels // groups,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    wavelet_type=wavelet_type,
                )
            case 'fast_plus_one':
                if isinstance(padding, str):
                    raise ValueError(
                        'fast_plus_one version does not support string padding.'
                    )
                wavelet_conv = WaveletConvNDFastPlusOne(
                    conv_class=conv_class,
                    conv_class_d_plus_one=conv_class_plus_one,
                    ndim=ndim,
                    in_channels=in_channels // groups,
                    out_channels=out_channels // groups,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    wavelet_type=wavelet_type,
                )

        self.wavelet_conv = nn.ModuleList([wavelet_conv] * groups)

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        self.base_activation = base_activation

    def forward_wavkan(self, x: Tensor, group_index: int) -> Tensor:
        # You may like test the cases like Spl-KAN
        base_output = self.base_conv[group_index](self.base_activation(x))

        if self.dropout:
            x = self.dropout(x)

        wavelet_output = self.wavelet_conv[group_index](x)
        combined_output = wavelet_output + base_output

        # Apply batch normalization
        return self.layer_norm[group_index](combined_output)

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_wavkan(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class WavKANConv3DLayer(WavKANConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Padding3D = 0,
        dilation: Size3D = 1,
        groups: int = 1,
        dropout: float = 0.0,
        wavelet_type: WaveletType = 'mexican_hat',
        wavelet_version: WaveletVersion = 'fast',
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv3d,
            conv_class_plus_one=nn.Identity,
            norm_class=nn.BatchNorm3d,
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            wavelet_type=wavelet_type,
            wavelet_version=wavelet_version,
            **norm_kwargs,
        )


class WavKANConv2DLayer(WavKANConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D,
        stride: Size2D = 1,
        padding: Padding2D = 0,
        dilation: Size2D = 1,
        groups: int = 1,
        dropout: float = 0.0,
        wavelet_type: WaveletType = 'mexican_hat',
        wavelet_version: WaveletVersion = 'fast',
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv2d,
            conv_class_plus_one=nn.Conv3d,
            norm_class=nn.BatchNorm2d,
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            wavelet_type=wavelet_type,
            wavelet_version=wavelet_version,
            **norm_kwargs,
        )


class WavKANConv1DLayer(WavKANConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Padding1D = 0,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        wavelet_type: WaveletType = 'mexican_hat',
        wavelet_version: WaveletVersion = 'fast',
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv1d,
            conv_class_plus_one=nn.Conv2d,
            norm_class=nn.BatchNorm1d,
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            wavelet_type=wavelet_type,
            wavelet_version=wavelet_version,
            **norm_kwargs,
        )
