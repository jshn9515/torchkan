import math
import random
from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .fastkan_conv import FastKANConv1DLayer, FastKANConv2DLayer, FastKANConv3DLayer
from .kabn_conv import KABNConv1DLayer, KABNConv2DLayer, KABNConv3DLayer
from .kacn_conv import KACNConv1DLayer, KACNConv2DLayer, KACNConv3DLayer
from .kagn_bottleneck_conv import (
    BottleNeckKAGNConv1DLayer,
    BottleNeckKAGNConv2DLayer,
    BottleNeckKAGNConv3DLayer,
    MoEBottleNeckKAGNConv1DLayer,
    MoEBottleNeckKAGNConv2DLayer,
    MoEBottleNeckKAGNConv3DLayer,
)
from .kagn_conv import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from .kagn_conv_v2 import KAGNConv1DLayerV2, KAGNConv2DLayerV2, KAGNConv3DLayerV2
from .kajn_conv import KAJNConv1DLayer, KAJNConv2DLayer, KAJNConv3DLayer
from .kaln_conv import KALNConv1DLayer, KALNConv2DLayer, KALNConv3DLayer
from .kan_conv import KANConv1DLayer, KANConv2DLayer, KANConv3DLayer
from .relukan_bottleneck_conv import (
    BottleNeckReLUKANConv1DLayer,
    BottleNeckReLUKANConv2DLayer,
    BottleNeckReLUKANConv3DLayer,
)
from .relukan_conv import ReLUKANConv1DLayer, ReLUKANConv2DLayer, ReLUKANConv3DLayer
from .wavkan_conv import WavKANConv1DLayer, WavKANConv2DLayer, WavKANConv3DLayer

PaddingType = Literal['valid', 'same']


def init_2d_freqs(
    dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
) -> Tensor:
    freqs_x = []
    freqs_y = []
    mag = 1 / (
        theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim)
    )
    for _ in range(num_heads):
        angles = random.random() * 2 * math.pi if rotate else 0
        fx = torch.concat(
            [mag * math.cos(angles), mag * math.cos(math.pi / 2 + angles)]
        )
        fy = torch.concat(
            [mag * math.sin(angles), mag * math.sin(math.pi / 2 + angles)]
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def init_t_xy(end_x: float, end_y: float) -> tuple[Tensor, Tensor]:
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.floor(t / end_x)
    return t_x, t_y


def compute_mixed_cis(
    freqs: Tensor, t_x: Tensor, t_y: Tensor, num_heads: int
) -> Tensor:
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.amp.autocast('cuda', enabled=False):
        freqs_x = (
            (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_y = (
            (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0) -> Tensor:
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim)
    )

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.concat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: Tensor, xk: Tensor, freqs_cis: Tensor
) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2).contiguous())
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2).contiguous())
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    xq_out = xq_out.type_as(xq).to(xq.device)
    xk_out = xk_out.type_as(xk).to(xk.device)
    return xq_out, xk_out


class SelfKANtentionND(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_kan_layer: type[nn.Module],
        norm_layer: type[nn.Module] | None = None,
        inner_projection: int | None = None,
        **kwargs,
    ):
        super(SelfKANtentionND, self).__init__()
        self.ndim = None
        self.input_dim = input_dim
        self.norm_layer = norm_layer
        affine = True
        if 'affine' in kwargs:
            affine = kwargs.pop('affine')

        kernel_size = kwargs.pop('kernel_size')

        if conv_kan_layer in [
            FastKANConv1DLayer,
            KANConv1DLayer,
            KALNConv1DLayer,
            KACNConv1DLayer,
            KAGNConv1DLayer,
            WavKANConv1DLayer,
            KAJNConv1DLayer,
            KABNConv1DLayer,
            BottleNeckKAGNConv1DLayer,
            MoEBottleNeckKAGNConv1DLayer,
            ReLUKANConv1DLayer,
            BottleNeckReLUKANConv1DLayer,
        ]:
            self.ndim = 1
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm1d(input_dim)
            else:
                self.norm_layer = self.norm_layer(input_dim, affine=affine)
        elif conv_kan_layer in [
            FastKANConv2DLayer,
            KANConv2DLayer,
            KALNConv2DLayer,
            KACNConv2DLayer,
            KAGNConv2DLayer,
            WavKANConv2DLayer,
            KAJNConv2DLayer,
            KABNConv2DLayer,
            BottleNeckKAGNConv2DLayer,
            MoEBottleNeckKAGNConv2DLayer,
            ReLUKANConv2DLayer,
            BottleNeckReLUKANConv2DLayer,
        ]:
            self.ndim = 2
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm2d(input_dim)
            else:
                self.norm_layer = self.norm_layer(input_dim, affine=affine)
        elif conv_kan_layer in [
            FastKANConv3DLayer,
            KANConv3DLayer,
            KALNConv3DLayer,
            KACNConv3DLayer,
            KAGNConv3DLayer,
            WavKANConv3DLayer,
            KAJNConv3DLayer,
            KABNConv3DLayer,
            BottleNeckKAGNConv3DLayer,
            MoEBottleNeckKAGNConv3DLayer,
            ReLUKANConv3DLayer,
            BottleNeckReLUKANConv3DLayer,
        ]:
            self.ndim = 3
            if self.norm_layer is None:
                self.norm_layer = nn.BatchNorm3d(input_dim)
            else:
                self.norm_layer = self.norm_layer(input_dim, affine=affine)

        assert self.norm_layer is not None
        assert self.ndim is not None, 'Unsupported conv_kan layer!'

        self.inner_proj = None
        self.outer_proj = None
        if inner_projection is not None:
            if self.ndim == 1:
                self.inner_proj = nn.Conv1d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv1d(inner_projection, input_dim, 1)
            if self.ndim == 2:
                self.inner_proj = nn.Conv2d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv2d(inner_projection, input_dim, 1)
            if self.ndim == 3:
                self.inner_proj = nn.Conv3d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv3d(inner_projection, input_dim, 1)

        num_channels = input_dim if inner_projection is None else inner_projection
        self.num_channels = num_channels

        self.proj_k = conv_kan_layer(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.proj_q = conv_kan_layer(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.proj_v = conv_kan_layer(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        input_shape = v.size()
        m_batchsize = input_shape[0]
        total_pixels = math.prod(input_shape[2:])

        proj_query = q.view(m_batchsize, -1, total_pixels).permute(0, 2, 1)  # B X CX(N)
        proj_key = k.view(m_batchsize, -1, total_pixels)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = v.view(m_batchsize, -1, total_pixels)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(input_shape)

        return out

    def forward(self, x: Tensor) -> Tensor:
        if self.inner_proj is not None:
            att = self.inner_proj(x)
        else:
            att = x

        q = self.proj_q(att)
        k = self.proj_k(att)
        v = self.proj_v(att)

        att = self.attention(q, k, v)

        if self.inner_proj is not None and self.outer_proj is not None:
            att = self.outer_proj(att)

        assert isinstance(self.norm_layer, nn.Module)
        return self.norm_layer(self.gamma * x + att)


class RoPESelfKANtentionND(SelfKANtentionND):
    def __init__(
        self, *args, rope_theta: float = 10.0, rope_mixed: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rope_mixed = rope_mixed

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=1)

            freqs = init_2d_freqs(
                dim=self.num_channels, num_heads=1, theta=rope_theta, rotate=True
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)

            t_x, t_y = init_t_xy(end_x=14, end_y=14)
            self.freqs_t_x = t_x
            self.freqs_t_y = t_y
        else:
            self.compute_cis = partial(
                compute_axial_cis, dim=self.num_channels, theta=rope_theta
            )
            freqs_cis = self.compute_cis(end_x=14, end_y=14)
            self.freqs_cis = freqs_cis

    def attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        input_shape = v.size()
        m_batchsize = input_shape[0]
        total_pixels = math.prod(input_shape[2:])

        proj_query = q.view(m_batchsize, -1, total_pixels).permute(
            0, 2, 1
        )  # B X C X(N)
        proj_key = k.view(m_batchsize, -1, total_pixels).permute(
            0, 2, 1
        )  # B X C x (*W*H)

        # Apply rotary position embedding
        w = h = math.sqrt(total_pixels - 1)
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if self.freqs_t_x.shape[0] != total_pixels - 1:
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = t_x.to(q.device), t_y.to(q.device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            freqs_cis = self.freqs_cis
            if self.freqs_cis.shape[0] != total_pixels - 1:
                freqs_cis = self.compute_cis(w, h)  # type: ignore
            freqs_cis = freqs_cis.to(q.device)
        freqs_cis = freqs_cis[:, : total_pixels - 1]

        proj_query = torch.unsqueeze(proj_query, dim=1)
        proj_key = torch.unsqueeze(proj_key, dim=1)
        proj_query[:, :, 1:], proj_key[:, :, 1:] = apply_rotary_emb(
            proj_query[:, :, 1:], proj_key[:, :, 1:], freqs_cis=freqs_cis
        )
        proj_query = proj_query[:, 0]
        proj_key = proj_key[:, 0].permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = v.view(m_batchsize, -1, total_pixels)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(*input_shape)

        return out


class SelfKAGNtention3D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(SelfKAGNtention3D, self).__init__(
            conv_kan_layer=KAGNConv3DLayer,
            norm_layer=nn.BatchNorm3d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class SelfKAGNtention2D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(SelfKAGNtention2D, self).__init__(
            conv_kan_layer=KAGNConv2DLayer,
            norm_layer=nn.BatchNorm2d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class SelfKAGNtention1D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(SelfKAGNtention1D, self).__init__(
            conv_kan_layer=KAGNConv1DLayer,
            norm_layer=nn.BatchNorm1d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckSelfKAGNtention3D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(BottleNeckSelfKAGNtention3D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv3DLayer,
            norm_layer=nn.BatchNorm3d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckSelfKAGNtention2D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(BottleNeckSelfKAGNtention2D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv2DLayer,
            norm_layer=nn.BatchNorm2d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckSelfKAGNtention1D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(BottleNeckSelfKAGNtention1D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv1DLayer,
            norm_layer=nn.BatchNorm1d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class RoPEBottleNeckSelfKAGNtention3D(RoPESelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
        **kwargs,
    ):
        super(RoPEBottleNeckSelfKAGNtention3D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv3DLayer,
            norm_layer=nn.BatchNorm3d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            spline_order=spline_order,
            groups=groups,
            padding=padding,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            **kwargs,
        )


class RoPEBottleNeckSelfKAGNtention2D(RoPESelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
        **kwargs,
    ):
        super(RoPEBottleNeckSelfKAGNtention2D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv2DLayer,
            norm_layer=nn.BatchNorm2d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            spline_order=spline_order,
            groups=groups,
            padding=padding,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            **kwargs,
        )


class RoPEBottleNeckSelfKAGNtention1D(RoPESelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
        **kwargs,
    ):
        super(RoPEBottleNeckSelfKAGNtention1D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv1DLayer,
            norm_layer=nn.BatchNorm1d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            spline_order=spline_order,
            groups=groups,
            padding=padding,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            **kwargs,
        )


class SelfReLUKANtention3D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(SelfReLUKANtention3D, self).__init__(
            conv_kan_layer=ReLUKANConv3DLayer,
            norm_layer=nn.BatchNorm3d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            g=g,
            k=k,
            train_ab=train_ab,
            dropout=dropout,
            **kwargs,
        )


class SelfReLUKANtention2D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(SelfReLUKANtention2D, self).__init__(
            conv_kan_layer=ReLUKANConv2DLayer,
            norm_layer=nn.BatchNorm2d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            g=g,
            k=k,
            train_ab=train_ab,
            dropout=dropout,
            **kwargs,
        )


class SelfReLUKANtention1D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(SelfReLUKANtention1D, self).__init__(
            conv_kan_layer=ReLUKANConv1DLayer,
            norm_layer=nn.BatchNorm1d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            g=g,
            k=k,
            train_ab=train_ab,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckSelfReLUKANtention3D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: PaddingType | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(BottleNeckSelfReLUKANtention3D, self).__init__(
            conv_kan_layer=BottleNeckReLUKANConv3DLayer,
            norm_layer=nn.BatchNorm3d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            g=g,
            k=k,
            train_ab=train_ab,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckSelfReLUKANtention2D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: PaddingType | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(BottleNeckSelfReLUKANtention2D, self).__init__(
            conv_kan_layer=BottleNeckReLUKANConv2DLayer,
            norm_layer=nn.BatchNorm2d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            g=g,
            k=k,
            train_ab=train_ab,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckSelfReLUKANtention1D(SelfKANtentionND):
    def __init__(
        self,
        input_dim: int,
        inner_projection: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: PaddingType | int = 0,
        dilation: int = 1,
        groups: int = 1,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(BottleNeckSelfReLUKANtention1D, self).__init__(
            conv_kan_layer=BottleNeckReLUKANConv1DLayer,
            norm_layer=nn.BatchNorm1d,
            input_dim=input_dim,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            g=g,
            k=k,
            train_ab=train_ab,
            dropout=dropout,
            **kwargs,
        )


class KANFocalModulationND(nn.Module):
    def __init__(
        self,
        conv_kan_layer: type[nn.Module],
        focal_norm_layer: dict[str, Any],
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False,
        full_kan: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        conv_kan_layer_focal = conv_kan_layer
        if conv_kan_layer in [
            FastKANConv1DLayer,
            KANConv1DLayer,
            KALNConv1DLayer,
            KACNConv1DLayer,
            KAGNConv1DLayer,
            WavKANConv1DLayer,
            KAJNConv1DLayer,
            KABNConv1DLayer,
            BottleNeckKAGNConv1DLayer,
            MoEBottleNeckKAGNConv1DLayer,
            ReLUKANConv1DLayer,
            BottleNeckReLUKANConv1DLayer,
        ]:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.ndim = 1
            if conv_kan_layer in [BottleNeckKAGNConv1DLayer, KAGNConv1DLayer]:
                conv_kan_layer_focal = KAGNConv1DLayerV2
        elif conv_kan_layer in [
            FastKANConv2DLayer,
            KANConv2DLayer,
            KALNConv2DLayer,
            KACNConv2DLayer,
            KAGNConv2DLayer,
            WavKANConv2DLayer,
            KAJNConv2DLayer,
            KABNConv2DLayer,
            BottleNeckKAGNConv2DLayer,
            MoEBottleNeckKAGNConv2DLayer,
            ReLUKANConv2DLayer,
            BottleNeckReLUKANConv2DLayer,
        ]:
            self.ndim = 2
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            if conv_kan_layer in [BottleNeckKAGNConv2DLayer, KAGNConv2DLayer]:
                conv_kan_layer_focal = KAGNConv2DLayerV2
        elif conv_kan_layer in [
            FastKANConv3DLayer,
            KANConv3DLayer,
            KALNConv3DLayer,
            KACNConv3DLayer,
            KAGNConv3DLayer,
            WavKANConv3DLayer,
            KAJNConv3DLayer,
            KABNConv3DLayer,
            BottleNeckKAGNConv3DLayer,
            MoEBottleNeckKAGNConv3DLayer,
            ReLUKANConv3DLayer,
            BottleNeckReLUKANConv3DLayer,
        ]:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.ndim = 3
            if conv_kan_layer in [BottleNeckKAGNConv3DLayer, KAGNConv3DLayer]:
                conv_kan_layer_focal = KAGNConv3DLayerV2

        if full_kan:
            self.f = conv_kan_layer(
                in_channels=num_channels,
                out_channels=2 * num_channels + (self.focal_level + 1),
                kernel_size=1,
                padding=0,
                **kwargs,
            )
            self.h = conv_kan_layer(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=1,
                padding=0,
                **kwargs,
            )
        else:
            if self.ndim == 1:
                self.f = nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=2 * num_channels + (self.focal_level + 1),
                    kernel_size=1,
                )
                self.h = nn.Conv1d(
                    in_channels=num_channels, out_channels=num_channels, kernel_size=1
                )
            elif self.ndim == 2:
                self.f = nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=2 * num_channels + (self.focal_level + 1),
                    kernel_size=1,
                )
                self.h = nn.Conv2d(
                    in_channels=num_channels, out_channels=num_channels, kernel_size=1
                )
            else:
                self.f = nn.Conv3d(
                    in_channels=num_channels,
                    out_channels=2 * num_channels + (self.focal_level + 1),
                    kernel_size=1,
                )
                self.h = nn.Conv3d(
                    in_channels=num_channels, out_channels=num_channels, kernel_size=1
                )

        self.proj = conv_kan_layer(
            in_channels=num_channels, out_channels=num_channels, kernel_size=1, **kwargs
        )

        self.focal_layers = nn.ModuleList([])
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                conv_kan_layer_focal(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=num_channels,
                    **kwargs,
                )
            )
            self.kernel_sizes.append(kernel_size)

        if use_postln_in_modulation:
            self.norm_layer = focal_norm_layer['layer'](
                num_channels, **focal_norm_layer['params']
            )

    def forward(self, x: Tensor) -> Tensor:
        channels = x.size(1)

        # pre linear projection
        x = self.f(x)
        q, ctx, self.gates = torch.split(
            tensor=x,
            split_size_or_sections=[channels, channels, self.focal_level + 1],
            dim=1,
        )

        # context aggregation
        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, level : level + 1]
        ctx_global = self.global_pool(ctx_all)
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level :]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        modulator = self.h(ctx_all)
        x_out = q * modulator
        if self.use_postln_in_modulation:
            x_out = self.norm_layer(x_out)

        # post projection
        x_out = self.proj(x_out)
        return x_out


class BottleNeckKAGNFocalModulation3D(KANFocalModulationND):
    def __init__(
        self,
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        focal_norm_layer = {'layer': nn.BatchNorm3d, 'params': kwargs}

        super(BottleNeckKAGNFocalModulation3D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv3DLayer,
            focal_norm_layer=focal_norm_layer,
            num_channels=num_channels,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckKAGNFocalModulation2D(KANFocalModulationND):
    def __init__(
        self,
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        focal_norm_layer = {'layer': nn.BatchNorm2d, 'params': kwargs}

        super(BottleNeckKAGNFocalModulation2D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv2DLayer,
            focal_norm_layer=focal_norm_layer,
            num_channels=num_channels,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class BottleNeckKAGNFocalModulation1D(KANFocalModulationND):
    def __init__(
        self,
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        focal_norm_layer = {'layer': nn.BatchNorm1d, 'params': kwargs}

        super(BottleNeckKAGNFocalModulation1D, self).__init__(
            conv_kan_layer=BottleNeckKAGNConv1DLayer,
            focal_norm_layer=focal_norm_layer,
            num_channels=num_channels,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class KAGNFocalModulation3D(KANFocalModulationND):
    def __init__(
        self,
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        focal_norm_layer = {'layer': nn.BatchNorm3d, 'params': kwargs}

        super(KAGNFocalModulation3D, self).__init__(
            conv_kan_layer=KAGNConv3DLayer,
            focal_norm_layer=focal_norm_layer,
            num_channels=num_channels,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class KAGNFocalModulation2D(KANFocalModulationND):
    def __init__(
        self,
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        focal_norm_layer = {'layer': nn.BatchNorm2d, 'params': kwargs}

        super(KAGNFocalModulation2D, self).__init__(
            conv_kan_layer=KAGNConv2DLayer,
            focal_norm_layer=focal_norm_layer,
            num_channels=num_channels,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )


class KAGNFocalModulation1D(KANFocalModulationND):
    def __init__(
        self,
        num_channels: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
        spline_order: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        focal_norm_layer = {'layer': nn.BatchNorm1d, 'params': kwargs}

        super(KAGNFocalModulation1D, self).__init__(
            conv_kan_layer=KAGNConv1DLayer,
            focal_norm_layer=focal_norm_layer,
            num_channels=num_channels,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            spline_order=spline_order,
            dropout=dropout,
            **kwargs,
        )
