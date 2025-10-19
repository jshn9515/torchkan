# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py

from functools import lru_cache
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from torch.nn.functional import conv1d, conv2d, conv3d

from ..linear import GRAMLayer
from ..utils import NoiseInjection
from .moe_utils import SparseDispatcher

from torchkan.utils.typing import (
    Activation,
    Padding1D,
    Padding2D,
    Padding3D,
    PaddingND,
    Size2D,
    Size3D,
    SizeND,
)


class BottleNeckKAGNConvNDLayer(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        conv_w_fun: Callable[..., Tensor],
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        base_activation: Activation = nn.SiLU(),
        dropout: float = 0.0,
        dim_reduction: float = 4.0,
        min_internal: int = 16,
        **norm_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = base_activation
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs

        inner_channels = int(
            max(
                (in_channels // groups) / dim_reduction,
                (out_channels // groups) / dim_reduction,
            )
        )
        if inner_channels < min_internal:
            self.inner_channels = min(
                min_internal, in_channels // groups, out_channels // groups
            )
        else:
            self.inner_channels = inner_channels

        if dropout > 0:
            self.dropout = NoiseInjection(p=dropout, alpha=0.05)

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

        inner_proj = conv_class(
            in_channels=in_channels // groups,
            out_channels=self.inner_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.inner_proj = nn.ModuleList([inner_proj] * groups)

        outer_proj = conv_class(
            in_channels=self.inner_channels,
            out_channels=out_channels // groups,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.outer_proj = nn.ModuleList([outer_proj] * groups)

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim

        poly_shape = (
            groups,
            self.inner_channels,
            self.inner_channels * (spline_order + 1),
            *kernel_size,
        )

        self.poly_weights = nn.Parameter(torch.randn(poly_shape))
        self.beta_weights = nn.Parameter(
            torch.zeros(spline_order + 1, dtype=torch.float32)
        )

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )
        for conv_layer in self.inner_proj:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )
        for conv_layer in self.outer_proj:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0
            / (
                (sum(kernel_size) / len(kernel_size) ** ndim)
                * in_channels
                * (spline_order + 1.0)
            ),
        )

    def beta(self, n: int, m: int) -> Tensor:
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Gram polynomials
    def gram_poly(self, x: Tensor) -> Tensor:
        P0 = x.new_ones(x.size())

        if self.spline_order == 0:
            return torch.unsqueeze(P0, dim=-1)

        P1 = x
        grams_basis = [P0, P1]

        for i in range(2, self.spline_order + 1):
            p2 = x * P1 - self.beta(i - 1, i) * P0
            grams_basis.append(p2)
            P0, P1 = P1, p2

        return torch.concat(grams_basis, dim=1)

    def forward_kag(self, x: Tensor, group_index: int) -> Tensor:
        if self.dropout:
            x = self.dropout(x)

        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = self.inner_proj[group_index](x)
        x = torch.tanh(x).contiguous()

        grams_basis = self.base_activation(self.gram_poly(x))

        y = self.conv_w_fun(
            grams_basis,
            self.poly_weights[group_index],
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=1,
        )
        y = self.outer_proj[group_index](y)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_kag(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class BottleNeckKAGNConv3DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Padding3D = 0,
        dilation: Size3D = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        dim_reduction: float = 4.0,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv3d,
            norm_class=nn.InstanceNorm3d,
            conv_w_fun=conv3d,
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            dim_reduction=dim_reduction,
            **norm_kwargs,
        )


class BottleNeckKAGNConv2DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D,
        stride: Size2D = 1,
        padding: Padding2D = 0,
        dilation: Size2D = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        dim_reduction: float = 4.0,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv2d,
            norm_class=nn.InstanceNorm2d,
            conv_w_fun=conv2d,
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            dim_reduction=dim_reduction,
            **norm_kwargs,
        )


class BottleNeckKAGNConv1DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Padding1D = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        dim_reduction: float = 4.0,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv1d,
            norm_class=nn.InstanceNorm1d,
            conv_w_fun=conv1d,
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            dim_reduction=dim_reduction,
            **norm_kwargs,
        )


class KAGNExpert(nn.Module):
    def __init__(
        self,
        conv_w_fun: Callable[..., Tensor],
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        base_activation: Activation = nn.SiLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = base_activation
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None

        if dropout > 0:
            self.dropout = NoiseInjection(p=dropout, alpha=0.05)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim

        poly_shape = (
            groups,
            out_channels // groups,
            (in_channels // groups) * (spline_order + 1),
            *kernel_size,
        )

        self.poly_weights = nn.Parameter(torch.randn(poly_shape))
        self.beta_weights = nn.Parameter(
            torch.zeros(spline_order + 1, dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0
            / (
                (sum(kernel_size) / len(kernel_size) ** ndim)
                * in_channels
                * (spline_order + 1.0)
            ),
        )

    def beta(self, n: int, m: int) -> Tensor:
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Gram polynomials
    def gram_poly(self, x: Tensor) -> Tensor:
        P0 = x.new_ones(x.shape)

        if self.spline_order == 0:
            return torch.unsqueeze(P0, dim=-1)

        P1 = x
        grams_basis = [P0, P1]

        for i in range(2, self.spline_order + 1):
            P2 = x * P1 - self.beta(i - 1, i) * P0
            grams_basis.append(P2)
            P0, P1 = P1, P2

        return torch.concat(grams_basis, dim=1)

    def forward_kag(self, x: Tensor, group_index: int) -> Tensor:
        x = torch.tanh(x).contiguous()

        grams_basis = self.base_activation(self.gram_poly(x))

        if self.dropout:
            grams_basis = self.dropout(grams_basis)

        y = self.conv_w_fun(
            grams_basis,
            self.poly_weights[group_index],
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=1,
        )

        return y

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_index, x in enumerate(split_x):
            y = self.forward_kag(x, group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y


class KAGNMoE(nn.Module):
    def __init__(
        self,
        num_experts,
        conv_w_fun: Callable,
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        dropout: float = 0.0,
        k: int = 4,
        noisy_gating: bool = True,
        pregate: bool = False,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        # instantiate experts
        experts = KAGNExpert(
            conv_w_fun=conv_w_fun,
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
        )
        self.experts = nn.ModuleList([experts] * num_experts)

        self.w_gate = nn.Parameter(
            torch.zeros(in_channels, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(in_channels, num_experts), requires_grad=True
        )

        self.pre_gate = None
        if pregate:
            self.pre_gate = GRAMLayer(
                in_channels, out_channels, spline_order=spline_order
            )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))

        if ndim == 1:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.conv_dims = 1
        elif ndim == 2:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.conv_dims = 2
        elif ndim == 3:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.conv_dims = 3
        else:
            raise ValueError(f'ndim must be 1, 2, or 3, but got {ndim}.')

        assert self.k <= self.num_experts

    def cv_squared(self, x: Tensor) -> Tensor:
        eps = 1e-10
        x = x.float()
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.var() / (x.mean() ** 2 + eps)

    def _gates_to_load(self, gates: Tensor) -> Tensor:
        return torch.count_nonzero(gates > 0, dim=0)

    def _prob_in_top_k(
        self,
        clean_values: Tensor,
        noisy_values: Tensor,
        noise_stddev: Tensor,
        noisy_top_values: Tensor,
    ) -> Tensor:
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), dim=1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), dim=1
        )
        # is each value currently in the top k.
        normal = Normal(loc=self.mean, scale=self.std)  # type: ignore
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(
        self, x: Tensor, train: bool, noise_epsilon: float = 1e-2
    ) -> tuple[Tensor, Tensor]:
        clean_logits = x @ self.w_gate
        noisy_logits = torch.tensor(0.0)
        noise_stddev = torch.tensor(0.0)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(dim=1, keepdim=True) + 1e-6
        )  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(dim=0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: Tensor, loss_coef: float = 1e-2) -> tuple[Tensor, Tensor]:
        gate_x = torch.flatten(self.avgpool(x), start_dim=1)
        if self.pre_gate:
            gate_x = self.pre_gate(gate_x)
        gates, load = self.noisy_top_k_gating(gate_x, self.training)
        # calculate importance loss
        importance = gates.sum(dim=0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs, self.conv_dims)
        return y, loss


class MoEBottleNeckKAGNConvND(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        norm_class: type[nn.Module],
        conv_w_fun: Callable[..., Tensor],
        ndim: int,
        in_channels: int,
        out_channels: int,
        spline_order: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        dilation: SizeND,
        groups: int = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        dim_reduction: float = 4,
        min_internal: int = 16,
        pregate: bool = False,
        base_activation: Activation = nn.SiLU(),
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.groups = groups
        self.base_activation = base_activation
        # instantiate experts
        inner_channels = int(
            max(
                (in_channels // groups) / dim_reduction,
                (out_channels // groups) / dim_reduction,
            )
        )
        if inner_channels < min_internal:
            self.inner_channels = min(
                min_internal, in_channels // groups, out_channels // groups
            )
        else:
            self.inner_channels = inner_channels

        self.experts = KAGNMoE(
            num_experts=num_experts,
            conv_w_fun=conv_w_fun,
            ndim=ndim,
            in_channels=inner_channels * groups,
            out_channels=inner_channels * groups,
            spline_order=spline_order,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            k=k,
            noisy_gating=noisy_gating,
            pregate=pregate,
        )

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

        inner_proj = conv_class(
            in_channels=in_channels // groups,
            out_channels=self.inner_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.inner_proj = nn.ModuleList([inner_proj] * groups)

        outer_proj = conv_class(
            in_channels=self.inner_channels,
            out_channels=out_channels // groups,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.outer_proj = nn.ModuleList([outer_proj] * groups)

        layer_norm = norm_class(out_channels // groups, **norm_kwargs)
        self.layer_norm = nn.ModuleList([layer_norm] * groups)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )
        for conv_layer in self.inner_proj:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )
        for conv_layer in self.outer_proj:
            nn.init.kaiming_uniform_(
                torch.as_tensor(conv_layer.weight), nonlinearity='linear'
            )

        self.w_gate = nn.Parameter(
            torch.zeros(self.inner_channels * groups, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(self.inner_channels * groups, num_experts), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        if ndim == 1:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.conv_dims = 1
        elif ndim == 2:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.conv_dims = 2
        elif ndim == 3:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.conv_dims = 3

        assert self.k <= self.num_experts

    def forward_moe_base(self, x: Tensor, group_index: int) -> Tensor:
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))
        return basis

    def forward_moe_inner(self, x: Tensor, group_index: int) -> Tensor:
        y = self.inner_proj[group_index](x)
        return y

    def forward_moe_outer(self, x: Tensor, basis: Tensor, group_index: int) -> Tensor:
        y = self.outer_proj[group_index](x)
        y = self.base_activation(self.layer_norm[group_index](y + basis))
        return y

    def forward(self, x: Tensor, loss_coef: float = 1e-2) -> tuple[Tensor, Tensor]:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        bases, output = [], []
        for group_index, x in enumerate(split_x):
            base = self.forward_moe_base(x, group_index)
            bases.append(base)
            y = self.forward_moe_inner(x, group_index)
            output.append(y)

        y, loss = self.experts.forward(torch.concat(output, dim=1), loss_coef=loss_coef)
        output = []
        for group_index, (xb, xe) in enumerate(
            zip(
                bases,
                torch.split(y, self.inner_channels, dim=1),
                strict=True,
            )
        ):
            y = self.forward_moe_outer(xe, xb, group_index=group_index)
            output.append(y)
        y = torch.concat(output, dim=1)
        return y, loss


class MoEBottleNeckKAGNConv3DLayer(MoEBottleNeckKAGNConvND):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Padding3D = 0,
        dilation: Size3D = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        dim_reduction: float = 4,
        num_experts: int = 16,
        k: int = 4,
        noisy_gating: bool = True,
        pregate: bool = False,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv3d,
            norm_class=nn.InstanceNorm3d,
            conv_w_fun=conv3d,
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            dim_reduction=dim_reduction,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            pregate=pregate,
            **norm_kwargs,
        )


class MoEBottleNeckKAGNConv2DLayer(MoEBottleNeckKAGNConvND):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D,
        stride: Size2D = 1,
        padding: Padding2D = 0,
        dilation: Size2D = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        dim_reduction: float = 4,
        num_experts: int = 16,
        k: int = 4,
        noisy_gating: bool = True,
        pregate: bool = False,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv2d,
            norm_class=nn.InstanceNorm2d,
            conv_w_fun=conv2d,
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            dim_reduction=dim_reduction,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            pregate=pregate,
            **norm_kwargs,
        )


class MoEBottleNeckKAGNConv1DLayer(MoEBottleNeckKAGNConvND):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Padding1D = 0,
        dilation: int = 1,
        groups: int = 1,
        spline_order: int = 3,
        dropout: float = 0.0,
        dim_reduction: float = 4,
        num_experts: int = 16,
        k: int = 4,
        noisy_gating: bool = True,
        pregate: bool = False,
        **norm_kwargs,
    ):
        super().__init__(
            conv_class=nn.Conv1d,
            norm_class=nn.InstanceNorm1d,
            conv_w_fun=conv1d,
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
            dim_reduction=dim_reduction,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            pregate=pregate,
            **norm_kwargs,
        )
