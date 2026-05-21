# The code is based on the Yeonwoo Sung's implementation:
# https://github.com/YeonwooSung/Pytorch_mixture-of-experts/blob/main/moe.py

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from .fastkan_conv import FastKANConv1DLayer, FastKANConv2DLayer, FastKANConv3DLayer
from .kacn_conv import KACNConv1DLayer, KACNConv2DLayer, KACNConv3DLayer
from .kagn_bottleneck_conv import (
    BottleNeckKAGNConv1DLayer,
    BottleNeckKAGNConv2DLayer,
    BottleNeckKAGNConv3DLayer,
)
from .kagn_conv import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from .kaln_conv import KALNConv1DLayer, KALNConv2DLayer, KALNConv3DLayer
from .kan_conv import KANConv1DLayer, KANConv2DLayer, KANConv3DLayer
from .moe_utils import SparseDispatcher
from .wavkan_conv import WavKANConv1DLayer, WavKANConv2DLayer, WavKANConv3DLayer

from torchkan.utils.typing import (
    Padding1D,
    Padding2D,
    Padding3D,
    PaddingND,
    Size2D,
    Size3D,
    SizeND,
)


class MoEKANConvBase(nn.Module):
    def __init__(
        self,
        conv_class: type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: SizeND,
        stride: SizeND,
        padding: PaddingND,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        experts = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        self.experts = nn.ModuleList([experts] * num_experts)

        self.w_gate = nn.Parameter(
            torch.zeros(in_channels, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(in_channels, num_experts), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))

        name = conv_class.__name__
        if name.endswith('1DLayer'):
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.conv_dims = 1
        elif name.endswith('2DLayer'):
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.conv_dims = 2
        elif name.endswith('3DLayer'):
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.conv_dims = 3
        else:
            raise ValueError('Unsupported dimention of conv_class.')

        for i in range(1, num_experts):
            self.experts[i].load_state_dict(self.experts[0].state_dict())

        assert self.k <= self.num_experts

    def cv_squared(self, x: Tensor) -> Tensor:
        eps = 1e-10
        # if only num_experts = 1
        x = x.float()
        if x.shape[0] == 1:
            return torch.tensor(0, device=x.device, dtype=x.dtype)
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
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = noisy_values > threshold_if_in
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)  # type: ignore
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x: Tensor, train: bool, noise_epsilon: float = 1e-2):
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
            top_k_logits.sum(1, keepdim=True) + 1e-6
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

    def forward(
        self, x: Tensor, train: bool, loss_coef: float = 1e-2
    ) -> tuple[Tensor, Tensor]:
        gate_x = torch.flatten(self.avgpool(x), 1)
        gates, load = self.noisy_top_k_gating(gate_x, train)
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


class MoEKALNConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KALNConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKALNConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KALNConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKALNConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KALNConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKANConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KANConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKANConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KANConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKANConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KANConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKAGNConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KAGNConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKAGNConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KAGNConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKAGNConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KAGNConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEFastKANConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=FastKANConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEFastKANConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=FastKANConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEFastKANConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=FastKANConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKACNConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KACNConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKACNConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KACNConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEKACNConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=KACNConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEWavKANConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=WavKANConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEWavKANConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=WavKANConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEWavKANConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=WavKANConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEFullBottleneckKAGNConv3DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D = 3,
        stride: Size3D = 1,
        padding: Padding3D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=BottleNeckKAGNConv3DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEFullBottleneckKAGNConv2DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D = 3,
        stride: Size2D = 1,
        padding: Padding2D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=BottleNeckKAGNConv2DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )


class MoEFullBottleneckKAGNConv1DLayer(MoEKANConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Padding1D = 1,
        num_experts: int = 16,
        noisy_gating: bool = True,
        k: int = 4,
        **kwargs,
    ):
        super().__init__(
            conv_class=BottleNeckKAGNConv1DLayer,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            **kwargs,
        )
