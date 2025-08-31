from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from torchkan.conv import (
    BottleNeckKAGNConv2DLayer,
    FastKANConv2DLayer,
    KACNConv2DLayer,
    KAGNConv2DLayer,
    KALNConv2DLayer,
    KANConv2DLayer,
)
from torchkan.linear import KACN, KAGN, KALN, KAN, BottleNeckKAGN, FastKAN

from .utils.conv_utils import (
    bottleneck_kagn_conv1x1,
    bottleneck_kagn_conv3x3,
    fast_kan_conv1x1,
    fast_kan_conv3x3,
    kacn_conv1x1,
    kacn_conv3x3,
    kagn_conv1x1,
    kagn_conv3x3,
    kaln_conv1x1,
    kaln_conv3x3,
    kan_conv1x1,
    kan_conv3x3,
    moe_bottleneck_kagn_conv3x3,
)

from torchkan.utils.typing import ConvFunc


class DenseLayer(nn.Module):
    def __init__(
        self,
        conv1x1_fun: ConvFunc,
        conv3x3_fun: ConvFunc,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        dropout: float,
        memory_efficient: bool = False,
        is_moe: bool = False,
    ):
        super().__init__()
        self.conv1 = conv1x1_fun(
            num_input_features,
            bn_size * growth_rate,
            stride=1,
        )
        self.conv2 = conv3x3_fun(
            bn_size * growth_rate,
            growth_rate,
            stride=1,
            dropout=dropout,
        )
        self.memory_efficient = memory_efficient
        self.is_moe = is_moe

    def any_requires_grad(self, tensors: list[Tensor]) -> bool:
        if any(tensor.requires_grad for tensor in tensors):
            return True
        return False

    def bn_function(self, tensors: list[Tensor]) -> Tensor:
        conconcated_features = torch.concat(tensors, dim=1)
        bottleneck_output = self.conv1(conconcated_features)
        return bottleneck_output

    def call_checkpoint_bottleneck(self, tensors: list[Tensor]) -> Any:
        def closure(*tensors):
            return self.bn_function(*tensors)

        return cp.checkpoint(closure, *tensors, use_reentrant=False)

    def forward(self, x: Tensor, **kwargs) -> Tensor | tuple[Tensor, float]:
        prev_features = [x] if isinstance(x, Tensor) else x

        if self.memory_efficient and self.any_requires_grad(prev_features):
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        if self.is_moe:
            return self.conv2(bottleneck_output, **kwargs)

        return self.conv2(bottleneck_output)


class DenseBlock(nn.ModuleDict):
    version = 2

    def __init__(
        self,
        conv1x1x1_fun: ConvFunc,
        conv3x3x3_fun: ConvFunc,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        dropout: float,
        memory_efficient: bool = False,
        is_moe: bool = False,
    ):
        super().__init__()
        self.is_moe = is_moe
        for i in range(num_layers):
            layer = DenseLayer(
                conv1x1x1_fun,
                conv3x3x3_fun,
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout,
                memory_efficient=memory_efficient,
                is_moe=is_moe,
            )
            self.add_module(f'DenseLayer{i + 1}', layer)

    def forward(self, x: Tensor, **kwargs) -> Tensor | tuple[Tensor, float]:
        features = [x]
        moe_loss = 0.0
        for _, layer in self.items():
            new_features = layer(features, **kwargs)
            if self.is_moe:
                new_features, loss = new_features
                moe_loss += loss
            features.append(new_features)
        if self.is_moe:
            return torch.concat(features, dim=1), moe_loss
        return torch.concat(features, dim=1)


class KANDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        spline_order: int = 3,
        groups: int = 1,
        grid_size: int = 5,
        base_activation: ConvFunc = nn.GELU(),
        grid_range: tuple[float, float] = (-1.0, 1.0),
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kan_conv1x1,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kan_conv3x3,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            l1_decay=l1_decay,
            groups=groups,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
        )


class KALNDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        spline_order: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kaln_conv1x1,
            spline_order=spline_order,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kaln_conv3x3,
            spline_order=spline_order,
            l1_decay=l1_decay,
            groups=groups,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
        )


class KAGNDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        spline_order: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kagn_conv1x1,
            spline_order=spline_order,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kagn_conv3x3,
            spline_order=spline_order,
            l1_decay=l1_decay,
            groups=groups,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
        )


class KACNDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        spline_order: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        l1_decay: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            kacn_conv1x1,
            spline_order=spline_order,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            kacn_conv3x3,
            spline_order=spline_order,
            l1_decay=l1_decay,
            groups=groups,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
        )


class FastKANDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        groups: int = 1,
        grid_size: int = 8,
        base_activation: nn.Module = nn.SiLU(),
        grid_range: tuple[float, float] = (-2.0, 2.0),
        l1_decay: float = 0.0,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            fast_kan_conv1x1,
            grid_range=grid_range,
            grid_size=grid_size,
            l1_decay=l1_decay,
            base_activation=base_activation,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            fast_kan_conv3x3,
            grid_range=grid_range,
            grid_size=grid_size,
            l1_decay=l1_decay,
            groups=groups,
            base_activation=base_activation,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
        )


class BottleNeckKAGNDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        spline_order: int = 3,
        groups: int = 1,
        l1_decay: float = 0.0,
        dropout: float = 0.0,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            bottleneck_kagn_conv1x1,
            spline_order=spline_order,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            bottleneck_kagn_conv3x3,
            spline_order=spline_order,
            l1_decay=l1_decay,
            groups=groups,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
        )


class MoEBottleNeckKAGNDenseBlock(DenseBlock):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        spline_order: int = 3,
        groups: int = 1,
        l1_decay: float = 0.0,
        dropout: float = 0.0,
        num_experts: int = 8,
        noisy_gating: bool = True,
        k: int = 2,
        **norm_kwargs,
    ):
        conv1x1x1_fun = partial(
            bottleneck_kagn_conv1x1,
            spline_order=spline_order,
            l1_decay=l1_decay,
            **norm_kwargs,
        )
        conv3x3x3_fun = partial(
            moe_bottleneck_kagn_conv3x3,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            spline_order=spline_order,
            l1_decay=l1_decay,
            groups=groups,
            **norm_kwargs,
        )
        super().__init__(
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers=num_layers,
            num_input_features=num_input_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            dropout=dropout,
            memory_efficient=memory_efficient,
            is_moe=True,
        )


class Transition(nn.Module):
    # switch to KAN Convs?
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.SELU()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseKANet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multipliconcative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classificoncation classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        block_class: type[nn.Module],
        input_channels: int = 3,
        num_init_features: int = 64,
        use_first_maxpool: bool = True,
        mp_kernel_size: int = 3,
        mp_stride: int = 2,
        mp_padding: int = 1,
        fcnv_kernel_size: int = 7,
        fcnv_stride: int = 2,
        fcnv_padding: int = 3,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 24, 16),
        bn_size: int = 4,
        dropout: float = 0,
        dropout_linear: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
        **kan_kwargs,
    ):
        super().__init__()

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('dropout', None)
        kan_kwargs_clean.pop('groups', None)
        kan_kwargs_clean.pop('num_experts', None)
        kan_kwargs_clean.pop('k', None)
        kan_kwargs_clean.pop('noisy_gating', None)

        self.is_moe = False
        if block_class == MoEBottleNeckKAGNDenseBlock:
            self.is_moe = True

        match block_class.__name__:
            case 'KANDenseBlock':
                conv1 = KANConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'FastKANDenseBlock':
                conv1 = FastKANConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KALNDenseBlock':
                conv1 = KALNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KAGNDenseBlock':
                conv1 = KAGNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'BottleNeckKAGNDenseBlock' | 'MoEBottleNeckKAGNDenseBlock':
                conv1 = BottleNeckKAGNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KACNDenseBlock':
                conv1 = KACNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case _:
                raise ValueError(f'Block {block_class.__name__} is not supported.')

        # First convolution
        self.layers_order = ['conv0']
        self.features = nn.ModuleDict({})
        self.features.add_module('conv0', conv1)
        if use_first_maxpool:
            self.features.add_module(
                name='pool0',
                module=nn.MaxPool2d(
                    kernel_size=mp_kernel_size,
                    stride=mp_stride,
                    padding=mp_padding,
                ),
            )
            self.layers_order.append('pool0')

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = block_class(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout=dropout,
                memory_efficient=memory_efficient,
                **kan_kwargs,
            )
            self.features.add_module(f'DenseBlock{i + 1}', block)
            self.layers_order.append(f'DenseBlock{i + 1}')

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'Transition{i + 1}', trans)
                self.layers_order.append(f'Transition{i + 1}')
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.dropout_lin = None
        if dropout_linear > 0:
            self.dropout_lin = nn.Dropout(p=dropout_linear)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor, **kwargs) -> Tensor | tuple[Tensor, float]:
        moe_loss = 0.0
        for layer_name in self.layers_order:
            if self.is_moe and 'DenseBlock' in layer_name:
                x, loss = self.features[layer_name](x, **kwargs)
                moe_loss += loss
            else:
                x = self.features[layer_name](x)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, start_dim=1)
        if self.dropout_lin is not None:
            x = self.dropout_lin(x)
        x = self.classifier(x)
        if self.is_moe:
            return x, moe_loss
        return x


class TinyDenseKANet(nn.Module):
    r"""Densenet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>` and https://arxiv.org/pdf/1904.10429v2.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multipliconcative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classificoncation classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        block_class: type[nn.Module],
        input_channels: int = 3,
        num_init_features: int = 64,
        fcnv_kernel_size: int = 5,
        fcnv_stride: int = 2,
        fcnv_padding: int = 2,
        growth_rate: int = 32,
        block_config: tuple[int, int, int] = (5, 5, 5),
        bn_size: int = 4,
        dropout: float = 0,
        dropout_linear: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
        **kan_kwargs,
    ) -> None:
        super().__init__()

        self.is_moe = False
        if MoEBottleNeckKAGNDenseBlock == block_class:
            self.is_moe = True

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)
        kan_kwargs_clean.pop('num_experts', None)
        kan_kwargs_clean.pop('k', None)
        kan_kwargs_clean.pop('noisy_gating', None)
        kan_kwargs_clean.pop('dropout', None)

        match block_class.__name__:
            case 'KANDenseBlock':
                conv1 = KANConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'FastKANDenseBlock':
                conv1 = FastKANConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KALNDenseBlock':
                conv1 = KALNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KAGNDenseBlock':
                conv1 = KAGNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'BottleNeckKAGNDenseBlock' | 'MoEBottleNeckKAGNDenseBlock':
                conv1 = BottleNeckKAGNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case 'KACNDenseBlock':
                conv1 = KACNConv2DLayer(
                    input_channels,
                    num_init_features,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **kan_kwargs_clean,
                )
            case _:
                raise ValueError(f'Block {block_class.__name__} is not supported.')

        self.layers_order = ['conv0']
        self.features = nn.ModuleDict({})
        self.features.add_module('conv0', conv1)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = block_class(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout=dropout,
                memory_efficient=memory_efficient,
                **kan_kwargs,
            )
            self.features.add_module(f'DenseBlock{i + 1}', block)
            self.layers_order.append(f'DenseBlock{i + 1}')

            num_features = num_features + num_layers * growth_rate
            trans = Transition(num_features, num_features // 2)
            self.features.add_module(f'Transition{i + 1}', trans)
            self.layers_order.append(f'Transition{i + 1}')
            num_features = num_features // 2

        self.dropout_lin = None
        if dropout_linear > 0:
            self.dropout_lin = nn.Dropout(p=dropout_linear)

        # Linear layer
        hidden_layers = (num_features, num_classes)
        match block_class.__name__:
            case 'KANDenseBlock':
                self.classifier = KAN(hidden_layers, **kan_kwargs_clean)
            case 'FastKANDenseBlock':
                self.classifier = FastKAN(hidden_layers, **kan_kwargs_clean)
            case 'KALNDenseBlock':
                self.classifier = KALN(hidden_layers, **kan_kwargs_clean)
            case 'KAGNDenseBlock':
                self.classifier = KAGN(hidden_layers, **kan_kwargs_clean)
            case 'BottleNeckKAGNDenseBlock' | 'MoEBottleNeckKAGNDenseBlock':
                self.classifier = BottleNeckKAGN(hidden_layers, **kan_kwargs_clean)
            case 'KACNDenseBlock':
                self.classifier = KACN(hidden_layers, **kan_kwargs_clean)
            case _:
                raise ValueError(f'Block {block_class.__name__} is not supported.')

    def forward(self, x: Tensor, **kwargs) -> Tensor | tuple[Tensor, float]:
        moe_loss = 0.0
        for layer_name in self.layers_order:
            if self.is_moe and 'DenseBlock' in layer_name:
                x, loss = self.features[layer_name](x, **kwargs)
                moe_loss += loss
            else:
                x = self.features[layer_name](x)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, start_dim=1)
        if self.dropout_lin is not None:
            x = self.dropout_lin(x)
        x = self.classifier(x)
        if self.is_moe:
            return x, moe_loss
        return x
