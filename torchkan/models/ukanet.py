import copy
from functools import partial
from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from torchkan.conv import (
    BottleNeckKAGNConv2DLayer,
    BottleNeckKAGNFocalModulation2D,
    BottleNeckSelfKAGNtention2D,
    FastKANConv2DLayer,
    KACNConv2DLayer,
    KAGNConv2DLayer,
    KAGNFocalModulation2D,
    KALNConv2DLayer,
    KANConv2DLayer,
    SelfKAGNtention2D,
)

from .reskanet import BlockProtocol
from .utils.conv_utils import (
    bottleneck_kagn_conv1x1,
    fast_kan_conv1x1,
    kacn_conv1x1,
    kagn_conv1x1,
    kaln_conv1x1,
    kan_conv1x1,
)


class UKANet(nn.Module):
    def __init__(
        self,
        block: type[BlockProtocol],
        layers: tuple[int, ...],
        in_channels: int = 3,
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        fcnv_kernel_size: int = 7,
        fcnv_stride: int = 1,
        fcnv_padding: int = 3,
        replace_stride_with_dilation: list[bool] | None = None,
        width_scale: int = 1,
        **kan_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inplanes = 8 * width_scale
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f'replace_stride_with_dilation should be None or a 3-element tuple, '
                f'got {replace_stride_with_dilation}.'
            )
        self.groups = groups
        self.base_width = width_per_group

        clean_params = copy.deepcopy(kan_kwargs)
        del clean_params['l1_decay']

        match block.__name__:
            case 'KANBasicBlock' | 'KANBottleneck':
                self.conv1 = KANConv2DLayer(
                    in_channels=in_channels,
                    out_channels=self.inplanes,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **clean_params,
                )
                self.merge1 = KANConv2DLayer(
                    in_channels=(8 + 16) * width_scale * block.expansion,
                    out_channels=8 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge2 = KANConv2DLayer(
                    in_channels=(32 + 16) * width_scale * block.expansion,
                    out_channels=16 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge3 = KANConv2DLayer(
                    in_channels=(32 + 64) * width_scale * block.expansion,
                    out_channels=32 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )

            case 'FastKANBasicBlock' | 'FastKANBottleneck':
                self.conv1 = FastKANConv2DLayer(
                    in_channels=in_channels,
                    out_channels=self.inplanes,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **clean_params,
                )
                self.merge1 = FastKANConv2DLayer(
                    in_channels=(8 + 16) * width_scale * block.expansion,
                    out_channels=8 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge2 = FastKANConv2DLayer(
                    in_channels=(32 + 16) * width_scale * block.expansion,
                    out_channels=16 * width_scale * block.expansion,
                    kernel_size=3,
                    groups=groups,
                    stride=1,
                    padding=1,
                    **clean_params,
                )
                self.merge3 = FastKANConv2DLayer(
                    in_channels=(32 + 64) * width_scale * block.expansion,
                    out_channels=32 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )

            case 'KALNBasicBlock' | 'KALNBottleneck':
                self.conv1 = KALNConv2DLayer(
                    in_channels=in_channels,
                    out_channels=self.inplanes,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **clean_params,
                )
                self.merge1 = KALNConv2DLayer(
                    in_channels=(8 + 16) * width_scale * block.expansion,
                    out_channels=8 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge2 = KALNConv2DLayer(
                    in_channels=(32 + 16) * width_scale * block.expansion,
                    out_channels=16 * width_scale * block.expansion,
                    kernel_size=3,
                    groups=groups,
                    stride=1,
                    padding=1,
                    **clean_params,
                )
                self.merge3 = KALNConv2DLayer(
                    in_channels=(32 + 64) * width_scale * block.expansion,
                    out_channels=32 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )

            case 'KAGNBasicBlock' | 'KAGNBottleneck':
                self.conv1 = KAGNConv2DLayer(
                    in_channels=in_channels,
                    out_channels=self.inplanes,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **clean_params,
                )
                self.merge1 = KAGNConv2DLayer(
                    in_channels=(8 + 16) * width_scale * block.expansion,
                    out_channels=8 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge2 = KAGNConv2DLayer(
                    in_channels=(32 + 16) * width_scale * block.expansion,
                    out_channels=16 * width_scale * block.expansion,
                    kernel_size=3,
                    groups=groups,
                    stride=1,
                    padding=1,
                    **clean_params,
                )
                self.merge3 = KAGNConv2DLayer(
                    in_channels=(32 + 64) * width_scale * block.expansion,
                    out_channels=32 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )

            case 'BottleneckKAGNBasicBlock':
                self.conv1 = BottleNeckKAGNConv2DLayer(
                    in_channels=in_channels,
                    out_channels=self.inplanes,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **clean_params,
                )
                self.merge1 = BottleNeckKAGNConv2DLayer(
                    in_channels=(8 + 16) * width_scale * block.expansion,
                    out_channels=8 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge2 = BottleNeckKAGNConv2DLayer(
                    in_channels=(32 + 16) * width_scale * block.expansion,
                    out_channels=16 * width_scale * block.expansion,
                    kernel_size=3,
                    groups=groups,
                    stride=1,
                    padding=1,
                    **clean_params,
                )
                self.merge3 = BottleNeckKAGNConv2DLayer(
                    in_channels=(32 + 64) * width_scale * block.expansion,
                    out_channels=32 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )

            case 'KACNBasicBlock' | 'KACNBottleneck':
                self.conv1 = KACNConv2DLayer(
                    in_channels=in_channels,
                    out_channels=self.inplanes,
                    kernel_size=fcnv_kernel_size,
                    stride=fcnv_stride,
                    padding=fcnv_padding,
                    **clean_params,
                )
                self.merge1 = KACNConv2DLayer(
                    in_channels=(8 + 16) * width_scale * block.expansion,
                    out_channels=8 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )
                self.merge2 = KACNConv2DLayer(
                    in_channels=(32 + 16) * width_scale * block.expansion,
                    out_channels=16 * width_scale * block.expansion,
                    kernel_size=3,
                    groups=groups,
                    stride=1,
                    padding=1,
                    **clean_params,
                )
                self.merge3 = KACNConv2DLayer(
                    in_channels=(32 + 64) * width_scale * block.expansion,
                    out_channels=32 * width_scale * block.expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    **clean_params,
                )

            case _:
                raise TypeError(f'Block {block.__name__} is not supported.')

        self.layer1e = self._make_layer(
            block,
            8 * block.expansion * width_scale,
            layers[0],
            **kan_kwargs,
        )
        l1e_inplanes = self.inplanes
        self.layer2e = self._make_layer(
            block,
            16 * block.expansion * width_scale,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            **kan_kwargs,
        )
        l2e_inplanes = self.inplanes
        self.layer3e = self._make_layer(
            block,
            32 * block.expansion * width_scale,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            **kan_kwargs,
        )
        l3e_inplanes = self.inplanes
        self.layer4e = self._make_layer(
            block,
            64 * block.expansion * width_scale,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            **kan_kwargs,
        )

        self.layer4d = self._make_layer(
            block,
            64 * block.expansion * width_scale,
            layers[3],
            **kan_kwargs,
        )
        self.inplanes = l1e_inplanes
        self.layer1d = self._make_layer(
            block,
            8 * block.expansion * width_scale,
            layers[0],
            **kan_kwargs,
        )
        self.inplanes = l2e_inplanes
        self.layer2d = self._make_layer(
            block,
            16 * block.expansion * width_scale,
            layers[1],
            **kan_kwargs,
        )
        self.inplanes = l3e_inplanes
        self.layer3d = self._make_layer(
            block,
            32 * block.expansion * width_scale,
            layers[2],
            **kan_kwargs,
        )

        self.output = nn.Conv2d(
            in_channels=8 * block.expansion * width_scale,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def _make_layer(
        self,
        block: type[BlockProtocol],
        planes: int,
        num_block: int,
        stride: int = 1,
        dilate: bool = False,
        **kan_kwargs,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            match block.__name__:
                case 'KANBasicBlock' | 'KANBottleneck':
                    conv1x1 = partial(kan_conv1x1, **kan_kwargs)
                case 'FastKANBasicBlock' | 'FastKANBottleneck':
                    conv1x1 = partial(fast_kan_conv1x1, **kan_kwargs)
                case 'KALNBasicBlock' | 'KALNBottleneck':
                    conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
                case 'KAGNBasicBlock' | 'KAGNBottleneck':
                    conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
                case 'BottleneckKAGNBasicBlock':
                    conv1x1 = partial(bottleneck_kagn_conv1x1, **kan_kwargs)
                case 'KACNBasicBlock' | 'KACNBottleneck':
                    conv1x1 = partial(kacn_conv1x1, **kan_kwargs)
                case _:
                    raise TypeError(f'Block {block.__name__} is not supported')

            downsample = conv1x1(self.inplanes, planes * block.expansion, stride=stride)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                **kan_kwargs,
            )
        )

        self.inplanes = planes * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.conv1(x)

        enc1 = self.layer1e(x)
        enc2 = self.layer2e(enc1)
        enc3 = self.layer3e(enc2)
        x = self.layer4e(enc3)

        x = self.layer4d(x)
        x = self.upsample(x)
        x = self.merge3(torch.concat([x, enc3], dim=1))
        x = self.layer3d(x)
        x = self.upsample(x)
        x = self.merge2(torch.concat([x, enc2], dim=1))
        x = self.layer2d(x)
        x = self.upsample(x)
        x = self.merge1(torch.concat([x, enc1], dim=1))
        x = self.layer1d(x)
        x = self.output(x)
        return x


class UKAGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        spline_order: int = 3,
        unet_depth: int = 4,
        unet_layers: int = 2,
        groups: int = 1,
        width_scale: int = 1,
        use_bottleneck: bool = True,
        mixer_type: Literal['conv', 'self-att', 'focal'] = 'conv',
        affine: bool = True,
        dropout: float = 0.0,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        inner_projection_attention: int | None = None,
        focal_window: int = 3,
        focal_level: int = 2,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = True,
        normalize_modulator: bool = True,
        full_kan: bool = True,
    ):
        super().__init__()
        self.unet_depth = unet_depth
        self.use_bottleneck = use_bottleneck

        attention = None
        if self.use_bottleneck:
            layer = BottleNeckKAGNConv2DLayer
            if mixer_type == 'self-att':
                attention = BottleNeckSelfKAGNtention2D
            if mixer_type == 'focal':
                attention = BottleNeckKAGNFocalModulation2D
        else:
            layer = KAGNConv2DLayer
            if mixer_type == 'self-att':
                attention = SelfKAGNtention2D
            if mixer_type == 'focal':
                attention = KAGNFocalModulation2D

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for depth_index in range(unet_depth):
            if depth_index == 0:
                layer_list_enc = [
                    layer(
                        in_channels=in_channels,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=0,
                        norm_layer=norm_layer,
                        affine=affine,
                    ),
                ] + [
                    layer(
                        in_channels=16 * width_scale * 2**depth_index,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for _ in range(unet_layers - 1)
                ]
            else:
                layer_list_enc = [
                    layer(
                        in_channels=16 * width_scale * 2 ** (depth_index - 1),
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    ),
                ] + [
                    layer(
                        in_channels=16 * width_scale * 2**depth_index,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for _ in range(unet_layers - 1)
                ]

            self.encoder.append(nn.Sequential(*layer_list_enc))

        for depth_index in reversed(range(0, unet_depth - 1)):
            if depth_index < unet_depth - 1:
                if attention is not None:
                    if mixer_type == 'self-att':
                        attention = cast(
                            type[SelfKAGNtention2D] | type[BottleNeckSelfKAGNtention2D],
                            attention,
                        )
                        layer_list_dec = [
                            attention(
                                input_dim=16 * 3 * width_scale * 2**depth_index,
                                inner_projection=inner_projection_attention,
                                spline_order=spline_order,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                dilation=1,
                                groups=groups,
                                norm_layer=norm_layer,
                                affine=affine,
                                dropout=dropout,
                            ),
                        ]
                    elif mixer_type == 'focal':
                        attention = cast(
                            type[KAGNFocalModulation2D]
                            | type[BottleNeckKAGNFocalModulation2D],
                            attention,
                        )
                        layer_list_dec = [
                            attention(
                                num_channels=16 * 3 * width_scale * 2**depth_index,
                                spline_order=spline_order,
                                dropout=dropout,
                                norm_layer=norm_layer,
                                affine=affine,
                                focal_window=focal_window,
                                focal_level=focal_level,
                                focal_factor=focal_factor,
                                use_postln_in_modulation=use_postln_in_modulation,
                                normalize_modulator=normalize_modulator,
                                full_kan=full_kan,
                            ),
                        ]
                    else:
                        layer_list_dec = []
                else:
                    layer_list_dec = []

                layer_list_dec += [
                    layer(
                        in_channels=16 * 3 * width_scale * 2**depth_index,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    ),
                ] + [
                    layer(
                        in_channels=16 * width_scale * 2**depth_index,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for _ in range(unet_layers - 1)
                ]

            else:
                if attention is not None:
                    if mixer_type == 'self-att':
                        attention = cast(
                            type[SelfKAGNtention2D] | type[BottleNeckSelfKAGNtention2D],
                            attention,
                        )
                        layer_list_dec = [
                            attention(
                                input_dim=16 * 3 * width_scale * 2**depth_index,
                                inner_projection=inner_projection_attention,
                                spline_order=spline_order,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                dilation=1,
                                groups=groups,
                                norm_layer=norm_layer,
                                affine=affine,
                                dropout=dropout,
                            ),
                        ]
                    elif mixer_type == 'focal':
                        attention = cast(
                            type[KAGNFocalModulation2D]
                            | type[BottleNeckKAGNFocalModulation2D],
                            attention,
                        )
                        layer_list_dec = [
                            attention(
                                16 * 3 * width_scale * 2**depth_index,
                                spline_order=spline_order,
                                dropout=dropout,
                                norm_layer=norm_layer,
                                affine=affine,
                                focal_window=focal_window,
                                focal_level=focal_level,
                                focal_factor=focal_factor,
                                use_postln_in_modulation=use_postln_in_modulation,
                                normalize_modulator=normalize_modulator,
                                full_kan=full_kan,
                            ),
                        ]
                    else:
                        layer_list_dec = []
                else:
                    layer_list_dec = []

                layer_list_dec += [
                    layer(
                        in_channels=16 * 3 * width_scale * 2**depth_index,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    ),
                ] + [
                    layer(
                        in_channels=16 * width_scale * 2**depth_index,
                        out_channels=16 * width_scale * 2**depth_index,
                        spline_order=spline_order,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        groups=groups,
                        dropout=dropout,
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for _ in range(unet_layers - 1)
                ]

            self.decoder.append(nn.Sequential(*layer_list_dec))
            self.output = nn.Conv2d(
                16 * width_scale * 2**depth_index,
                num_classes,
                kernel_size=1,
            )
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for block_index, block in enumerate(self.encoder):
            x = block(x)
            if block_index < self.unet_depth - 1:
                skips.append(x)

        for block in self.decoder:
            skip_x = skips.pop(-1)
            x = self.upsample(x)
            x = torch.concat([x, skip_x], dim=1)
            x = block(x)

        x = self.output(x)
        return x
