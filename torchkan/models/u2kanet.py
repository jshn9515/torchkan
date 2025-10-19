import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchkan.utils.typing import ConvFunc


def upsample_like(src: Tensor, tar: Tensor) -> Tensor:
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


class ResidualUNetBase(nn.Module):
    def __init__(
        self,
        conv_func: ConvFunc,
        conf_fun_first: ConvFunc | None = None,
        depth: int = 7,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        assert depth >= 4, f'Minimum supported depth = 4, but provided {depth}.'
        self.depth = depth

        if conf_fun_first is not None:
            self.input_conv = conf_fun_first(in_channels, out_channels, dilation=1)
        else:
            self.input_conv = conv_func(in_channels, out_channels, dilation=1)

        self.encoder_list = nn.ModuleList(
            [
                conv_func(
                    mid_channels if i > 0 else out_channels,
                    mid_channels,
                    dilation=1 if i < depth - 1 else 2,
                )
                for i in range(depth)
            ]
        )
        self.decoder_list = nn.ModuleList(
            [
                conv_func(
                    mid_channels * 2,
                    mid_channels if i < depth - 2 else out_channels,
                    dilation=1,
                )
                for i in range(depth - 1)
            ]
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
        x_input = self.input_conv(x)

        x_enc = x_input
        encoder_list = []
        for layer_index, layer in enumerate(self.encoder_list):
            x_enc = layer(x_enc)
            # save this to something
            if layer_index < self.depth - 1:
                encoder_list.append(x_enc)
            if layer_index < self.depth - 2:
                x_enc = self.pool(x_enc)

        x_dec = x_enc
        for layer_index, layer in enumerate(self.decoder_list):
            skip = encoder_list.pop()
            x_dec = layer(torch.concat((x_dec, skip), dim=1))
            if layer_index < self.depth - 2:
                x_dec = self.upsample(x_dec)

        x = x_dec + x_input
        return x


class ResidualUNetBaseF(nn.Module):
    def __init__(
        self,
        conv_func: ConvFunc,
        depth: int = 4,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        assert depth >= 4, f'Minimum supported depth = 4, but provided {depth}.'
        self.depth = depth

        self.input_conv = conv_func(in_channels, out_channels, dilation=1)
        self.encoder_list = nn.ModuleList(
            [
                conv_func(
                    mid_channels if i > 0 else out_channels, mid_channels, dilation=2**i
                )
                for i in range(depth)
            ]
        )
        self.decoder_list = nn.ModuleList(
            [
                conv_func(
                    mid_channels * 2,
                    mid_channels if i < depth - 2 else out_channels,
                    dilation=2 ** (depth - 2 - i),
                )
                for i in range(depth - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x_input = self.input_conv(x)

        x_enc = x_input
        encoder_list = []
        for layer_index, layer in enumerate(self.encoder_list):
            x_enc = layer(x_enc)
            # save this to something
            if layer_index < self.depth - 1:
                encoder_list.append(x_enc)

        x_dec = x_enc
        for layer_index, layer in enumerate(self.decoder_list):
            x_dec = layer(torch.concat((x_dec, encoder_list.pop()), 1))

        x = x_dec + x_input
        return x


class U2KANet(nn.Module):
    def __init__(
        self,
        conv_func: ConvFunc,
        conf_fun_first: ConvFunc | None = None,
        in_channels: int = 3,
        out_channels: int = 1,
        width_factor: int = 1,
    ):
        super().__init__()
        self.stage1 = ResidualUNetBase(
            conv_func=conv_func,
            conf_fun_first=conf_fun_first,
            depth=7,
            in_channels=in_channels,
            mid_channels=8 * width_factor,
            out_channels=16 * width_factor,
        )
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = ResidualUNetBase(
            conv_func=conv_func,
            depth=6,
            in_channels=16 * width_factor,
            mid_channels=8 * width_factor,
            out_channels=32 * width_factor,
        )
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = ResidualUNetBase(
            conv_func=conv_func,
            depth=5,
            in_channels=32 * width_factor,
            mid_channels=16 * width_factor,
            out_channels=64 * width_factor,
        )
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = ResidualUNetBase(
            conv_func=conv_func,
            depth=4,
            in_channels=64 * width_factor,
            mid_channels=32 * width_factor,
            out_channels=128 * width_factor,
        )
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = ResidualUNetBaseF(
            conv_func=conv_func,
            depth=4,
            in_channels=128 * width_factor,
            mid_channels=64 * width_factor,
            out_channels=128 * width_factor,
        )
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = ResidualUNetBaseF(
            conv_func=conv_func,
            depth=4,
            in_channels=128 * width_factor,
            mid_channels=64 * width_factor,
            out_channels=128 * width_factor,
        )

        # decoder
        self.stage5d = ResidualUNetBaseF(
            conv_func=conv_func,
            depth=4,
            in_channels=256 * width_factor,
            mid_channels=64 * width_factor,
            out_channels=128 * width_factor,
        )
        self.stage4d = ResidualUNetBase(
            conv_func=conv_func,
            depth=4,
            in_channels=256 * width_factor,
            mid_channels=32 * width_factor,
            out_channels=64 * width_factor,
        )
        self.stage3d = ResidualUNetBase(
            conv_func=conv_func,
            depth=5,
            in_channels=128 * width_factor,
            mid_channels=16 * width_factor,
            out_channels=32 * width_factor,
        )
        self.stage2d = ResidualUNetBase(
            conv_func=conv_func,
            depth=6,
            in_channels=64 * width_factor,
            mid_channels=8 * width_factor,
            out_channels=16 * width_factor,
        )
        self.stage1d = ResidualUNetBase(
            conv_func=conv_func,
            depth=7,
            in_channels=32 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )

        self.side1 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side2 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side3 = nn.Conv2d(
            32 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side4 = nn.Conv2d(
            64 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side5 = nn.Conv2d(
            128 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side6 = nn.Conv2d(
            128 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.outconv = nn.Conv2d(
            6 * out_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        hx = x
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        # stage 6
        hx6 = self.stage6(hx)
        hx6up = upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.concat((hx6up, hx5), 1))
        hx5dup = upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.concat((hx5dup, hx4), 1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.concat((hx4dup, hx3), 1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.concat((hx3dup, hx2), 1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.concat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = upsample_like(d4, d1)
        d5 = self.side5(hx5d)
        d5 = upsample_like(d5, d1)
        d6 = self.side6(hx6)
        d6 = upsample_like(d6, d1)
        d0 = self.outconv(torch.concat((d1, d2, d3, d4, d5, d6), dim=1))
        return d0, d1, d2, d3, d4, d5, d6


class TinyU2KANet(nn.Module):
    def __init__(
        self,
        conv_func: ConvFunc,
        conf_fun_first: ConvFunc | None = None,
        in_channels: int = 3,
        out_channels: int = 1,
        width_factor: int = 1,
    ):
        super().__init__()
        self.stage1 = ResidualUNetBase(
            conv_func=conv_func,
            conf_fun_first=conf_fun_first,
            depth=7,
            in_channels=in_channels,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = ResidualUNetBase(
            conv_func=conv_func,
            depth=6,
            in_channels=16 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = ResidualUNetBase(
            conv_func=conv_func,
            depth=5,
            in_channels=16 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = ResidualUNetBase(
            conv_func=conv_func,
            depth=4,
            in_channels=16 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = ResidualUNetBaseF(
            conv_func=conv_func,
            depth=4,
            in_channels=16 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = ResidualUNetBaseF(
            conv_func=conv_func,
            depth=4,
            in_channels=16 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )

        # decoder
        self.stage5d = ResidualUNetBaseF(
            conv_func=conv_func,
            depth=4,
            in_channels=32 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.stage4d = ResidualUNetBase(
            conv_func=conv_func,
            depth=4,
            in_channels=32 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.stage3d = ResidualUNetBase(
            conv_func=conv_func,
            depth=5,
            in_channels=32 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.stage2d = ResidualUNetBase(
            conv_func=conv_func,
            depth=6,
            in_channels=32 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )
        self.stage1d = ResidualUNetBase(
            conv_func=conv_func,
            depth=7,
            in_channels=32 * width_factor,
            mid_channels=4 * width_factor,
            out_channels=16 * width_factor,
        )

        self.side1 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side2 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side3 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side4 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side5 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.side6 = nn.Conv2d(
            16 * width_factor,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.outconv = nn.Conv2d(
            6 * out_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        hx = x
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        # stage 6
        hx6 = self.stage6(hx)
        hx6up = upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.concat((hx6up, hx5), dim=1))
        hx5dup = upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.concat((hx5dup, hx4), dim=1))
        hx4dup = upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.concat((hx4dup, hx3), dim=1))
        hx3dup = upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.concat((hx3dup, hx2), dim=1))
        hx2dup = upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.concat((hx2dup, hx1), dim=1))

        # side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = upsample_like(d4, d1)
        d5 = self.side5(hx5d)
        d5 = upsample_like(d5, d1)
        d6 = self.side6(hx6)
        d6 = upsample_like(d6, d1)
        d0 = self.outconv(torch.concat((d1, d2, d3, d4, d5, d6), dim=1))
        return d0, d1, d2, d3, d4, d5, d6
