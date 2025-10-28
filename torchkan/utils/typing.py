from collections.abc import Callable
from typing import Literal, TypeAlias

import torch.nn as nn
from torch import Tensor

Size1D: TypeAlias = int
Size2D: TypeAlias = int | tuple[int, int]
Size3D: TypeAlias = int | tuple[int, int, int]
SizeND: TypeAlias = int | tuple[int, ...]

PaddingType: TypeAlias = Literal['valid', 'same']
Padding1D: TypeAlias = PaddingType | Size1D
Padding2D: TypeAlias = PaddingType | Size2D
Padding3D: TypeAlias = PaddingType | Size3D
PaddingND: TypeAlias = PaddingType | SizeND

NDim: TypeAlias = Literal[1, 2, 3]
ConvFunc: TypeAlias = Callable[..., nn.Module]
Activation: TypeAlias = Callable[[Tensor], Tensor]

WaveletType: TypeAlias = Literal['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']
WaveletVersion: TypeAlias = Literal['base', 'fast', 'fast_plus_one']
