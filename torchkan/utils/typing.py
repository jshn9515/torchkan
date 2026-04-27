from collections.abc import Callable
from typing import Literal

import torch.nn as nn
from torch import Tensor

type Size1D = int
type Size2D= int | tuple[int, int]
type Size3D= int | tuple[int, int, int]
type SizeND= int | tuple[int, ...]

type PaddingType= Literal['valid', 'same']
type Padding1D= PaddingType | Size1D
type Padding2D= PaddingType | Size2D
type Padding3D= PaddingType | Size3D
type PaddingND= PaddingType | SizeND

type NDim= Literal[1, 2, 3]
type ConvFunc= Callable[..., nn.Module]
type Activation= Callable[[Tensor], Tensor]

type WaveletType= Literal['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']
type WaveletVersion= Literal['base', 'fast', 'fast_plus_one']
