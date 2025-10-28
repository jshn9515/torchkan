# ruff: noqa: I001
from .regularization import (
    L1 as L1,
    L2 as L2,
    NoiseInjection as NoiseInjection,
    NoiseMultiplicativeInjection as NoiseMultiplicativeInjection,
)
from .normalization import (
    SelfSpatialNorm as SelfSpatialNorm,
    SpatialNorm as SpatialNorm,
)
