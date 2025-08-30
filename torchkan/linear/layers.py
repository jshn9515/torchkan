# taken from and based on https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/KALnet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, ZiYao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# and https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# and https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# and https://github.com/zavareh1/Wav-KAN
# and https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import RadialBasisFunction

from torchkan.utils.typing import Activation, WaveletType


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: Activation = nn.GELU(),
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # base_activation function used for the initial transformation of the input.
        self.base_activation = base_activation
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range
        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order)
        )
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(out_features)
        # Add a PReLU base_activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()

        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = (
            torch.linspace(
                self.grid_range[0] - h * spline_order,
                self.grid_range[1] + h * spline_order,
                grid_size + 2 * spline_order + 1,
                dtype=torch.float32,
            )
            .expand(in_features, -1)
            .contiguous()
        )

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')

    def forward(self, x: Tensor) -> Tensor:
        # Process each layer using the defined base weights, spline weights, norms, and base_activations.
        grid = self.grid.to(
            x.device
        )  # Move the input tensor to the device where the weights are located.

        # Perform the base linear transformation followed by the base_activation function.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = torch.unsqueeze(x, dim=-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        bases = torch.logical_and(x_uns >= grid[:, :-1], x_uns < grid[:, 1:])
        bases = bases.to(device=x.device, dtype=x.dtype)
        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[:, : -(k + 1)]
            right_intervals = grid[:, k:-1]
            delta = torch.where(
                right_intervals == left_intervals,
                torch.ones_like(right_intervals),
                right_intervals - left_intervals,
            )
            bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + (
                (grid[:, k + 1 :] - x_uns)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        bases = bases.contiguous()

        # Compute the spline transformation and combine it with the base transformation.
        spline_output = F.linear(
            bases.view(x.size(0), -1),
            self.spline_weight.view(self.spline_weight.size(0), -1),
        )
        # Apply layer normalization and PReLU base_activation to the combined output.
        x = self.prelu(self.layer_norm(base_output + spline_output))

        return x


class KALNLayer(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int = 3,
        base_activation: Activation = nn.SiLU(),
    ):
        super().__init__()  # Initialize the parent nn.Module class

        self.in_features = in_features
        self.out_features = out_features
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = spline_order
        # base_activation: base_activation function used after each layer's computation
        self.base_activation = base_activation

        # Base weight for linear transformation in each layer
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        # Polynomial weight for handling Legendre polynomial expansions
        self.poly_weight = nn.Parameter(
            torch.randn(out_features, in_features * (spline_order + 1))
        )
        # Layer normalization to stabilize learning and outputs
        self.layer_norm = nn.LayerNorm(out_features)

        # Initialize weights using Kaiming uniform distribution for better training start
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid re-computation of Legendre polynomials
    def compute_legendre_polynomials(self, x: Tensor, order: int) -> Tensor:
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return torch.unsqueeze(P0, dim=-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (
                n + 1.0
            )
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        # Apply base base_activation to input and then linear transform with base weights
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        # Compute Legendre polynomials for the normalized x
        legendre_basis = self.compute_legendre_polynomials(
            x_normalized, self.polynomial_order
        )
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        legendre_basis = legendre_basis.view(x.size(0), -1)

        # Compute polynomial output using polynomial weights
        poly_output = F.linear(legendre_basis, self.poly_weight)
        # Combine base and polynomial outputs, normalize, and base_activation
        x = self.base_activation(self.layer_norm(base_output + poly_output))

        return x


class SplineLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kwargs
    ):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kwargs)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation: Activation = nn.SiLU(),
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            in_features * num_grids, out_features, spline_weight_init_scale
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, time_benchmark: bool = False) -> Tensor:
        if not time_benchmark:
            spline_basis = self.rbf(self.layer_norm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, spline_order: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order

        self.cheby_coeff = nn.Parameter(
            torch.empty(in_features, out_features, spline_order + 1)
        )
        nn.init.normal_(
            self.cheby_coeff, mean=0.0, std=1 / (in_features * (spline_order + 1))
        )
        # self.register_buffer('arange', torch.arange(0, spline_order + 1, 1))
        self.arange = torch.arange(0, spline_order + 1, 1, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input spline_order + 1 times
        # shape = (batch_size, in_features, self.spline_order + 1)
        x = x.view(-1, self.in_features, 1).expand(-1, -1, self.spline_order + 1)
        # Apply acos
        x = torch.acos(x)
        # Multiply by arange [0 ... spline_order]
        x *= self.arange
        # Apply cos
        x = torch.cos(x)
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            'bid,iod->bo', x, self.cheby_coeff
        )  # shape = (batch_size, out_features)
        y = y.view(-1, self.out_features)
        return y


class GRAMLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int = 3,
        base_activation: Activation = nn.SiLU(),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.norm = nn.LayerNorm(out_features, dtype=torch.float32)
        self.beta_weights = nn.Parameter(
            torch.zeros(spline_order + 1, dtype=torch.float32)
        )

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(
                in_features, out_features, spline_order + 1, dtype=torch.float32
            )
        )
        self.base_weights = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float32)
        )
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_features * (self.spline_order + 1.0)),
        )
        nn.init.xavier_uniform_(self.grams_basis_weights)
        nn.init.xavier_uniform_(self.base_weights)

    def beta(self, n: int, m: int) -> Tensor:
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x: Tensor) -> Tensor:
        P0 = x.new_ones(x.size())
        if self.spline_order == 0:
            return torch.unsqueeze(P0, dim=-1)

        P1 = x
        grams_basis = [P0, P1]

        for i in range(2, self.spline_order + 1):
            P2 = x * P1 - self.beta(i - 1, i) * P0
            grams_basis.append(P2)
            P0, P1 = P1, P2

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        basis = F.linear(self.base_activation(x), self.base_weights)
        x = torch.tanh(x).contiguous()
        grams_basis = self.base_activation(self.gram_poly(x))

        y = torch.einsum('bid,iod->bo', grams_basis, self.grams_basis_weights)

        y = self.base_activation(self.norm(y + basis))
        y = y.view(-1, self.out_features)
        return y


class WavKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet_type: WaveletType = 'mexican_hat',
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # not used; you may like to use it for weighting base activation and adding it like Spl-KAN paper
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Base base_activation function # not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x: Tensor):
        if x.ndim == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        match self.wavelet_type:
            case 'mexican_hat':
                term1 = (x_scaled**2) - 1
                term2 = torch.exp(-0.5 * x_scaled**2)
                wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2

            case 'morlet':
                omega0 = 5.0  # Central frequency
                real = torch.cos(omega0 * x_scaled)
                envelope = torch.exp(-0.5 * x_scaled**2)
                wavelet = envelope * real

            case 'dog':
                # Implementing Derivative of Gaussian Wavelet
                wavelet = -x_scaled * torch.exp(-0.5 * x_scaled**2)

            case 'meyer':
                # Implement Meyer Wavelet here
                # Constants for the Meyer wavelet transition boundaries
                v = torch.abs(x_scaled)

                def meyer_aux(v: Tensor) -> Tensor:
                    return torch.where(
                        v <= 1 / 2,
                        torch.ones_like(v),
                        torch.where(
                            v >= 1,
                            torch.zeros_like(v),
                            torch.cos(torch.pi / 2 * nu(2 * v - 1)),
                        ),
                    )

                def nu(t: Tensor) -> Tensor:
                    return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)

                # Meyer wavelet calculation using the auxiliary function
                wavelet = torch.sin(torch.pi * v) * meyer_aux(v)

            case 'shannon':
                # Windowing the sinc function to limit its support
                sinc = torch.sinc(x_scaled / torch.pi)  # sinc(x) = sin(pi*x) / (pi*x)

                # Applying a Hamming window to limit the infinite support of the sinc function
                window = torch.hamming_window(
                    x_scaled.size(-1),
                    periodic=False,
                    dtype=x_scaled.dtype,
                    device=x_scaled.device,
                )
                # Shannon wavelet is the product of the sinc function and the window
                wavelet = sinc * window

                # You can try many more wavelet types ...
            case _:
                raise TypeError(f'Unsupported wavelet type: {self.wavelet_type}')

        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(
            wavelet
        )
        wavelet_output = wavelet_weighted.sum(dim=2)
        return wavelet_output

    def forward(self, x: Tensor) -> Tensor:
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        base_output = F.linear(self.base_activation(x), self.weight)
        # base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output + base_output
        # Apply batch normalization
        return self.bn(combined_output)


class JacobiKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int,
        a: float = 1.0,
        b: float = 1.0,
        base_activation: Activation = nn.SiLU(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = a
        self.b = b
        self.spline_order = spline_order

        self.base_activation = base_activation
        self.norm = nn.LayerNorm(out_features, dtype=torch.float32)

        self.base_weights = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float32)
        )

        self.jacobi_coeff = nn.Parameter(
            torch.empty(in_features, out_features, spline_order + 1)
        )

        nn.init.normal_(
            self.jacobi_coeff, mean=0.0, std=1 / (in_features * (spline_order + 1))
        )
        nn.init.xavier_uniform_(self.base_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, self.in_features)  # shape = (batch_size, in_features)

        basis = F.linear(self.base_activation(x), self.base_weights)

        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(
            x.shape[0], self.in_features, self.spline_order + 1, device=x.device
        )
        if self.spline_order > 0:
            # spline_order = 0: jacobi[:, :, 0] = 1 (already initialized); spline_order = 1: jacobi[:, :, 1] = x;
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.spline_order + 1):
            theta_k = (
                (2 * i + self.a + self.b)
                * (2 * i + self.a + self.b - 1)
                / (2 * i * (i + self.a + self.b))
            )
            theta_k1 = (
                (2 * i + self.a + self.b - 1)
                * (self.a * self.a - self.b * self.b)
                / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            theta_k2 = (
                (i + self.a - 1)
                * (i + self.b - 1)
                * (2 * i + self.a + self.b)
                / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[
                :, :, i - 1
            ].clone() - theta_k2 * jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum(
            'bid,iod->bo', jacobi, self.jacobi_coeff
        )  # shape = (batch_size, out_features)
        y = y.view(-1, self.out_features)

        y = self.base_activation(self.norm(y + basis))
        return y


class BernsteinKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int,
        base_activation: Activation = nn.SiLU(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order
        self.norm = nn.LayerNorm(out_features, dtype=torch.float32)

        self.base_weights = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float32)
        )

        self.bernstein_coeff = nn.Parameter(
            torch.empty(in_features, out_features, spline_order + 1)
        )
        self.base_activation = base_activation

        nn.init.normal_(
            self.bernstein_coeff, mean=0.0, std=1 / (in_features * (spline_order + 1))
        )
        nn.init.xavier_uniform_(self.base_weights)

    @lru_cache(maxsize=128)
    def bernstein_poly(self, x: Tensor, spline_order: int) -> Tensor:
        bernsteins = torch.ones(
            x.shape + (self.spline_order + 1,), dtype=x.dtype, device=x.device
        )
        for j in range(1, spline_order + 1):
            for k in range(spline_order + 1 - j):
                bernsteins[..., k] = (
                    bernsteins[..., k] * (1 - x) + bernsteins[..., k + 1] * x
                )
        return bernsteins

    def forward(self, x: Tensor) -> Tensor:
        x = torch.reshape(
            x, (-1, self.in_features)
        )  # shape = (batch_size, in_features)

        basis = F.linear(self.base_activation(x), self.base_weights)

        # Since Bernstein polynomial is defined in [0, 1]
        # We need to normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        bernsteins = self.bernstein_poly(x, self.spline_order)
        y = torch.einsum(
            'bid,iod->bo', bernsteins, self.bernstein_coeff
        )  # shape = (batch_size, out_features)
        y = y.view(-1, self.out_features)

        y = self.base_activation(self.norm(y + basis))
        return y


class ReLUKANLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, g: int, k: int, train_ab: bool = True
    ):
        super().__init__()
        self.g = g
        self.k = k
        self.r = 4 * g * g / ((k + 1) * (k + 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        # modification here
        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g
        # modification here
        self.phase_low = nn.Parameter(
            phase_low[None, :].expand(in_channels, -1), requires_grad=train_ab
        )
        # modification here, and: `phase_height` to `phase_high`
        self.phase_high = nn.Parameter(
            phase_high[None, :].expand(in_channels, -1), requires_grad=train_ab
        )
        self.equal_size_conv = nn.Conv2d(
            1, out_channels, kernel_size=(g + k, in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x[..., None]
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x * x1 * x2 * self.r
        x = x.reshape(len(x), 1, self.g + self.k, self.in_channels)
        x = self.equal_size_conv(x)
        x = x.reshape(len(x), self.out_channels)
        return x


class BottleNeckGRAMLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int = 3,
        base_activation: Activation = nn.SiLU(),
        dim_reduction: float = 8,
        min_internal: int = 16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order

        self.dim_reduction = dim_reduction
        self.min_internal = min_internal

        inner_dim = int(max(in_features / dim_reduction, out_features / dim_reduction))
        if inner_dim < min_internal:
            self.inner_dim = min(min_internal, in_features, out_features)
        else:
            self.inner_dim = inner_dim

        self.base_activation = base_activation

        self.inner_proj = nn.Linear(in_features, self.inner_dim)
        self.outer_proj = nn.Linear(self.inner_dim, out_features)

        self.norm = nn.LayerNorm(out_features, dtype=torch.float32)

        self.beta_weights = nn.Parameter(
            torch.zeros(spline_order + 1, dtype=torch.float32)
        )

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(
                self.inner_dim, self.inner_dim, spline_order + 1, dtype=torch.float32
            )
        )

        self.base_weights = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_features * (self.spline_order + 1.0)),
        )
        nn.init.xavier_uniform_(self.grams_basis_weights)
        nn.init.xavier_uniform_(self.base_weights)
        nn.init.xavier_uniform_(self.inner_proj.weight)
        nn.init.xavier_uniform_(self.outer_proj.weight)

    def beta(self, n: int, m: int) -> Tensor:
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x: Tensor) -> Tensor:
        P0 = x.new_ones(x.size())

        if self.spline_order == 0:
            return torch.unsqueeze(P0, dim=-1)

        P1 = x
        grams_basis = [P0, P1]

        for i in range(2, self.spline_order + 1):
            P2 = x * P1 - self.beta(i - 1, i) * P0
            grams_basis.append(P2)
            P0, P1 = P1, P2

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        basis = F.linear(self.base_activation(x), self.base_weights)
        x = self.inner_proj(x)
        x = torch.tanh(x).contiguous()
        grams_basis = self.base_activation(self.gram_poly(x))

        y = torch.einsum('bid,iod->bo', grams_basis, self.grams_basis_weights)

        y = self.outer_proj(y)
        y = self.base_activation(self.norm(y + basis))
        y = y.view(-1, self.out_features)
        return y
