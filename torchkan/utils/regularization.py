# Based on this implementations: https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/regularization.py

import abc

import torch
import torch.nn as nn
from torch import Tensor


class NoiseInjection(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def get_noise(self, x: Tensor) -> Tensor:
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        if x.numel() > 1:
            std = torch.std(x, dim=dims, keepdim=True)
        else:
            std = torch.tensor(0.0)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float()
            x = x + self.alpha * mask * self.get_noise(x)
            return x
        return x


class NoiseMultiplicativeInjection(nn.Module):
    def __init__(self, p: float = 0.05, alpha: float = 0.05, betta: float = 0.01):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.betta = betta

    def get_noise(self, x: Tensor) -> Tensor:
        dims = tuple(i for i in range(x.ndim) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def get_m_noise(self, x: Tensor) -> Tensor:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * self.betta + 1
        return noise

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            mask_m = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask_m = (mask_m < self.p).float() * 1
            x = (
                x
                + x * mask_m * self.get_m_noise(x)
                + self.alpha * mask * self.get_noise(x)
            )
            return x
        return x


class WeightDecay(nn.Module):
    def __init__(self, module: nn.Module, weight_decay: float, name: str | None = None):
        if weight_decay < 0.0:
            raise ValueError(
                f"Regularization's weight_decay should be greater than 0.0, got {weight_decay}"
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *args):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.allclose(param.grad, torch.tensor(0.0)):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.allclose(param.grad, torch.tensor(0.0))
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = f'weight_decay={self.weight_decay}'
        if self.name:
            representation += f', name={self.name}'
        return representation

    @abc.abstractmethod
    def regularize(self, parameter: nn.Parameter) -> Tensor:
        pass


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").
    """

    def regularize(self, parameter: nn.Parameter) -> Tensor:
        return self.weight_decay * torch.sign(parameter.data)


class L2(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").
    """

    def regularize(self, parameter: nn.Parameter) -> Tensor:
        return self.weight_decay * parameter.data
