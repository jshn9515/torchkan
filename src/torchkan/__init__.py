from torch.nn import Module

from .conv import *  # noqa: F403
from .linear import *  # noqa: F403


def get_all_kan_layers() -> list[str]:
    return [
        name
        for name, obj in globals().items()
        if isinstance(obj, type) and issubclass(obj, Module)
    ]
