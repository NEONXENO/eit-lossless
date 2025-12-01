"""A lightweight torch compatibility stub for environments without PyTorch.
This stub implements just enough Tensor behavior for unit tests using NumPy arrays.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Optional

float32 = np.float32
float16 = np.float16
bfloat16 = np.float16  # approximate using float16
dtype = np.dtype
bool = np.bool_

class Tensor(np.ndarray):
    __array_priority__ = 1000

    def __new__(cls, input_array: Any, dtype: Any | None = None):
        arr = np.array(input_array, dtype=dtype)
        obj = np.asarray(arr).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def device(self) -> str:
        return "cpu"

    def clone(self) -> "Tensor":
        return Tensor(np.array(self, copy=True))

    def detach(self) -> "Tensor":
        return self.clone()

    def to(self, dtype: Any = None) -> "Tensor":
        if dtype is None:
            return self
        return Tensor(self.astype(dtype))

    def view(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.reshape(self, shape))

    def sum(self, *args: Any, **kwargs: Any):
        base = np.asarray(self)
        return Tensor(np.sum(base, *args, **kwargs))

    def item(self) -> Any:
        return np.ndarray.item(self)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def copy_(self, other: "Tensor"):
        np.copyto(self, np.asarray(other))
        return self


BoolTensor = Tensor


def tensor(data: Any, dtype: Any | None = None) -> Tensor:
    return Tensor(data, dtype=dtype)


def zeros(*shape: int, dtype: Any = float32, device: Optional[str] = None) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype))


def zeros_like(t: Tensor) -> Tensor:
    return Tensor(np.zeros_like(t))


def randn(*shape: int, device: Optional[str] = None, dtype: Any = float32) -> Tensor:
    return Tensor(np.random.randn(*shape).astype(dtype))


def randn_like(t: Tensor) -> Tensor:
    return Tensor(np.random.randn(*t.shape).astype(t.dtype))


def rand(*shape: int, device: Optional[str] = None) -> Tensor:
    return Tensor(np.random.rand(*shape))


def mean(t: Tensor) -> Tensor:
    return Tensor(np.mean(t))


def allclose(a: Tensor, b: Tensor, atol: float = 1e-8) -> bool:
    return np.allclose(a, b, atol=atol)


def no_grad(func=None):
    def decorator(f):
        return f

    if func is None:
        return decorator
    return decorator(func)

# Submodules
class _CudaModule:
    @staticmethod
    def is_available() -> bool:
        return False

class _NNModule:
    class Module:
        def __init__(self):
            pass

        def __call__(self, *args: Any, **kwargs: Any):
            return self.forward(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any):
            raise NotImplementedError


# expose submodules for import torch.nn / torch.cuda
import sys
import types

cuda = types.ModuleType('torch.cuda')
cuda.is_available = _CudaModule.is_available

nn = types.ModuleType('torch.nn')
nn.Module = _NNModule.Module

sys.modules[__name__ + '.cuda'] = cuda
sys.modules[__name__ + '.nn'] = nn

__all__ = [
    "Tensor",
    "tensor",
    "zeros",
    "zeros_like",
    "randn",
    "randn_like",
    "rand",
    "mean",
    "allclose",
    "float32",
    "float16",
    "bfloat16",
    "no_grad",
    "cuda",
    "nn",
]
