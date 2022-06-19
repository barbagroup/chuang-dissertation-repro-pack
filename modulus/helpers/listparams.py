#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Operations for lists of parameters.
"""
from typing import List as _List
from typing import Union as _Union
from torch import Tensor as _Tensor
from torch import zeros as _torchzeros
from torch import maximum as _torchmaximum
from torch import add as _torchadd
from torch import sub as _torchminus
from torch import mul as _torchmul
from torch import div as _torchdiv
from torch import jit as _torchjit

Vector = _List[_Tensor]


@_torchjit.script
def detach(params: Vector) -> Vector:
    """Return a new list of tensors that are all deatched from the graph.
    """
    new: Vector = []
    for param in params:
        new.append(param.detach())
    return new


@_torchjit.script
def neg(params: Vector) -> Vector:
    """Return a new list of tensors that are all multily with a negative signe.
    """
    new: Vector = []
    for param in params:
        new.append(-param)
    return new


@_torchjit.script
def notzero(params: Vector) -> bool:
    """Return whether all elements in all parameters are zeros.
    """
    result = True
    for param in params:
        result = (result and bool((param != 0).all()))
        if not result:
            break
    return result


@_torchjit.script
def linfnorm(params: Vector) -> _Tensor:
    """Return the infinity norm.
    """
    ans: _Tensor = _torchzeros((), dtype=params[0].dtype, device=params[0].device)
    for param in params:
        ans = _torchmaximum(ans, param.abs().max())
    return ans


@_torchjit.script
def l2norm(params: Vector) -> _Tensor:
    """Return the infinity norm.
    """
    ans: _Tensor = _torchzeros((), dtype=params[0].dtype, device=params[0].device)
    for param in params:
        ans = ans + (param**2).sum()
    return ans.sqrt()


@_torchjit.script
def dot(left: Vector, right: Vector) -> _Tensor:
    """Perform left being divided by right.
    """
    new: _Tensor = _torchzeros(len(left), dtype=left[0].dtype, device=left[0].device)
    for i, (lparam, rparam) in enumerate(zip(left, right)):
        new[i] = (lparam*rparam).sum()
    return new.sum()


def elementwise_factory(operation):
    """Factory to create elementwise operation.
    """

    def list_operation(left: _Union[_Tensor, Vector, int, float], right: _Union[_Tensor, Vector, int, float]) -> Vector:
        """Elementwise operation.
        """
        new: Vector = []
        if isinstance(left, list) and isinstance(right, list):
            for lparam, rparam in zip(left, right):
                new.append(operation(lparam, rparam))
            return new

        if isinstance(left, list) and isinstance(right, _Tensor):
            for lparam in left:
                new.append(operation(lparam, right))
            return new

        if isinstance(left, list) and isinstance(right, float):
            for lparam in left:
                new.append(operation(lparam, right))
            return new

        if isinstance(left, list) and isinstance(right, int):
            for lparam in left:
                new.append(operation(lparam, right))
            return new

        if isinstance(left, _Tensor) and isinstance(right, list):
            for rparam in right:
                new.append(operation(left, rparam))
            return new

        if isinstance(left, float) and isinstance(right, list):
            for rparam in right:
                new.append(operation(left, rparam))
            return new

        if isinstance(left, int) and isinstance(right, list):
            for rparam in right:
                new.append(operation(left, rparam))
            return new

        # torchscript hates return a Union, so sorry to Tensor-Tensor or other non-Vector operations
        raise ValueError("Both inputs are torch.Tensors. Use regular PyTorch functions.")

    return _torchjit.script(list_operation)


plus = elementwise_factory(_torchadd)
minus = elementwise_factory(_torchminus)
mul = elementwise_factory(_torchmul)
div = elementwise_factory(_torchdiv)
