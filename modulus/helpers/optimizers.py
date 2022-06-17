#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Optimizers
"""

# otherwise, import normally
from os import devnull as _devnull
from io import TextIOWrapper as _TextIOWrapper
from typing import Callable as _Callable
from typing import Tuple as _Tuple
from typing import Optional as _Optional
from torch import Tensor as _Tensor
from torch import tensor as _torchtensor
from torch import zeros as _torchzeros
from torch import zeros_like as _zeros_like
from torch import jit as _torchjit
from torch import no_grad as _no_grad
from torch import maximum as _torchmax
from torch import minimum as _torchmin
from torch import isnan as _torchisnan
from torch import isinf as _torchisinf

# to make relative import work
if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from linesearch import linesearch as _linesearch
    from linesearch import HZLineSearchConf as _HZLineSearchConf
else:
    from .linesearch import linesearch as _linesearch
    from .linesearch import HZLineSearchConf as _HZLineSearchConf


def polak_rebiere_plus(gold: _Tensor, gnew: _Tensor) -> _Tensor:
    """Polak-Ribiere Plus algorithm to determine beta parameter for CG solvers.
    """
    beta: _Tensor = gnew.dot(gnew-gold) / gold.dot(gold)
    beta = max(beta, _zeros_like(beta))  # adjust beta to avoid negative beta
    return beta


@_torchjit.script
def hager_zhang_dynamic(gold: _Tensor, gnew: _Tensor, pold: _Tensor, eta: _Tensor) -> _Tensor:
    """The dynamic version of the Hager-Zhang (2005) algorithm to determine beta parameter for CG solvers.
    """
    # calcuate beta
    y: _Tensor = gnew - gold
    pdoty: _Tensor = pold.dot(y)
    beta: _Tensor = (y - 2. * pold * y.dot(y) / pdoty).dot(gnew) / pdoty

    # adjust beta to avoid negative beta
    eta: _Tensor = - 1.0 / (pold.dot(pold).sqrt() * _torchmin(eta, gold.dot(gold).sqrt()))
    beta: _Tensor = _torchmax(beta, eta)

    # nan/inf happens when gradients do not change (especially when using SINGLE PRECISION floats)
    # in this case, the next searching direction should just be the negative of gradients, so beta = 0
    if _torchisnan(beta) or _torchisinf(beta):
        beta = _torchzeros((), dtype=beta.dtype, device=beta.device)

    return beta


def nonlinear_cg(
    objective: _Callable[[_Tensor], _Tensor],
    params_init: _Tensor,
    max_iters: int = 1000,
    tol: float = 1e-7,
    disp: _Optional[_TextIOWrapper] = None,
    **kwargs
):
    """Nonlinear conjugate-gradient optimizer.
    """

    # null writer if disp is not provided
    if disp is None:
        disp = open(_devnull, "w")

    # initialize values
    params: _Tensor = params_init.detach()  # current location on the parameter hyperplane
    loss, grad = objective(params)
    conj: _Tensor = - grad.detach()  # the current searching direction

    # initialize other variables
    alpha: _Tensor = _torchzeros((), dtype=params.dtype, device=params.device)
    tol: _Tensor = _torchtensor(tol, dtype=params.dtype, device=params.device)
    Qk: _Tensor = _torchzeros((), dtype=params.dtype, device=params.device)
    Ck: _Tensor = _torchzeros((), dtype=params.dtype, device=params.device)

    # configuration of Hager-Zhang line search algorithm
    config: _HZLineSearchConf = _HZLineSearchConf()
    config.update(kwargs)
    config.set("tol", tol)

    for iters in range(max_iters):

        def phi(alpha: _Tensor) -> _Tuple[_Tensor, _Tensor]:
            _loss, _grad = objective(params+alpha*conj)
            _loss = _loss.detach()
            _grad = _grad.detach().dot(conj).detach()
            return _loss, _grad

        # update relative error controlling coefficients
        with _no_grad():
            Qk = 1 + Qk * config.nabla
            Ck = Ck + (loss.abs() - Ck) / Qk

        # step 1: determine step size along p
        alpha = _linesearch(phi, params, grad, alpha, iters, Ck, config).detach()

        # step 2: get updated x
        with _no_grad():
            params = params + alpha * conj

        # step 3: calculate the gradient at new x
        lossnew, gradnew = objective(params)
        lossnew = lossnew.detach()
        gradnew = gradnew.detach()

        with _no_grad():

            # step 4: calculate beta parameter for the next search direction
            beta = hager_zhang_dynamic(gradnew, grad, conj, config.eta)

            # step 5: obtain the new search direction
            conjnew = - gradnew + beta * conj

            # step 6: determine the actual search direction
            if gradnew.dot(conjnew) >= 0:  # happens A LOT when using SINGLE-PRECISION floating points
                conj = - gradnew
            else:
                conj = conjnew

            # check if loss improvement is smaller than the tolerance
            if lossnew == 0.0:  # comparing to exact zero
                cond1 = (loss - lossnew).abs() <= tol
            else:
                cond1 = ((loss - lossnew) / lossnew).abs() <= tol

            # check if the infinity norm of new gradients is smaller than tolerance
            cond2 = gradnew.abs().max() <= tol

            # bump the counter
            grad, loss = gradnew, lossnew

            # write result
            disp.write(f"{iters}, {loss.cpu().item()}, {grad.abs().max().cpu().item()}\n")

            # either situation we can terminate the optimization
            if cond1 or cond2:
                break
    else:
        raise RuntimeError("Infinite loop detected.")

    return params


if __name__ == "__main__":
    from torch import float32 as _float32
    from torch import float64 as _float64
    from torch.cuda import is_available as _cuda_is_available

    @_torchjit.script
    def rosenbrock(x: _Tensor):
        """The Rosenbrock function in two dimensions with a=1, b=100.
        """
        val = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        dval = _torchzeros((2,), dtype=x.dtype, device=x.device)
        dval[0] = 2 * (x[0] - 1) + 400 * x[0] * (x[0]**2 - x[1])
        dval[1] = 200 * (x[1] - x[0]**2)
        return val, dval

    # float64
    x0 = _torchtensor([0, 0], dtype=_float64, device="cpu")
    results = nonlinear_cg(rosenbrock, x0, 1000, 1e-12, sys.stdout)
    print(f"\nfloat64: {results.tolist()}\n")

    # float32
    x0 = _torchtensor([0, 0], dtype=_float32, device="cpu")
    results = nonlinear_cg(rosenbrock, x0, 1000, 1e-6, sys.stdout)
    print(f"\nfloat32: {results.tolist()}\n")

    if not _cuda_is_available():
        sys.exit(0)

    # gpu float64
    x0 = _torchtensor([0, 0], dtype=_float64, device="cuda")
    results = nonlinear_cg(rosenbrock, x0, 1000, 1e-12, sys.stdout)
    print(f"\nfloat64: {results.tolist()}\n")

    # gpu float32
    x0 = _torchtensor([0, 0], dtype=_float32, device="cuda")
    results = nonlinear_cg(rosenbrock, x0, 1000, 1e-6, sys.stdout)
    print(f"\nfloat32: {results.tolist()}\n")
