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
from typing import List as _List
from typing import Optional as _Optional
from torch import Tensor as _Tensor
from torch import jit as _torchjit
from torch import no_grad as _no_grad
from torch import enable_grad as _torchenablegrad
from torch import cat as _torchcat
from torch import contiguous_format as _torchcontifmt
from torch.optim import Optimizer as _Optimizer
from .linesearch import linesearch as _linesearch
from .linesearch import HZLineSearchConf as _HZLineSearchConf


@_torchjit.script
def hager_zhang_dynamic(gold: _Tensor, gnew: _Tensor, pold: _Tensor, eta: float) -> float:
    """The dynamic version of the Hager-Zhang (2005) algorithm to determine beta parameter for CG solvers.
    """
    # calcuate beta
    y = gnew - gold
    pdoty = pold.dot(y)

    # pdoty=0 happens when gradients do not change (especially under SINGLE PRECISION)
    # So the next search direction should just be the negative of gradients, i.e., beta=0
    if pdoty == 0.0:  # comparing to exact zero
        return 0.0

    beta = float((y - 2. * pold * y.dot(y) / pdoty).dot(gnew) / pdoty)

    # adjust beta to avoid negative beta
    etak = - 1.0 / (float(pold.dot(pold).sqrt()) * min(eta, float(gold.dot(gold).sqrt())))
    beta = max(beta, etak)

    return beta


@_no_grad()
def nonlinear_cg(
    objective: _Callable[[float], _Tuple[float, _Tensor]],
    x0: _Tensor,
    max_iters: int = 1000,
    gtol: float = 1e-7,
    ftol: float = 1e-7,
    disp: _Optional[_TextIOWrapper] = None,
    error: bool = False,
    **kwargs
):
    """Nonlinear conjugate-gradient optimizer.
    """

    # null writer if disp is not provided
    if disp is None:
        disp = open(_devnull, "w")

    # make sure objective function is differentiable (as others will work under torch.no_grad)
    objective = _torchenablegrad()(objective)

    # initialize the states (i.e., step k = 0)
    stepk: int = 0
    xk: _Tensor = x0.clone()
    lossk, gk = objective(xk)
    dk: _Tensor = - gk
    alphak: float = 0.0
    betak: float = float("NaN")
    Qk: float = 0.0
    Ck: float = 0.0
    disp.write(f"{stepk}, {lossk}, {gk.abs().max()}\n")

    # configuration of Hager-Zhang line search algorithm
    config: _HZLineSearchConf = _HZLineSearchConf()
    config.update(kwargs)
    config.set("tol", gtol)

    # local variables
    fcond: _List[bool] = [False, False]  # loss-based condition for the most recent two steps
    gcond: bool = False  # the current step gradient-based stop condition

    for stepk in range(max_iters):

        # bump k index
        stepk += 1

        def _directional_evaluate(_alpha):
            _loss, _g = objective(xk+_alpha*dk)
            _g = float(_g.dot(dk))
            return _loss, _g

        # update relative error controlling coefficients
        Qk = 1 + Qk * config.nabla
        Ck = Ck + (abs(lossk) - Ck) / Qk

        # step 1: determine step size along p
        alphak = _linesearch(_directional_evaluate, xk, gk, alphak, stepk, Ck, config)

        # step 2: update parameters and make a copy
        xk.add_(dk, alpha=alphak)

        # step 3: calculate the gradient with new parameters (end string `kp1` means "k plus 1")
        losskp1, gkp1 = objective(xk)

        # step 4: calculate beta parameter for the next search direction
        betak = hager_zhang_dynamic(gk, gkp1, dk, config.eta)

        # step 5: update the search direction
        dk.mul_(betak).sub_(gkp1)

        # check if loss improvement is smaller than the tolerance
        if losskp1 == 0.0:  # comparing to exact zero
            fcond[-1] = abs(lossk-losskp1) <= ftol
        else:
            fcond[-1] = abs((lossk-losskp1)/losskp1) <= ftol

        # check if the infinity norm of new gradients is smaller than tolerance
        gknorm = float(gkp1.abs().max())
        gcond = gknorm <= gtol

        # update loss and gradients
        lossk, gk = losskp1, gkp1

        # write result and save update state dict
        disp.write(f"{stepk}, {lossk}, {gknorm}, {alphak}, {betak}\n")

        # either condition was satisfied we can terminate the optimization
        if all(fcond) or gcond:
            break
        fcond[0] = fcond[1]
    else:
        if error:
            raise RuntimeError("Infinite loop detected.")

    return xk


class NonlinearCG(_Optimizer):
    """A pytorch optimizer wrapper for nonlinear conjugate-gradient solver.
    """

    def __init__(self, params, max_iters=1000, gtol=1e-6, ftol=1e-6, disp=None, error=False, **kwargs):

        defaults = dict(max_iters=max_iters, gtol=gtol, ftol=ftol, disp=disp, error=error, lnskwargs=kwargs)
        super().__init__(params, defaults)
        assert len(self.param_groups) == 1, "NonlinearCG doesn't support per-parameter options (parameter groups)"

        # aliases
        self._params = self.param_groups[0]["params"]
        self._state0 = self.state[self._params[0]]

        # for convenience
        self.dtype = self._params[0].dtype
        self.device = self._params[0].device

        # to keep the record for the state
        self._state0["stepk"] = None
        self._state0["lossk"] = None
        self._state0["xk"] = None
        self._state0["gk"] = None
        self._state0["dk"] = None
        self._state0["alphak"] = None
        self._state0["betak"] = None
        self._state0["Qk"] = None
        self._state0["Ck"] = None

        # closure should also be in the state dict, but it's probably not picklable
        self._closure = None

    @_no_grad()
    def _clone_param(self):
        """Deep copy of parameters.
        """
        return [p.clone(memory_format=_torchcontifmt) for p in self._params]

    @_no_grad()
    def _gather_flat_grad(self):
        """Get a flattened and concatenated gradient vector.
        """
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return _torchcat(views, 0)

    @_no_grad()
    def _set_param(self, paramsnew):
        """Set the paremteres to the underlying parameters.
        """
        for p, pnew in zip(self._params, paramsnew):
            p.copy_(pnew)

    @_no_grad()
    def _add_to_params(self, alpha: float, update: _Tensor):
        """Add a flattened-concatenated vector with a scale coefficient to the parameters.
        """
        beg = 0
        for p in self._params:
            end = beg + p.numel()
            p.add_(update[beg:end].view_as(p), alpha=alpha)
            beg = end
        assert end == update.numel()

    @_no_grad()
    def _directional_evaluate(self, alpha: float):
        """Calculate the value and projected gradients along a given direction.

        Under fixed xk and dk, given an alpha, calculate
        1. phi(alpha) = closure() with model parameters being xk + alpha * dk
        2. dphi(alpha) = grad(closure()) dot dk with model parameters being xk + alpha * dk
        """
        self._add_to_params(alpha, self._state0["dk"])  # modify the model parameters
        loss = float(self._closure())
        grad = float(self._gather_flat_grad().dot(self._state0["dk"]))
        self._set_param(self._state0["xk"])  # recover the model parameters
        return loss, grad

    @_no_grad()
    def step(self, closure):
        """Carry out a nonlinear conjugate-gradient optimization.
        """

        # retrieve configurations
        disp = self.param_groups[0]["disp"]
        max_iters = self.param_groups[0]["max_iters"]
        gtol = self.param_groups[0]["gtol"]
        ftol = self.param_groups[0]["ftol"]
        disp = self.param_groups[0]["disp"]
        error = self.param_groups[0]["error"]
        lnskwargs = self.param_groups[0]["lnskwargs"]

        # null writer if disp is not provided
        _stdout = False
        if disp is None:
            _stdout = True
            disp = open(_devnull, "w")

        # make sure objective function is differentiable (as others will work under torch.no_grad)
        self._closure = _torchenablegrad()(closure)

        # initialize the states (i.e., step k = 0)
        stepk: int = 0
        lossk: float = float(self._closure())  # loss.backward() is assumed to happen in `closure()`
        xk: _List[_Tensor] = self._clone_param()
        gk: _Tensor = self._gather_flat_grad()
        dk: _Tensor = - gk
        alphak: float = 0.0
        betak: float = float("NaN")
        Qk: float = 0.0
        Ck: float = 0.0
        disp.write(f"{stepk}, {lossk}, {gk.abs().max()}\n")

        # save back to state dictionary
        self._state0.update(stepk=stepk, lossk=lossk, xk=xk, gk=gk, dk=dk, alphak=alphak, betak=betak, Qk=Qk, Ck=Ck)

        # configuration of Hager-Zhang line search algorithm
        config: _HZLineSearchConf = _HZLineSearchConf()
        config.update(lnskwargs)
        config.set("tol", gtol)

        # local variables
        fcond: _List[bool] = [False, False]  # loss-based condition for the most recent two steps
        gcond: bool = False  # the current step gradient-based stop condition

        for stepk in range(max_iters):

            # bump k index
            stepk += 1

            # update relative error controlling coefficients
            Qk = 1 + Qk * config.nabla
            Ck = Ck + (abs(lossk) - Ck) / Qk

            # step 1: determine step size along p
            alphak = _linesearch(self._directional_evaluate, xk, gk, alphak, stepk, Ck, config)

            # step 2: update parameters and make a copy (DON'T call _directional_evaluation until _state0 is updated!)
            self._add_to_params(alphak, dk)
            xk = self._clone_param()

            # step 3: calculate the gradient with new parameters (end string `kp1` means "k plus 1")
            losskp1 = float(self._closure())
            gkp1 = self._gather_flat_grad()

            # step 4: calculate beta parameter for the next search direction
            betak = hager_zhang_dynamic(gk, gkp1, dk, config.eta)

            # step 5: update the search direction
            dk.mul_(betak).sub_(gkp1)

            # check if loss improvement is smaller than the tolerance
            if losskp1 == 0.0:  # comparing to exact zero
                fcond[-1] = abs(lossk-losskp1) <= ftol
            else:
                fcond[-1] = abs((lossk-losskp1)/losskp1) <= ftol

            # check if the infinity norm of new gradients is smaller than tolerance
            gknorm = float(gkp1.abs().max())
            gcond = gknorm <= gtol

            # update loss and gradients
            lossk, gk = losskp1, gkp1

            # write result and save update state dict
            disp.write(f"{stepk}, {lossk}, {gknorm}, {alphak}, {betak}\n")
            self._state0.update(stepk=stepk, lossk=lossk, xk=xk, gk=gk, dk=dk, alphak=alphak, betak=betak, Qk=Qk, Ck=Ck)

            # either condition was satisfied we can terminate the optimization
            if all(fcond) or gcond:
                break
            fcond[0] = fcond[1]
        else:
            if error:
                raise RuntimeError("Infinite loop detected.")

        if _stdout:
            disp.close()
