#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Line search algorithms.
"""
from typing import Tuple as _Tuple
from typing import Dict as _Dict
from typing import Union as _Union
from typing import Callable as _Callable
from torch import zeros as _torchzeros
from torch import ones as _torchones
from torch import tensor as _torchtensor
from torch import nan as _torchnan
from torch import Tensor as _Tensor
from torch import dtype as _torchdtype
from torch import device as _torchdevice
from torch import isinf as _torchisinf
from torch import isnan as _torchisnan
from torch import jit as torchjit


# type shorhand for 1D objective function
ObjectiveFn1D = _Callable[[_Tensor], _Tuple[_Tensor, _Tensor]]


@torchjit.script
def standard_wolfe(
    alphak: _Tensor, phik: _Tensor, dphik: _Tensor, phi0: _Tensor,
    dphi0: _Tensor, delta: _Tensor, sigma: _Tensor
) -> bool:
    """Standard Wolfe condition to indicate if a line search should stop.

    Returns
    -------
    A `bool` indicating if the standard Wolfe condition is met.
    """

    cond1: bool = bool((phik - phi0) <= delta * alphak * dphi0)
    cond2: bool = bool(dphik >= sigma * dphi0)
    return (cond1 and cond2)


@torchjit.script
def strong_wolfe(
    alphak: _Tensor, phik: _Tensor, dphik: _Tensor, phi0: _Tensor,
    dphi0: _Tensor, delta: _Tensor, sigma: _Tensor
) -> bool:
    """Strong Wolfe condition to indicate if a line search should stop.

    Returns
    -------
    A `bool` indicating if the strong Wolfe condition is met.
    """

    cond1: bool = bool((phik - phi0) <= delta * alphak * dphi0)
    cond2: bool = bool(dphik.abs() <= sigma * dphi0.abs())
    return (cond1 and cond2)


@torchjit.script
def approximate_wolfe(
    phik: _Tensor, dphik: _Tensor, phi0: _Tensor, dphi0: _Tensor,
    epsk0: _Tensor, delta: _Tensor, sigma: _Tensor
) -> bool:
    """Approximate Wolfe condition proposed by Hager and Zhang (2005).

    Returns
    -------
    A `bool` indicating if the strong Wolfe condition is met.

    Notes
    -----
    This conditions must meet:
    1. 0 < delta < sigma < 1.
    2. delta < min(0.5, sigma)
    """
    cond1: bool = bool(dphik <= (2.0 * delta - 1.0) * dphi0)
    cond2: bool = bool(dphik >= sigma * dphi0)
    cond3: bool = bool(phik <= epsk0)
    return (cond1 and cond2 and cond3)


@torchjit.script
def opsite_slope_condition(phia: _Tensor, dphia: _Tensor, dphib: _Tensor, epsk0: _Tensor) -> None:
    """Check for the opsite slope condition in Hager and Zhang, 2005.

    Notes
    -----
    These conditions have to be met:
        1. phi(low) <= phi(0) + eps_k
        2. phi'(low) < 0
        3. phi'(high) >= 0
    """
    assert phia <= epsk0  # check cond 1
    assert dphia < 0.  # check cond 2
    assert dphib >= 0.  # check cond 3


@torchjit.script
def secant_step(a: _Tensor, b: _Tensor, dphia: _Tensor, dphib: _Tensor) -> _Tensor:
    """Calculate the result of a single secant step.
    """
    val: _Tensor = (a * dphib - b * dphia) / (dphib - dphia)
    if _torchisinf(val) or _torchisnan(val):
        val = (a + b) / 2.0
    return val


# @torchjit.script  # totally a waste of time; very limited
class LineSearchResults:
    """A data holder to store line searching results.
    """

    def __init__(self, phi: ObjectiveFn1D, checkfn: _Callable, epsk: _Tensor):

        # enable write permision
        self._locked: bool = False

        # one-dimensional line-searching objective function
        self.phi: ObjectiveFn1D = phi

        # function to check if the stopping critera meets
        self.checkfn: _Callable = checkfn

        # parameter for error tolerance at alpha = 0
        self.epsk: _Tensor = epsk.clone()

        # convergence status
        self.converged: bool = False
        self.diverged: bool = False
        self.message: str = ""

        # function value adn derivative at alpha = 0
        results: _Tuple[_Tensor, _Tensor] = self.phi(_torchzeros((), dtype=epsk.dtype, device=epsk.device))
        self.phi0: _Tensor = results[0]
        self.dphi0: _Tensor = results[1]

        # check if initial dphi0 < 0
        assert self.dphi0 < 0, f"{self.phi0}, {self.dphi0}"

        # phi(0) plus some small tolerance term, e.g., phi(0) + epsilon * |phi(0)|
        self.epsk0: _Tensor = self.phi0 + epsk

        # braket bounds, their funtion values, and their derivatives
        self.braket: _Tensor = _torchtensor((_torchnan, _torchnan))
        self.phis: _Tensor = _torchtensor((_torchnan, _torchnan))
        self.dphis: _Tensor = _torchtensor((_torchnan, _torchnan))

        # possible alpha_k, its function values, and its derivative
        self.alpha: _Tensor = _torchtensor(_torchnan)
        self.phik: _Tensor = _torchtensor(_torchnan)
        self.dphik: _Tensor = _torchtensor(_torchnan)

        # tracking how many time the phi function has been called
        self.counter: int = 0

        # disable write permision
        self._locked = True

    def __setattr__(self, key: str, val: _Union[_Tensor, int, bool, str]):
        """Disable direct writing access to attributes.
        """
        if key != "_locked" and self._locked:
            raise RuntimeError("Remove lock first by setting locked = False")
        super().__setattr__(key, val)

    def __str__(self):
        s = ""
        s += f"phi0: {self.phi0.item()}, dphi0: {self.dphi0.item()}\n"
        s += f"alphak: {self.alpha.item()}, phik: {self.phik.item()}, dphik: {self.dphik.item()}\n"
        s += f"low: {self.braket[0].item()}, phi_low: {self.phis[0].item()}, dphi_low: {self.dphis[0].item()}\n"
        s += f"high: {self.braket[1].item()}, phi_high: {self.phis[1].item()}, dphi_high: {self.dphis[1].item()}\n"
        s += f"epsk: {self.epsk}, epsk0: {self.epsk0}"
        return s

    def to(self, val: _Union[_torchdtype, _torchdevice]):
        """Move data to a device or set the precision.
        """
        new = self.__class__(self.phi, self.checkfn, self.epsk)
        new._locked = False
        new.epsk = self.epsk.clone().to(val)
        new.phi0 = self.phi0.clone().to(val)
        new.dphi0 = self.dphi0.clone().to(val)
        new.epsk0 = self.epsk0.clone().to(val)
        new.braket = self.braket.clone().to(val)
        new.phis = self.phis.clone().to(val)
        new.dphis = self.dphis.clone().to(val)
        new.alpha = self.alpha.clone().to(val)
        new.phik = self.phik.clone().to(val)
        new.dphik = self.dphik.clone().to(val)
        new._locked = True
        return new

    def set_alpha(self, val: _Tensor):
        """Set a new candidate alpha and calculate the loss and derivative.
        """
        results: _Tuple[_Tensor, _Tensor] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.alpha.copy_(val.clone())
        self.phik.copy_(results[0])
        self.dphik.copy_(results[1])
        self._locked = True
        if self.checkfn(self): self.set_converged()  # noqa: E701

    def set_low(self, val: _Tensor):
        """Set the lower bound of the braket and calculate the loss and derivative.
        """
        results: _Tuple[_Tensor, _Tensor] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.braket[0].copy_(val.clone())
        self.phis[0].copy_(results[0])
        self.dphis[0].copy_(results[1])
        self._locked = True

    def set_high(self, val: _Tensor):
        """Set the upper bound of the braket and calculate the loss and derivative.
        """
        results: _Tuple[_Tensor, _Tensor] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.braket[1].copy_(val.clone())
        self.phis[1].copy_(results[0])
        self.dphis[1].copy_(results[1])
        self._locked = True

    def set_message(self, val: str):
        """Set the message.
        """
        self._locked = False
        self.message = val
        self._locked = True

    def set_converged(self):
        """Set the status to converged.
        """
        self._locked = False
        self.converged = True
        self.diverged = False
        self._locked = True

    def set_diverged(self):
        """Set the status to diverged.
        """
        self._locked = False
        self.converged = False
        self.diverged = True
        self._locked = True

    def unset_status(self):
        """Make the statuse neither converged nor diverged.
        """
        self._locked = False
        self.diverged = False
        self.converged = False
        self._locked = True

    def swap(self, bound: str):
        """Replace either lower or higher bound of the braket with current alpha.
        """
        if bound == "low":
            idx = 0
        elif bound == "high":
            idx = 1
        else:
            raise ValueError(f"Unrecognized bound: {bound}")

        self._locked = False
        self.braket[idx].copy_(self.alpha.clone())
        self.phis[idx].copy_(self.phik.clone())
        self.dphis[idx].copy_(self.dphik.clone())
        self._locked = True

    def validate(self, phi: ObjectiveFn1D):
        """Validate attributes.
        """

        temp1, temp2 = phi(_torchzeros((), dtype=self.alpha.dtype, device=self.alpha.device))
        assert self.phi0 == temp1, f"self.phi0 != phi(0): {self.phi0}, {temp1}"
        assert self.dphi0 == temp2, f"self.dphi0 != dphi(0): {self.dphi0}, {temp2}"

        temp1, temp2 = phi(self.alpha)
        assert self.phik == temp1, f"self.phik != phi(alpha): {self.phik}, {temp1}"
        assert self.dphik == temp2, f"self.dphik != dphi(alpha): {self.dphik}, {temp2}"

        temp1, temp2 = phi(self.braket[0])
        assert self.phis[0] == temp1, f"self.phis[0] != phi(low): {self.phis[0]}, {temp1}"
        assert self.dphis[0] == temp2, f"self.dphis[0] != dphi(low): {self.dphis[0]}, {temp2}"

        temp1, temp2 = phi(self.braket[1])
        assert self.phis[1] == temp1, f"self.phis[1] != phi(high): {self.phis[1]}, {temp1}"
        assert self.dphis[1] == temp2, f"self.dphis[1] != dphi(high): {self.dphis[1]}, {temp2}"

        assert self.dphi0 < 0., f"Initial slope is not negative. Got: {self.dphi0}"
        opsite_slope_condition(self.phis[0], self.dphis[0], self.dphis[1], self.epsk0)


# @torchjit.script  # totally a waste of time; very limited
class HZLineSearchConf:
    """A holder for used-procided parameters needed by Hager-Zhang linee search algorithm.

    For referenced equation numbers, please see p.125 in W. W. Hager and H. Zhang, “Algorithm 851: CG_DESCENT, a
    conjugate gradient method with guaranteed descent,” ACM Trans. Math. Softw., vol. 32, no. 1, pp. 113–137,
    Mar. 2006, doi: 10.1145/1132973.1132979.


    Arguments from original algorithm
    ---------------------------------
    delta : range (0, .5), used in the Wolfe conditions (22) and (23)
    alpha : range [delta, 1), used in the Wolfe conditions (22) and (23)
    epsilon : range [0, inf), used in the approximate Wolfe termination (T2)
    omega : range [0, 1], used in switching from Wolfe to approximate Wolfe conditions
    nabla : range [0, 1], decay factor for Q k in the recurrence (26)
    theta : range (0, 1), used in the update rules when the potential intervals [a, c]
            or [c, b] violate the opposite slope condition contained in (29)
    gamma : range (0, 1), determines when a bisection step is performed (L2 below)
    eta : range (0, inf), enters into the lower bound for beta in (7) through eta_k
    rho : range (1, inf), expansion factor used in the bracket rule B3.
    psi0 : range (0, 1), small factor used in starting guess I0.
    psi1 : range (0, 1), small factor used in I1.
    psi2 : range (1, inf), factor multiplying previous step of alpha in I2.

    Additional Arguments
    --------------------
    tol : torch.Tensor of a single float
        The tolerance to define two floats being equivalent.
    ls_max_iters : int
        Allowed max iteration numbers during line search.
    locked : bool
        Whether the attributes can be modified through attribute interface.
    """

    def __init__(self):

        # enable the writing permision
        self._locked: bool = False

        # torch.jit needs this native attribute declared first
        self.__dict__: _Dict[str, _Union[int, bool, _Tensor]] = {}

        # parameters
        self.delta: _Tensor = _torchtensor(0.1)
        self.sigma: _Tensor = _torchtensor(0.9)
        self.epsilon: _Tensor = _torchtensor(1e-6)
        self.omega: _Tensor = _torchtensor(1e-3)
        self.nabla: _Tensor = _torchtensor(0.7)
        self.theta: _Tensor = _torchtensor(0.5)
        self.gamma: _Tensor = _torchtensor(0.5)
        self.eta: _Tensor = _torchtensor(0.01)
        self.rho: _Tensor = _torchtensor(5.0)
        self.psi0: _Tensor = _torchtensor(0.01)
        self.psi1: _Tensor = _torchtensor(0.1)
        self.psi2: _Tensor = _torchtensor(2.0)
        self.tol: _Tensor = _torchtensor(1e-7)

        # control loops
        self.max_evals: int = 100

        # lock the writing access
        self._locked = True

    def __setattr__(self, key: str, val: _Union[_Tensor, bool, int]):
        """Not allowing changing values through attributes when locked.
        """
        if key != "_locked" and self._locked:
            raise RuntimeError("Remove lock first by setting locked = False")
        self.__dict__[key] = val

    def set(self, key: str, val: _Union[_Tensor, bool, int]):
        """Explicitly requiring users to set values using this function.
        """
        self.__dict__[key] = val

    def update(self, iterable: _Dict[str, _Union[_Tensor, bool, int]]):
        """Dictionary-alike updating.
        """
        for key, val in iterable.items():
            self.set(key, val)

    def stop_check(self, data: LineSearchResults):
        """A combination of the standard and approximation Wolfe conditions used by Hager and Zhang.
        """
        cond1 = standard_wolfe(data.alpha, data.phik, data.dphik, data.phi0, data.dphi0, self.delta, self.sigma)
        cond2 = approximate_wolfe(data.phik, data.dphik, data.phi0, data.dphi0, data.epsk0, self.delta, self.sigma)
        return (cond1 or cond2)

    def validate(self):
        """Check if all parameters are in the correct ranges.
        """
        assert 0 < self.delta < 0.5
        assert self.delta <= self.alpha < 1
        assert 0 <= self.epsilon
        assert 0 <= self.omega <= 1
        assert 0 <= self.nabla <= 1
        assert 0 < self.theta < 1
        assert 0 < self.gamma < 1
        assert 0 < self.eta
        assert 1 < self.rho
        assert 0 < self.eta0 < 1
        assert 0 < self.eta1 < 1
        assert 1 < self.eta2

    def to(self, val: _Union[_torchdtype, _torchdevice]):
        """Triger the `to` members of each held tensors.
        """
        new = self.__class__(self)
        new._locked = False
        new.delta = self.delta.clone().to(val)
        new.sigma = self.sigma.clone().to(val)
        new.epsilon = self.epsilon.clone().to(val)
        new.omega = self.omega.clone().to(val)
        new.nabla = self.nabla.clone().to(val)
        new.theta = self.theta.clone().to(val)
        new.gamma = self.gamma.clone().to(val)
        new.eta = self.eta.clone().to(val)
        new.rho = self.rho.clone().to(val)
        new.psi0 = self.psi0.clone().to(val)
        new.psi1 = self.psi1.clone().to(val)
        new.psi2 = self.psi2.clone().to(val)
        new.tol = self.tol.clone().to(val)
        new._locked = True
        return new


# @torchjit.script  # totally a waste of time; very limited
def single_braketing(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Find a new interval [a, b] that is narrower than the original one.

    Notes
    -----
    The interval bounds have to meet the following condition upon entering and exiting this function:
    1. phi(low) <= phi(0) + eps * |phi(0)|
    2. phi'(low) < 0
    3. phi'(high) >= 0
    """

    # situation 1: if target not in the current braket
    if data.alpha < data.braket.min() or data.alpha > data.braket.max():
        return data

    # situation 2: phi'(alpha) >= 0, replacing high
    if data.dphik >= 0.:
        data.swap("high")
        data.set_message("U1")
        return data

    # situation 3: phi'(alpha) < 0 and phi(alpha) <= phi0 + epsk (= epsk0); replacing low
    if data.phik <= data.epsk0:
        data.swap("low")
        data.set_message("U2")
        return data

    # situation 4: phi'(alpha) < 0 and phi(alpha) > phi0 + epsk
    data.swap("high")  # making high the only one that doesn't meet the opsite slope condition
    while data.counter < config.max_evals:

        # low, high, and alpha are almost the same, but alpha violates the stopping condition
        if (data.braket[1] - data.braket[0]).abs() < config.tol:
            assert data.converged, "low ~ high ~ alpha, but alpha violates the stopping condition"
            return data

        # interpolate to a new alpha, calculate phi & dphi, and check if it satiefying confitions
        data.set_alpha((1.0-config.theta)*data.braket[0]+config.theta*data.braket[1])

        # alpha meets high's opsite slope condition -> replacing high, and both low and high are good
        if data.dphik >= 0.0:
            data.swap("high")
            data.unset_status()  # though we found a good interval, but havn't found proper alpha
            return data

        # alpha meets low's opsite slope condition -> replacing low, but high is still not good
        if data.phik <= data.epsk0:
            data.swap("low")
        # alpha does not meet both conditions -> replacing high, keeping high the only one isn't good
        else:
            data.swap("high")
    else:  # if running into this block, we didn't find a good [a, b] within max_iters iterations
        raise RuntimeError("Exceeding maximum allowed function evaluation limit")

    return data


# @torchjit.script  # totally a waste of time; very limited
def secant_braketing(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Updating an interval [a, b] using double secant algorithm from Hager and Zhang, 2005.
    """

    # get an initial alpha, phi, dphi
    data.set_alpha(secant_step(*data.braket, *data.dphis))

    # get a copy of the current data
    oldlow, oldhigh = data.braket.clone()
    olddphilow, olddphihigh = data.dphis.clone()
    data = single_braketing(data, config)

    if data.message == "U1":  # alpha = high bound
        data.set_alpha(secant_step(oldhigh, data.braket[1], olddphihigh, data.dphis[1]))
        data = single_braketing(data, config)
    elif data.message == "U2":  # alpha = low bound
        data.set_alpha(secant_step(oldlow, data.braket[0], olddphilow, data.dphis[0]))
        data = single_braketing(data, config)

    return data


# @torchjit.script  # totally a waste of time; very limited
def initial_alpha(
    phi: ObjectiveFn1D, step: int, aprev: _Tensor, params0: _Tensor, grads0: _Tensor,
    phi0: _Tensor, dphi0: _Tensor, config: HZLineSearchConf
):
    """Generate an initial guess of target alpha based on the current iteration number.

    Notes
    -----
    We ignore the QuadStep portion in the original algorithm and simply use $\\phi_2 * \\alpha_{k-1}$.
    """

    if step == 0:
        if (params0 != 0).all():
            return config.psi0 * params0.abs().max() / grads0.abs().max()

        if (phi0 != 0).all():
            return config.psi0 * phi0.abs() / (grads0**2).sum().sqrt()

        return _torchones((), dtype=params0.dtype, device=params0.device)

    # quadratic interpolant: q(alpha) = (C0 / aprev**2) * alpha**2 + C1 * alpha + C2
    # its derivative: 2 * (C0 / aprev**2) * alpha + C1 -> critical pt: - C1 * aprev**2 / (2 * C0)
    aprev = aprev * config.psi1
    phia: _Tensor = phi(aprev)[0]
    c0 = (phia - phi0 - aprev * dphi0)
    if c0 > 0:  # convex
        return - dphi0 * aprev**2 / (2. * c0)

    # if neither step 0 nor convex quadratic approximation
    return config.psi2 * aprev


# @torchjit.script  # totally a waste of time; very limited
def initial_braketing(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Get an initial interval that meet opsite slope condition.
    """
    candidates = [_torchzeros((), dtype=data.alpha.dtype, device=data.alpha.device)]

    while data.counter < config.max_evals:

        # if phi'(cj) >= 0, meaning it's a good option for high bound
        if data.dphik >= 0:
            data.swap("high")
            data.set_low(max(candidates))  # candidate is a native list
            return data

        # if phi'(cj) < 0 and phi(cj) > epsk0
        if data.phik > data.epsk0:
            data.set_low(_torchzeros((), dtype=data.alpha.dtype, device=data.alpha.device))
            data.swap("high")
            while data.counter < config.max_evals:

                # low, high, and alpha are almost the same, but alpha violates the stopping condition
                if (data.braket[1] - data.braket[0]).abs() < config.tol:
                    assert data.converged, "low ~ high ~ alpha, but alpha violates the stopping condition"
                    return data

                data.set_alpha((1-config.theta)*data.braket[0]+config.theta*data.braket[1])

                # target meets high's opsite slope condition -> replacing high, and both low and high are good
                if data.dphik >= 0.0:
                    data.swap("high")
                    return data

                # target meets low's opsite slope condition -> replacing low, but high is still not good
                if data.phik <= data.epsk0:
                    data.swap("low")
                # target does not meet both conditions -> replacing high to make high the only one always not good
                else:
                    data.swap("high")
            else:  # if running into this block, we didn't find a good [a, b] within max_iters iterations
                raise RuntimeError("Exceeding maximum allowed function evaluation limit")

        # if phi'(cj) < 0 and phi(cj) <= epsk0
        candidates.append(data.alpha.clone())
        data.set_alpha(config.rho*data.alpha)  # hence rho must > 1
    else:  # if running into this block, we didn't find a good [a, b] within max_iters iterations
        raise RuntimeError(f"Exceeding max_evals: {data}")


# @torchjit.script  # totally a waste of time; very limited
def linesearch(
    phi: ObjectiveFn1D, params0: _Tensor, grads0: _Tensor, aprev: _Tensor, step: int,
    epsk: _Tensor, config: HZLineSearchConf,
) -> _Tensor:
    """Hager and Zhang's line searching algorithm.
    """

    data = LineSearchResults(phi, config.stop_check, epsk).to(params0.dtype).to(params0.device)

    # get an initial guess of target
    data.set_alpha(initial_alpha(phi, step, aprev, params0, grads0, data.phi0, data.dphi0, config))

    # get an initial interval
    data = initial_braketing(data, config)

    # for debugging purpose
    data.validate(phi)

    while data.counter < config.max_evals:

        # update interval with secnat algorithm
        oldlow, oldhigh = data.braket.clone()
        data = secant_braketing(data, config)

        if data.converged:
            return data.alpha

        # bisection
        if data.braket[1] - data.braket[0] > config.gamma * (oldhigh - oldlow):
            data.set_alpha(data.braket.sum()/2.)
            data = single_braketing(data, config)

            if data.converged:
                return data.alpha

        # probably we found an exact local minimum
        if (data.braket[1] - data.braket[0]).abs() < config.tol:
            break
    else:
        assert data.converged, f"Exceeding max_evals: {data}"

    return data.alpha


if __name__ == "__main__":  # doing some tests if run this module as a program
    import torch

    @torch.jit.script
    def test_loss_func(x: torch.Tensor) -> _Tuple[torch.Tensor, torch.Tensor]:
        """A test 1D loss-grad function.

        This function has two minima in the direction of positive x.
        The first is around x=0.4584 and the second around 2.6547.
        """
        val = 0.988 * x**5 - 4.96 * x**4 + 4.978 * x**3 + 5.015 * x**2 - 6.043 * x - 1
        dval = 4.94 * x**4 - 19.84 * x**3 + 14.934 * x**2 + 10.03 * x - 6.043
        return val, dval

    config = HZLineSearchConf()
    config.set("max_evals", 30)

    results = linesearch(
        test_loss_func,
        torch.tensor(0, dtype=torch.float32, device="cpu"),
        torch.tensor(-0.6, dtype=torch.float32, device="cpu"),
        torch.zeros((), dtype=torch.float32, device="cpu"),
        0,
        config.tol,
        config
    )
    print(results)

    @torch.jit.script
    def rosenbrock(x: torch.Tensor):
        """The Rosenbrock function in two dimensions with a=1, b=100.

        When x = [-3, -4] + alpha * [0.5, 1.0], there are three alpha values that give the local minumums along
        direction [0.5, 1.0], that is alpha = 4.53878634002464, alpha = 8, or alpha = 11.4612136599754.
        """
        val = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        dval = torch.zeros((2,))
        dval[0] = 2 * (x[0] - 1) + 400 * x[0] * (x[0]**2 - x[1])
        dval[1] = 200 * (x[1] - x[0]**2)
        return val, dval

    x0 = torch.tensor((-3.0, -4.0))
    searchdir = torch.tensor((0.5, 1.0))

    @torch.jit.script
    def projected(alpha: torch.Tensor, x0=x0, searchdir=searchdir):
        """loss function being projected to one dimension.
        """
        val, grads = rosenbrock(x0 + alpha * searchdir)
        dval = grads.dot(searchdir)
        return val, dval

    results = linesearch(projected, x0, rosenbrock(x0)[1], torch.zeros(()), 0, config.tol, config)
    print(results)
