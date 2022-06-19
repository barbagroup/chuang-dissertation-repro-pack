#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Line search algorithms.
"""
from warnings import warn as _warn
from typing import Tuple as _Tuple
from typing import List as _List
from typing import Dict as _Dict
from typing import Union as _Union
from typing import Callable as _Callable
from torch import Tensor as _Tensor
from torch import jit as torchjit


# type shorhand for 1D objective function
ObjectiveFn1D = _Callable[[float], _Tuple[float, float]]


@torchjit.script
def standard_wolfe(
    alphak: float, phik: float, dphik: float, phi0: float,
    dphi0: float, delta: float, sigma: float
) -> bool:
    """Standard Wolfe condition to indicate if a line search should stop.

    Returns
    -------
    A `bool` indicating if the standard Wolfe condition is met.
    """

    cond1: bool = (phik - phi0) <= delta * alphak * dphi0
    cond2: bool = dphik >= sigma * dphi0
    return (cond1 and cond2)


@torchjit.script
def strong_wolfe(
    alphak: float, phik: float, dphik: float, phi0: float,
    dphi0: float, delta: float, sigma: float
) -> bool:
    """Strong Wolfe condition to indicate if a line search should stop.

    Returns
    -------
    A `bool` indicating if the strong Wolfe condition is met.
    """

    cond1: bool = (phik - phi0) <= delta * alphak * dphi0
    cond2: bool = abs(dphik) <= sigma * abs(dphi0)
    return (cond1 and cond2)


@torchjit.script
def approximate_wolfe(
    phik: float, dphik: float, phi0: float, dphi0: float,
    epsk0: float, delta: float, sigma: float
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
    cond1: bool = dphik <= (2.0 * delta - 1.0) * dphi0
    cond2: bool = dphik >= sigma * dphi0
    cond3: bool = phik <= epsk0
    return (cond1 and cond2 and cond3)


@torchjit.script
def opsite_slope_condition(phia: float, dphia: float, dphib: float, epsk0: float) -> None:
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
def secant_step(a: float, b: float, dphia: float, dphib: float) -> float:
    """Calculate the result of a single secant step.
    """
    val: float = (a * dphib - b * dphia) / (dphib - dphia)
    if val == float("inf") or val != val:
        val = (a + b) / 2.0
    return val


# @torchjit.script  # totally a waste of time; very limited
class LineSearchResults:
    """A data holder to store line searching results.
    """

    def __init__(self, phi: ObjectiveFn1D, checkfn: _Callable, epsk: float):

        # enable write permision
        self._locked: bool = False

        # one-dimensional line-searching objective function
        self.phi: ObjectiveFn1D = phi

        # function to check if the stopping critera meets
        self.checkfn: _Callable = checkfn

        # parameter for error tolerance at alpha = 0
        self.epsk: float = epsk

        # convergence status
        self.converged: bool = False
        self.diverged: bool = False
        self.message: str = ""

        # function value adn derivative at alpha = 0
        results: _Tuple[float, float] = self.phi(0.0)
        self.phi0: float = results[0]
        self.dphi0: float = results[1]

        # check if initial dphi0 < 0
        assert self.dphi0 < 0, f"{self.phi0}, {self.dphi0}"

        # phi(0) plus some small tolerance term, e.g., phi(0) + epsilon * |phi(0)|
        self.epsk0: float = self.phi0 + epsk

        # braket bounds, their funtion values, and their derivatives
        self.braket: _List[float, float] = [float("NaN"), float("NaN")]
        self.phis: _List[float, float] = [float("NaN"), float("NaN")]
        self.dphis: _List[float, float] = [float("NaN"), float("NaN")]

        # possible alpha_k, its function values, and its derivative
        self.alpha: float = float("NaN")
        self.phik: float = float("NaN")
        self.dphik: float = float("NaN")

        # tracking how many time the phi function has been called
        self.counter: int = 0

        # disable write permision
        self._locked = True

    def __setattr__(self, key: str, val: _Union[int, bool, str, float]):
        """Disable direct writing access to attributes.
        """
        if key != "_locked" and self._locked:
            raise RuntimeError("Remove lock first by setting locked = False")
        super().__setattr__(key, val)

    def __str__(self):
        s = ""
        s += f"phi0: {self.phi0}, dphi0: {self.dphi0}\n"
        s += f"alphak: {self.alpha}, phik: {self.phik}, dphik: {self.dphik}\n"
        s += f"low: {self.braket[0]}, phi_low: {self.phis[0]}, dphi_low: {self.dphis[0]}\n"
        s += f"high: {self.braket[1]}, phi_high: {self.phis[1]}, dphi_high: {self.dphis[1]}\n"
        s += f"epsk: {self.epsk}, epsk0: {self.epsk0}"
        return s

    def set_alpha(self, val: float):
        """Set a new candidate alpha and calculate the loss and derivative.
        """
        results: _Tuple[float, float] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.alpha = val
        self.phik = results[0]
        self.dphik = results[1]
        self._locked = True
        if self.checkfn(self):
            self.set_converged()
        else:
            self.unset_status()

    def set_low(self, val: float):
        """Set the lower bound of the braket and calculate the loss and derivative.
        """
        results: _Tuple[float, float] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.braket[0] = val
        self.phis[0] = results[0]
        self.dphis[0] = results[1]
        self._locked = True

    def set_high(self, val: float):
        """Set the upper bound of the braket and calculate the loss and derivative.
        """
        results: _Tuple[float, float] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.braket[1] = val
        self.phis[1] = results[0]
        self.dphis[1] = results[1]
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
        self.braket[idx] = self.alpha
        self.phis[idx] = self.phik
        self.dphis[idx] = self.dphik
        self._locked = True

    def validate(self, phi: ObjectiveFn1D):
        """Validate attributes.
        """

        temp1, temp2 = phi(0.0)
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
        self.__dict__: _Dict[str, _Union[int, bool, float]] = {}

        # parameters
        self.delta: float = 0.1
        self.sigma: float = 0.9
        self.epsilon: float = 1e-6
        self.omega: float = 1e-3
        self.nabla: float = 0.7
        self.theta: float = 0.5
        self.gamma: float = 0.5
        self.eta: float = 0.01
        self.rho: float = 5.0
        self.psi0: float = 0.01
        self.psi1: float = 0.1
        self.psi2: float = 2.0
        self.tol: float = 1e-7

        # control loops
        self.max_evals: int = 100

        # lock the writing access
        self._locked = True

    def __setattr__(self, key: str, val: _Union[bool, int, float]):
        """Not allowing changing values through attributes when locked.
        """
        if key != "_locked" and self._locked:
            raise RuntimeError("Remove lock first by setting locked = False")
        self.__dict__[key] = val

    def set(self, key: str, val: _Union[float, bool, int]):
        """Explicitly requiring users to set values using this function.
        """
        self.__dict__[key] = val

    def update(self, iterable: _Dict[str, _Union[float, bool, int]]):
        """Dictionary-alike updating.
        """
        for key, val in iterable.items():
            self.set(key, val)

    def stop_check(self, data: LineSearchResults):
        """A combination of the standard and approximation Wolfe conditions used by Hager and Zhang.
        """
        if data.alpha <= 0.0:
            return False

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
        assert 0 < self.psi0 < 1
        assert 0 < self.psi1 < 1
        assert 1 < self.psi2


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
    if data.alpha < min(data.braket) or data.alpha > max(data.braket):
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
        if abs(data.braket[1] - data.braket[0]) < config.tol:
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
        if data.braket[0] == 0.0:  # comparing exact zero
            if data.braket[1] != 0.0:
                data.set_alpha(sum(data.braket)/2.)
            else:
                data.set_alpha(0.5)
        else:
            data.set_alpha(data.braket[0])
        data.set_diverged()
        _warn(f"Exceeding function evaluation allowance. Use alhpa={data.alpha}.", RuntimeWarning)
        return data

    return data


# @torchjit.script  # totally a waste of time; very limited
def secant_braketing(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Updating an interval [a, b] using double secant algorithm from Hager and Zhang, 2005.
    """

    # get an initial alpha, phi, dphi
    data.set_alpha(secant_step(*data.braket, *data.dphis))

    # get a copy of the current data
    oldlow, oldhigh = data.braket
    olddphilow, olddphihigh = data.dphis
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
    phi: ObjectiveFn1D, step: int, aprev: float, params0: _Tensor,
    grads0: _Tensor, phi0: float, dphi0: float, config: HZLineSearchConf
) -> float:
    """Generate an initial guess of target alpha based on the current iteration number.

    Notes
    -----
    We ignore the QuadStep portion in the original algorithm and simply use $\\phi_2 * \\alpha_{k-1}$.
    """

    if step == 0:
        raise ValueError("The first iteration should be step 1. Got step 0 instead.")

    if step == 1:  # the step counter should start from 1; anyway, this `if` means the first iteration
        if isinstance(params0, list):
            norm = 0.0
            for p in params0:
                norm = max(norm, float(p.abs().max()))
        else:  # must be a torch.Tensor
            norm = float(params0.abs().max())

        if norm != 0.0:  # comparing to exact zero
            return config.psi0 * norm / float(grads0.abs().max())

        if phi0 != 0.0:  # comparing to exact zero
            return config.psi0 * abs(phi0) / float((grads0**2).sum().sqrt())

        return 1.0

    # quadratic interpolant: q(alpha) = (C0 / aprev**2) * alpha**2 + C1 * alpha + C2
    # its derivative: 2 * (C0 / aprev**2) * alpha + C1 -> critical pt: - C1 * aprev**2 / (2 * C0)
    psiaprev = aprev * config.psi1
    phia: float = phi(psiaprev)[0]
    c0 = (phia - phi0 - psiaprev * dphi0)
    if c0 > config.tol**0.5:  # strong (?) convex
        return - dphi0 * psiaprev**2 / (2. * c0)

    # if neither step 0 nor strongly convex quadratic approximation
    return config.psi2 * aprev


# @torchjit.script  # totally a waste of time; very limited
def initial_braketing(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Get an initial interval that meet opsite slope condition.
    """
    candidates = [0.0]

    while data.counter < config.max_evals:

        # if phi'(cj) >= 0, meaning it's a good option for high bound
        if data.dphik >= 0:
            data.swap("high")
            data.set_low(max(candidates))  # candidate is a native list
            return data

        # if phi'(cj) < 0 and phi(cj) > epsk0
        if data.phik > data.epsk0:
            data.set_low(0.0)
            data.swap("high")
            while data.counter < config.max_evals:

                # low, high, and alpha are almost the same, but alpha violates the stopping condition
                if abs(data.braket[1] - data.braket[0]) < config.tol:
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
                if data.braket[0] == 0.0:  # comparing exact zero
                    if data.braket[1] != 0.0:
                        data.set_alpha(sum(data.braket)/2.)
                    else:
                        data.set_alpha(0.5)
                else:
                    data.set_alpha(data.braket[0])
                data.set_diverged()
                _warn(f"Exceeding function evaluation allowance. Use alhpa={data.alpha}.", RuntimeWarning)
                return data

        # if phi'(cj) < 0 and phi(cj) <= epsk0
        candidates.append(data.alpha)
        data.set_alpha(config.rho*data.alpha)  # hence rho must > 1
    else:  # if running into this block, we didn't find a good [a, b] within max_iters iterations
        data.set_alpha(0.5)
        data.set_diverged()
        _warn(f"Exceeding function evaluation allowance. Use alhpa={data.alpha}.", RuntimeWarning)
        return data


# @torchjit.script  # totally a waste of time; very limited
def linesearch(
    phi: ObjectiveFn1D, params0: _Union[_Tensor, _List[_Tensor]], grads0: _Tensor, aprev: float, step: int, epsk: float,
    config: HZLineSearchConf,
) -> _Tensor:
    """Hager and Zhang's line searching algorithm.
    """

    data = LineSearchResults(phi, config.stop_check, epsk)

    # get an initial guess of target
    data.set_alpha(initial_alpha(phi, step, aprev, params0, grads0, data.phi0, data.dphi0, config))

    # get an initial interval
    data = initial_braketing(data, config)

    if data.diverged:  # the initial scan did not find a proper range, so we just use whatever alpha we have now
        return data.alpha

    # for debugging purpose
    data.validate(phi)

    while data.counter < config.max_evals:

        # update interval with secnat algorithm
        oldlow, oldhigh = data.braket
        data = secant_braketing(data, config)

        if data.converged or data.diverged:  # returen when either find a proper alpha or no way to find a proper alpha
            return data.alpha

        # bisection
        if data.braket[1] - data.braket[0] > config.gamma * (oldhigh - oldlow):
            data.set_alpha(sum(data.braket)/2.)
            data = single_braketing(data, config)

        if data.converged or data.diverged:  # returen when either find a proper alpha or no way to find a proper alpha
            return data.alpha

        # probably we found an exact local minimum
        if abs(data.braket[1] - data.braket[0]) < config.tol:
            return data.alpha
    else:
        if data.braket[0] == 0.0:  # comparing exact zero
            if data.braket[1] != 0.0:  # comparing exact zero
                alpha = sum(data.braket) / 2.
            else:
                alpha = 0.5
        else:
            alpha = data.braket[0]
        _warn(f"Exceeding function evaluation allowance. Use alhpa={alpha}.", RuntimeWarning)
        return alpha

    return data.alpha
