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
def cubic_minimum(braket: _Tuple[float, float], phis: _Tuple[float, float], dphis: _Tuple[float, float]):
    """Finding the mimimum beteen [low, high] using a cubic approximation.

    Note, this is only for estimating alpha when both dphis[0] and dphis[1] are both negative.
    """

    a, b = braket
    phia, phib = phis
    dphia, dphib = dphis

    amb: float = a - b
    apb: float = a + b
    phiamb: float = phia - phib
    dphiapb: float = dphia + dphib

    # denominator = a**3 - 3 * a**2 * b + 3 * a * b**2 - b**3; ignored because will be canceled out
    c1: float = amb * dphiapb - 2 * phiamb
    c2: float = - 2 * apb * amb * dphiapb + amb * (a * dphia + b * dphib) + 3 * apb * phiamb
    c3: float = 2 * apb * amb * (b * dphia + a * dphib) - amb * (a**2 * dphib + b**2 * dphia) - 6 * a * b * phiamb

    # solving for where the derivatives of the cubic are zeros
    candidate1: float = (-c2 + (-3*c1*c3 + c2**2)**0.5)/(3*c1)
    candidate2: float = -(c2 + (-3*c1*c3 + c2**2)**0.5)/(3*c1)

    # given both values in dphis are smaller than zero, so there are only limited possibilities for a cubic
    if phis[0] <= phis[1]:  # both extremes are in [a, b], the 1st is minimum, and the 2nd is the maximum
        return min(candidate1, candidate2)

    # if phis[0] > phis[1], either none or both extremes are in [a, b]
    if braket[0] < candidate1 < braket[1]:  # both are in [a, b]
        return min(candidate1, candidate2)

    # both are outside [a, b], then the higher bound is the current best alpha
    return braket[1]


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
def interval_check(
    braket: _Tuple[float, float], phis: _Tuple[float, float], dphis: _Tuple[float, float],
    epsk0: float, tol: float
) -> None:
    """Check for the opsite slope condition.
    """
    msg = f"braket: {braket}, phis: {phis}, dphis: {dphis}, epsk0: {epsk0}, tol: {tol}"

    # both conditions require the same following assertions
    assert phis[0] <= epsk0, msg
    assert dphis[0] < 0, msg
    assert dphis[1] >= 0


@torchjit.script
def secant_step(a: float, b: float, dphia: float, dphib: float) -> float:
    """Calculate the result of a single secant step.
    """
    if dphia == dphib:  # exactly equals floating numbers
        return (a + b) / 2.0

    return (a * dphib - b * dphia) / (dphib - dphia)


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

        # to record status
        self.done: bool = False
        self.message: str = ""

        # function value adn derivative at alpha = 0 and check if initial dphi0 < 0
        _results: _Tuple[float, float] = self.phi(0.0)
        self.phi0: float = _results[0]
        self.dphi0: float = _results[1]
        assert self.dphi0 < 0, f"{self.phi0}, {self.dphi0}"

        # phi(0) plus some small tolerance term
        self.epsk0: float = self.phi0 + epsk

        # braket bounds, their funtion values, and their derivatives
        self.braket: _List[float, float] = [0.0, float("inf")]
        self.phis: _List[float, float] = [_results[0], float("inf")]
        self.dphis: _List[float, float] = [_results[1], float("inf")]

        # possible alpha_k, its function values, and its derivative
        self.alphak: float = float("NaN")
        self.phik: float = float("NaN")
        self.dphik: float = float("NaN")
        self.valid_alphak: bool = False

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
        s += f"alphak: {self.alphak}, phik: {self.phik}, dphik: {self.dphik}\n"
        s += f"low: {self.braket[0]}, phi_low: {self.phis[0]}, dphi_low: {self.dphis[0]}\n"
        s += f"high: {self.braket[1]}, phi_high: {self.phis[1]}, dphi_high: {self.dphis[1]}\n"
        s += f"epsk: {self.epsk}, epsk0: {self.epsk0}"
        return s

    def reset_status(self):
        """reset self.done and self.valid_alphak.
        """
        self._locked = False
        self.done = False
        self.valid_alphak = False
        self._locked = True

    def set_alphak(self, val: float):
        """Set a new candidate alpha and calculate the loss and derivative.
        """
        results: _Tuple[float, float] = self.phi(val)
        self._locked = False
        self.counter += 1
        self.alphak = val
        self.phik, self.dphik = results
        self._locked = True

    def set_message(self, val: str):
        """Set the message.
        """
        self._locked = False
        self.message = val
        self._locked = True

    def set_done(self):
        """Set the status to converged.
        """
        self._locked = False
        self.done = True
        self._locked = True

    def check_alphak(self):
        """Set valid_alphak to True/False for Wolfe condition.
        """
        self._locked = False
        self.valid_alphak = True if self.checkfn(self) else False
        self._locked = True

    def replace_with_zero(self, bound: str):
        """Replace either lower or higher bound of the braket with current alpha.
        """
        if bound == "low":
            idx = 0
        elif bound == "high":
            idx = 1
        else:
            raise ValueError(f"Unrecognized bound: {bound}")

        self._locked = False
        self.braket[idx] = 0.0
        self.phis[idx] = self.phi0
        self.dphis[idx] = self.dphi0
        self._locked = True

    def replace_with_alphak(self, bound: str):
        """Replace either lower or higher bound of the braket with current alpha.
        """
        if bound == "low":
            idx = 0
        elif bound == "high":
            idx = 1
        else:
            raise ValueError(f"Unrecognized bound: {bound}")

        self._locked = False
        self.braket[idx] = self.alphak
        self.phis[idx] = self.phik
        self.dphis[idx] = self.dphik
        self._locked = True

    def replace_alphak_with(self, bound: str):
        """Replace alphak with either one of the bounds.
        """
        try:
            idx = {"low": 0, "high": 1}[bound]
        except KeyError as err:
            raise ValueError(f"Unrecognized bound: {bound}") from err

        self._locked = False
        self.alphak = self.braket[idx]
        self.phik = self.phis[idx]
        self.dphik = self.dphis[idx]
        self._locked = True

    def validate(self, phi: ObjectiveFn1D, tol: float):
        """Validate attributes.
        """

        temp1, temp2 = phi(0.0)
        assert self.phi0 == temp1, f"self.phi0 != phi(0): {self.phi0}, {temp1}"
        assert self.dphi0 == temp2, f"self.dphi0 != dphi(0): {self.dphi0}, {temp2}"

        temp1, temp2 = phi(self.alphak)
        assert self.phik == temp1, f"self.phik != phi(alpha): {self.phik}, {temp1}"
        assert self.dphik == temp2, f"self.dphik != dphi(alpha): {self.dphik}, {temp2}"

        temp1, temp2 = phi(self.braket[0])
        assert self.phis[0] == temp1, f"self.phis[0] != phi(low): {self.phis[0]}, {temp1}"
        assert self.dphis[0] == temp2, f"self.dphis[0] != dphi(low): {self.dphis[0]}, {temp2}"

        temp1, temp2 = phi(self.braket[1])
        assert self.phis[1] == temp1, f"self.phis[1] != phi(high): {self.phis[1]}, {temp1}"
        assert self.dphis[1] == temp2, f"self.dphis[1] != dphi(high): {self.dphis[1]}, {temp2}"

        assert self.dphi0 < 0., f"Initial slope is not negative. Got: {self.dphi0}"

        interval_check(self.braket, self.phis, self.dphis, self.epsk0, tol)


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
    max_evals : int
        Allowed max iteration numbers during line search.
    _locked : bool
        Whether the attributes can be modified through attribute interface.
    """

    def __init__(self):

        # enable the writing permision
        self._locked: bool = False

        # torch.jit needs this native attribute declared first (though we're not using torch.jit now)
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
        self.max_evals: int = 10

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
        if data.alphak <= 0.0:
            return False

        cond1 = standard_wolfe(data.alphak, data.phik, data.dphik, data.phi0, data.dphi0, self.delta, self.sigma)
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
def search_step(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Get a new and shorter intervel.

    Notes
    -----
    This function assumes the input interval, [low, high], has the following properties: phi(low) < epsk0 and
    phi'(low) < 0. It has no assumption over the high bound. That is, only the low bound needs to meet the slope
    conditions.

    Also, a alphak has to be pre-generated before calling this function, and low < alphak < high.

    Upon existing the function, the interval will have either:
    - a new and valid interval (i.e., both bounds meet the slope conditions), or
    - an interval (may be valid or invalid), and data.done = True indicating the end of line search
    """

    # alphak not in [low, high]
    if data.alphak <= data.braket[0] or data.alphak >= data.braket[1]:
        return data

    # phi'(alpha) >= 0: alphak becomes the new high bound
    if data.dphik >= 0.:
        data.replace_with_alphak("high")
        data.set_message("high bound only")
        return data

    # phi'(alphak) < 0 & phi(alphak) < epsk0: alphak becomes the new low bound
    if data.phik < data.epsk0:
        data.replace_with_alphak("low")
        data.set_message("low bound only")
        return data

    # phi'(alphak) < 0 & phi(alphak) > epsk0 > phi(low): keep finding until the high bound meets the slope conditions
    data.replace_with_alphak("high")  # temporarily use high bound to keep the value
    data.set_message("")  # no message
    while data.counter < config.max_evals:

        # if the interval is too short or the alphak (now stored at the high bound) is good enough
        if (data.braket[1] - data.braket[0]) < config.tol:
            data.set_done()
            break

        # generate a new alphak using bisection (when theta=0.5, it's classic bisection)
        data.set_alphak((1.0 - config.theta) * data.braket[0] + config.theta * data.braket[1])

        # phi'(alphak) >= 0: meets high bound's slope condition
        if data.dphik >= 0:
            data.replace_with_alphak("high")
            break

        # phi'(alphak) < 0 and phi(alphak) <= epsk0: meets low bound's slope condition
        if data.phik <= data.epsk0:
            data.replace_with_alphak("low")  # replace low bound with alphak
        # otherwise, still use high bound to hold value temporarily
        else:
            data.replace_with_alphak("high")

    # if running into this block, we have not found a valid high bound in iteration allowance
    else:
        # TODO: record all valid alphak and use the one with smallest phi(alphak)
        data.replace_alphak_with("low")
        data.set_done()
        _warn(f"Exceeding function evaluation allowance. Use alhpak={data.alphak}.", RuntimeWarning)

    return data


# @torchjit.script  # totally a waste of time; very limited
def secant_search(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Combination of a search step and the secant method to shrink interval.

    Notes
    -----
    Interval bounds are assumed to meet the oppsite slope conditions when entering this function.
    They will also meet the oppsite slope condition when existing this function.
    """

    # keep a copy
    oldlow, oldhigh = data.braket
    olddphilow, olddphihigh = data.dphis

    # use secant method on the high and low bound to get a new alphak and update interval
    data.set_alphak(secant_step(*data.braket, *data.dphis))
    data = search_step(data, config)

    # done will be True if reaching maximum eval limit or low ~ high
    if data.done: return data  # noqa: E701

    # try to shrink the bound further by changing the bound that has not been modified
    if data.message == "high bound only":
        candidate = secant_step(oldhigh, data.braket[1], olddphihigh, data.dphis[1])
    elif data.message == "low bound only":
        candidate = secant_step(oldlow, data.braket[0], olddphilow, data.dphis[0])
    else:  # both bounds have been modified in previous `search_step`
        data.check_alphak()  # mark data.valid_alphak True or False for the Wolfe conditions
        return data

    # `set_alphak` invoke the expansive phi function, so only do so when needed
    if data.braket[0] < candidate < data.braket[1]:
        data.set_alphak(candidate)
        data = search_step(data, config)
        data.check_alphak()  # mark data.valid_alphak True or False for the Wolfe conditions

    # secant returns one of the bound, meaning alphak is the exact minimum, then check the Wolfe condition
    elif candidate in data.braket:
        data.check_alphak()

    return data


# @torchjit.script  # totally a waste of time; very limited
def initialize_interval(data: LineSearchResults, config: HZLineSearchConf) -> LineSearchResults:
    """Get an initial intervel given a alphak.

    Notes
    -----
    This function behaves similar to `search_step` except that the alphak updating stragety is
    differemt. On thing to consider when initialize the interval is that high bound is inf at
    the beginning. The other thing is that we want the interval to be as big as possible, and the
    low bound to be as high as possible.
    """

    while data.counter < config.max_evals:

        # if the interval is too short
        if abs(data.braket[1] - data.braket[0]) < config.tol:
            data.set_done()
            break

        # phi'(alphak) >= 0: meets high bound's slope condition
        if data.dphik >= 0:
            data.replace_with_alphak("high")
            break

        # phi'(alphak) < 0 and phi(alphak) <= epsk0: meets low bound's slope condition, and alphak > low
        if data.phik <= data.epsk0:
            data.replace_with_alphak("low")  # replace low bound with alphak
            data.set_alphak(config.rho*data.alphak)
            continue

        # otherwise, still use high bound to hold value temporarily
        data.replace_with_alphak("high")
        data.set_alphak((1.0 - config.theta) * data.braket[0] + config.theta * data.braket[1])

    # if running into this block, we have not found a valid high bound in iteration allowance
    else:
        # TODO: record all valid alphak and use the largest one if the list is not empty
        data.replace_alphak_with("low")
        data.set_done()
        _warn(f"Exceeding function evaluation allowance. Use alhpak={data.alphak}.", RuntimeWarning)

    return data


# @torchjit.script  # totally a waste of time; very limited
def initialize_alphak(
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

    if step == 1:  # the step counter should start from 1; simply put, this `if` means the first iteration
        if isinstance(params0, list):  # the original parameters from nn.Module are not flattened
            norm = 0.0
            for p in params0:
                norm = max(norm, float(p.abs().max()))
        else:  # must be a torch.Tensor, representing the flattened parameters
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
    if c0 > config.tol**0.5:  # strongly (?) convex
        return - dphi0 * psiaprev**2 / (2. * c0)

    # if neither step 0 nor strongly convex quadratic approximation
    return config.psi2 * aprev


# @torchjit.script  # totally a waste of time; very limited
def linesearch(
    phi: ObjectiveFn1D, params0: _Union[_Tensor, _List[_Tensor]], grads0: _Tensor, aprev: float, step: int, epsk: float,
    config: HZLineSearchConf,
) -> _Tensor:
    """Hager and Zhang's line searching algorithm.
    """

    # should have an initial interval [0, inf] and alphak = NaN
    data = LineSearchResults(phi, config.stop_check, epsk)

    # get an initial alphak
    data.set_alphak(initialize_alphak(phi, step, aprev, params0, grads0, data.phi0, data.dphi0, config))

    # get an initial interval
    data = initialize_interval(data, config)

    # the search step indicates we should stop searching (not necessarrily having a valid alphak)
    if data.done: return data.alphak  # noqa: E701

    while data.counter < config.max_evals:

        # reset the values of done and valid_alphak
        data.reset_status()

        # keep a copy
        oldlow, oldhigh = data.braket

        # use secant method on the high and low bound to get a new alphak and an updated interval
        data = secant_search(data, config)

        # the search step indicates we should stop searching, or we found a good alpha
        if data.valid_alphak or data.done: break  # noqa: E701

        # for debugging purpose (TODO: turn it off for production run)
        data.validate(phi, config.tol)

        # claissic bisection if the interval did not shrink enough
        if (data.braket[1] - data.braket[0]) > config.gamma * (oldhigh - oldlow):
            data.set_alphak(sum(data.braket)/2.)
            data = search_step(data, config)
            data.check_alphak()  # mark data.valid_alphak True or False for the Wolfe conditions

            # the search step indicates we should stop searching, or we found a good alpha
            if data.valid_alphak or data.done: break  # noqa: E701

        # bounds collapsed; use the midpoint for alpha; bounds might be changed in the classic bisection
        if (data.braket[1] - data.braket[0]) < config.tol:
            return sum(data.braket) / 2.

    # did not find a porper alpha, but the interval meets all oppsite slope conditions
    else:
        alphak = sum(data.braket) / 2.
        _warn(f"Exceeding function evaluation allowance. Use alhpak={alphak}.", RuntimeWarning)
        return alphak

    return data.alphak
