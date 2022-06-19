#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Test Hager-Zhang line search and nonlinear CG solver.
"""
import sys
import pathlib
import torch
from typing import Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from helpers.linesearch import HZLineSearchConf  # noqa: E402
from helpers.linesearch import linesearch  # noqa: E402
from helpers.optimizers import nonlinear_cg  # noqa: E402


def callback(stepk, lossk, gknorm, alphak, betak, *args, **kwargs):
    """A call back to print info every step.
    """
    print(stepk, lossk, gknorm, alphak, betak)


if __name__ == "__main__":

    @torch.jit.script
    def test_loss_func(x: float) -> Tuple[float, float]:
        """A test 1D loss-grad function.
        """
        val = float(0.988 * x**5 - 4.96 * x**4 + 4.978 * x**3 + 5.015 * x**2 - 6.043 * x - 1)
        dval = float(4.94 * x**4 - 19.84 * x**3 + 14.934 * x**2 + 10.03 * x - 6.043)
        return val, dval

    # linesearch 1 (won't give exact solution because this is an inexact line search algorithm)
    # -----------------------------------------------------------------------------------------------------------------
    config = HZLineSearchConf()
    config.set("max_evals", 30)
    x0 = torch.tensor(0, dtype=torch.float32, device="cpu")
    g0 = torch.tensor(-0.6, dtype=torch.float32, device="cpu")
    a0 = 0.0
    results = linesearch(test_loss_func, x0, g0, a0, 1, config.tol, config)
    print("linesearch test 1:", results, "\n")

    def rosenbrock(x: torch.Tensor):
        """The Rosenbrock function in two dimensions with a=1, b=100.
        """
        x.grad = None  # mimics model.zero_grad() of optimizer.zero_grad()
        x.requires_grad_(True)  # allow autograd against this variable
        loss = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2  # mimic loss
        loss.backward()
        grad = x.grad.detach().clone()  # copy gradients
        x.requires_grad_(False)  # turn autograd off
        return float(loss), grad

    # linesearch 2 (won't give exact solution because this is an inexact line search algorithm)
    # -----------------------------------------------------------------------------------------------------------------
    config = HZLineSearchConf()
    config.set("max_evals", 30)
    x0 = torch.tensor((-3.0, -4.0))
    d0 = torch.tensor((0.5, 1.0))
    g0 = rosenbrock(x0)[1]
    a0 = 0.0

    def projected(alpha: float):
        """loss function being projected to one dimension.
        """
        val, grads = rosenbrock(x0 + alpha * d0)
        dval = float(grads.dot(d0))
        return val, dval

    results = linesearch(projected, x0, g0, a0, 1, config.tol, config)
    print("linesearch test 2:", results, "\n")

    # cg test: float32
    # -----------------------------------------------------------------------------------------------------------------
    x0 = torch.tensor([0, 0], dtype=torch.float32, device="cpu")
    results = nonlinear_cg(rosenbrock, x0, 1000, 1e-7, 1e-7, callback=callback)
    print(f"\nNonlinear CG test, CPU, float32: {results.tolist()}\n")

    # cg test: float64
    # -----------------------------------------------------------------------------------------------------------------
    x0 = torch.tensor([0, 0], dtype=torch.float64, device="cpu")
    results = nonlinear_cg(rosenbrock, x0, 1000, 1e-14, 1e-14, callback=callback)
    print(f"\nNonlinear CG test, CPU, float64: {results.tolist()}\n")

    # GPU tests
    # -----------------------------------------------------------------------------------------------------------------
    if torch.cuda.is_available():

        # cg test: gpu float32
        # --------------------------------------------------------------------------------------------------------------
        x0 = torch.tensor([0, 0], dtype=torch.float32, device="cuda")
        results = nonlinear_cg(rosenbrock, x0, 1000, 1e-7, 1e-7, callback=callback)
        print(f"\nNonlinear CG test, GPU, float32: {results.tolist()}\n")

        # cg test: gpu float64
        # --------------------------------------------------------------------------------------------------------------
        x0 = torch.tensor([0, 0], dtype=torch.float64, device="cuda")
        results = nonlinear_cg(rosenbrock, x0, 1000, 1e-14, 1e-14, callback=callback)
        print(f"\nNonlinear CG test, GPU, float64: {results.tolist()}\n")
