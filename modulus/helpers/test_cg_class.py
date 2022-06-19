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
import time
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from helpers.optimizers import NonlinearCG  # noqa: E402


class FakeRosenbrockSolutionModel(torch.nn.Module):
    """Basically just a length-2 vector.
    """
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.zeros(2))

    @torch.jit.export
    def closure(self):
        """Closure function for CG solver.
        """
        self.zero_grad()
        loss = (1 - self.x[0])**2 + 100 * (self.x[1] - self.x[0]**2)**2
        loss.backward()
        return float(loss)


if __name__ == "__main__":

    # Rosenbrock, CPU, float32
    # -----------------------------------------------------------------------------------------------------------------
    x = FakeRosenbrockSolutionModel()
    optimizers = NonlinearCG(x.parameters(), max_iters=100, gtol=1e-7, ftol=1e-7, disp=sys.stdout)
    tbg = time.time()
    optimizers.step(x.closure)
    print(f"Rosenbrock, CPU, float32: x = {x.x.tolist()}, walltime: {time.time()-tbg}")

    # Rosenbrock, CPU, float64
    # -----------------------------------------------------------------------------------------------------------------
    x = FakeRosenbrockSolutionModel().to(torch.float64)
    optimizers = NonlinearCG(x.parameters(), max_iters=100, gtol=1e-14, ftol=1e-14, disp=sys.stdout)
    tbg = time.time()
    optimizers.step(x.closure)
    print(f"Rosenbrock, CPU, float64: x = {x.x.tolist()}, walltime: {time.time()-tbg}")

    # Rosenbrock, GPU, float32
    # -----------------------------------------------------------------------------------------------------------------
    if torch.cuda.is_available():
        x = FakeRosenbrockSolutionModel().to("cuda")
        optimizers = NonlinearCG(x.parameters(), max_iters=100, gtol=1e-7, ftol=1e-7, disp=sys.stdout)
        tbg = time.time()
        optimizers.step(x.closure)
        print(f"Rosenbrock, GPU, float32: x = {x.x.detach().cpu().tolist()}, walltime: {time.time()-tbg}")

    # Rosenbrock, GPU, float64
    # -----------------------------------------------------------------------------------------------------------------
    if torch.cuda.is_available():
        x = FakeRosenbrockSolutionModel().to(torch.float64).to("cuda")
        optimizers = NonlinearCG(x.parameters(), max_iters=100, gtol=1e-14, ftol=1e-14, disp=sys.stdout)
        tbg = time.time()
        optimizers.step(x.closure)
        print(f"Rosenbrock, GPU, float64: x = {x.x.detach().cpu().tolist()}, walltime: {time.time()-tbg}")

    # Random fitting problem, CPU, float32
    # -----------------------------------------------------------------------------------------------------------------
    # training data
    trainx = torch.meshgrid(torch.linspace(0., 1., 101), torch.linspace(0., 1., 101), indexing="xy")
    trainx = torch.concat([val.reshape(-1, 1) for val in trainx], dim=1)
    trainy = ((1. - trainx[:, 0])**2 + 100 * (trainx[:, 1] - trainx[:, 0]**2)**2).reshape(-1, 1)

    torch.manual_seed(0)
    # neural network model
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16, True), torch.nn.SiLU(),
        torch.nn.Linear(16, 16, True), torch.nn.SiLU(),
        torch.nn.Linear(16, 16, True), torch.nn.SiLU(),
        torch.nn.Linear(16, 1, True)
    )

    # loss and cg optimizer
    lossfn = torch.nn.MSELoss()
    optimizer = NonlinearCG(model.parameters(), max_iters=10000, gtol=1e-7, ftol=1e-7, disp=sys.stdout)

    def closure():
        optimizer.zero_grad()
        pred = model(trainx)
        loss = lossfn(trainy, pred)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = float(closure())
    print(f"Random problem, CPU, float32: loss = {loss}")

    # Random fitting problem, GPU, float32
    # -----------------------------------------------------------------------------------------------------------------
    if torch.cuda.is_available():
        # training data
        trainx = torch.linspace(0., 1., 101, dtype=torch.float64, device="cuda")
        trainx = torch.meshgrid(trainx, trainx, indexing="xy")
        trainx = torch.concat([val.reshape(-1, 1) for val in trainx], dim=1)
        trainy = ((1. - trainx[:, 0])**2 + 100 * (trainx[:, 1] - trainx[:, 0]**2)**2).reshape(-1, 1)

        torch.manual_seed(0)
        # neural network model
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 16, True), torch.nn.SiLU(),
            torch.nn.Linear(16, 16, True), torch.nn.SiLU(),
            torch.nn.Linear(16, 16, True), torch.nn.SiLU(),
            torch.nn.Linear(16, 1, True)
        ).to(torch.float64).to("cuda")

        # loss and cg optimizer
        lossfn = torch.nn.MSELoss().to(torch.float64).to("cuda")
        optimizer = NonlinearCG(model.parameters(), max_iters=10000, gtol=1e-7, ftol=1e-7, disp=sys.stdout)

        def closure():
            optimizer.zero_grad()
            pred = model(trainx)
            loss = lossfn(trainy, pred)
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = float(closure())
        print(f"Random problem, GPU, float32: loss = {loss}")
