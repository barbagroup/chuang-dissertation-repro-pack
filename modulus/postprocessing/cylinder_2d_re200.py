#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of Taylor-Green vortex 2D Re100 w/ single network mode.
"""
import itertools
import sys
import pathlib
import numpy
import torch
from h5py import File as h5open
from modulus.key import Key
from modulus.graph import Graph
from omegaconf import OmegaConf
from sympy import sympify

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.utils import get_model_from_file  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_case_data(cfg, workdir, fields=["u", "v", "p"]):
    """Get data from a single case.
    """

    # identify the last iteration
    mxstep = max([
        fname.stem.replace("flow-net-", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    _, _, graph, dtype = get_model_from_file(cfg, workdir.joinpath("inferencers", f"flow-net-{mxstep}.pth"))

    # get a subset in the computational graph that gives us desired quantities
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    # gridlines
    npx = numpy.linspace(cfg.custom.xbg, cfg.custom.xed, 743)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = numpy.linspace(cfg.custom.ybg, cfg.custom.yed, 361)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # torch version of gridlines; reshape to N by 1
    torchx = torch.tensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)
    torchy = torch.tensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)

    # snapshot data holder (for contour plotting)
    snapshots = {"x": npx, "y": npy}

    for time in cfg.eval_times:

        preds = model({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        # save the prediction data
        snapshots[time] = preds

    return snapshots


def get_drag_lift_coefficients(cfg, workdir):
    """Get drag and lift coefficients.
    """

    # identify the last iteration
    mxstep = max([
        fname.stem.replace("flow-net-", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    _, _, graph, dtype = get_model_from_file(cfg, workdir.joinpath("inferencers", f"flow-net-{mxstep}.pth"))

    # required fields for calculating forces
    fields = ["u", "v", "p", "u_x", "u_y", "v_x", "v_y"]

    # get a subset in the computational graph that gives us desired quantities
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    # coordinates to infer (on the cylinder surface)
    nr = 720
    theta = numpy.linspace(0., 2*numpy.pi, 720, False)
    nx = numpy.cos(theta)  # x component of normal vector
    ny = numpy.sin(theta)  # y component of normal vector
    npx = numpy.cos(theta) * float(cfg.custom.radius)
    npy = numpy.sin(theta) * float(cfg.custom.radius)

    # time
    nt = 201
    times = numpy.linspace(0., 200., nt)

    # reshape to N by 1 vectors and create torch vectors (sharing the same memory space)
    torchx = torch.tensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)
    torchy = torch.tensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)

    # plot time frame by time frame
    cd = numpy.zeros_like(times)
    cl = numpy.zeros_like(times)
    for i, time in enumerate(times):

        preds = model({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
        preds = {k: v.detach().cpu().numpy().flatten() for k, v in preds.items()}

        fd = cfg.custom.nu * (
            nx * ny**2 * preds["u_x"] + ny**3 * preds["u_y"] - nx**2 * ny * preds["v_x"] - nx * ny**2 * preds["v_y"]
        )

        pd = preds["p"] * nx

        cd[i] = 2 * 2 * numpy.pi * cfg.custom.radius * numpy.sum(fd - pd) / nr

        fl = cfg.custom.nu * (
            nx**2 * ny * preds["u_x"] + nx * ny**2 * preds["u_y"] - nx**3 * preds["v_x"] - nx**2 * ny * preds["v_y"]
        )

        pl = preds["p"] * ny

        cl[i] = - 2 * 2 * numpy.pi * cfg.custom.radius * numpy.sum(fl - pl) / nr

    return cd, cl


def main(workdir):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("output").mkdir(exist_ok=True)

    # cases' names
    cases = [f"nn_{n}" for n in [256, 512]]

    # target fields
    fields = ["u", "v", "p", "vorticity_z"]

    # hdf5 file
    with h5open(workdir.joinpath("output", "snapshots.h5"), "w") as h5file:

        # read and process data case-by-case
        for job in cases:
            print(f"Handling {job}")

            jobdir = workdir.joinpath(job, "outputs")

            cfg = OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
            cfg.device = "cpu"
            cfg.custom.scale = float(sympify(cfg.custom.scale).evalf())
            cfg.eval_times = [200.0]

            snapshots = get_case_data(cfg, jobdir, fields)

            h5file.create_dataset(f"{job}/x", data=snapshots["x"], compression="gzip")
            h5file.create_dataset(f"{job}/y", data=snapshots["y"], compression="gzip")
            for time, field in itertools.product(cfg.eval_times, fields):
                h5file.create_dataset(f"{job}/{time}/{field}", data=snapshots[time][field], compression="gzip")

            cd, cl = get_drag_lift_coefficients(cfg, jobdir)
            h5file.create_dataset(f"{job}/cd", data=cd, compression="gzip")
            h5file.create_dataset(f"{job}/cl", data=cl, compression="gzip")


if __name__ == "__main__":

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("cases").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `cases`.")

    root = root.joinpath("cases", "cylinder-2d-re200")

    main(root)
