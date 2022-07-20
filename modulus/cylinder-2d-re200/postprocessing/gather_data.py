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
        from helpers.utils import process_domain  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_case_data(cfg, workdir, fields=["u", "v", "p"]):
    """Get data from a single case.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

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
    npx = numpy.linspace(xbg, xed, 743)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = numpy.linspace(ybg, yed, 361)  # vertices
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

    radius = sympify(cfg.custom.radius).evalf()
    tbg, ted = process_domain(cfg.custom.t)

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
    npx = numpy.cos(theta) * float(radius)
    npy = numpy.sin(theta) * float(radius)

    # time
    nt = int(ted - tbg) + 1
    times = numpy.linspace(tbg, ted, nt)

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

        cd[i] = 2 * 2 * numpy.pi * radius * numpy.sum(fd - pd) / nr

        fl = cfg.custom.nu * (
            nx**2 * ny * preds["u_x"] + nx * ny**2 * preds["u_y"] - nx**3 * preds["v_x"] - nx**2 * ny * preds["v_y"]
        )

        pl = preds["p"] * ny

        cl[i] = - 2 * 2 * numpy.pi * radius * numpy.sum(fl - pl) / nr

    return times, cd, cl


def main(workdir, force=False):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("output").mkdir(exist_ok=True)

    # cases' names
    cases = ["nl5-nn256-npts81920", "nl5-nn256-npts81920-shedding-ic-t130"]

    eval_times = {
        "nl5-nn256-npts81920": [float(val) for val in range(0, 201, 5)],
        "nl5-nn256-npts81920-shedding-ic-t130": [float(val) for val in range(130, 201, 5)],
    }

    # target fields
    fields = ["u", "v", "p", "vorticity_z"]

    # hdf5 file
    print(workdir)
    with h5open(workdir.joinpath("output", "snapshots.h5"), "a") as h5file:

        # read and process data case-by-case
        for job in cases:

            if job in h5file:
                if not force:
                    print(f"Skipping {job}")
                    continue
                else:
                    del h5file[f"{job}"]

            print(f"Handling {job}")

            jobdir = workdir.joinpath(job, "outputs")

            cfg = OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
            cfg.device = "cpu"
            cfg.eval_times = eval_times[job]

            snapshots = get_case_data(cfg, jobdir, fields)

            h5file.create_dataset(f"{job}/x", data=snapshots["x"], compression="gzip")
            h5file.create_dataset(f"{job}/y", data=snapshots["y"], compression="gzip")
            for time, field in itertools.product(cfg.eval_times, fields):
                h5file.create_dataset(f"{job}/{time}/{field}", data=snapshots[time][field], compression="gzip")

            # times, cd, cl = get_drag_lift_coefficients(cfg, jobdir)
            # h5file.create_dataset(f"{job}/times", data=times, compression="gzip")
            # h5file.create_dataset(f"{job}/cd", data=cd, compression="gzip")
            # h5file.create_dataset(f"{job}/cl", data=cl, compression="gzip")

    return 0


if __name__ == "__main__":
    import argparse

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("cylinder-2d-re200").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `cylinder-2d-re200`.")

    root = root.joinpath("cylinder-2d-re200")
    assert root.is_dir()
    print(root)

    # cmd arguments
    parser = argparse.ArgumentParser(description="Post-processing Modulus Cylinder 2D Re200")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    # calling the main function
    sys.exit(main(root, args.force))
