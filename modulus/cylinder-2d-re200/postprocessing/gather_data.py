#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of Cylinder 2D Re200.
"""
import sys
import itertools
import multiprocessing
import pathlib
import numpy
import torch
from h5py import File as h5open
from modulus.key import Key
from modulus.graph import Graph
from omegaconf import OmegaConf
from sympy import sympify

# find helpers and locate workdir
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        projdir = parent
        sys.path.insert(0, str(projdir))
        from helpers.utils import get_graph_from_checkpoint  # pylint: disable=import-error
        from helpers.utils import process_domain  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_casedata(workdir, casename, mtype, rank=0):
    """Based on the type of a case, use different ways to load cfg and graph.
    """

    # cases solving steady N-S equation
    steadycases = ["nl6-nn512-npts6400-steady"]

    # files, directories, and paths
    datadir = workdir.joinpath(casename, "outputs")

    # case configuration
    cfg = OmegaConf.load(datadir.joinpath(".hydra", "config.yaml"))

    # will get the computational graph from the latest checkpoint
    if mtype == "raw":
        modelfile = datadir.joinpath("flow-net.pth")
    elif mtype == "swa":
        modelfile = datadir.joinpath("swa-model.pth")
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # extra configurations
    cfg.device = "cpu"
    cfg.eval_times = [float(val) for val in range(0, 201, 10)]
    cfg.nx = 400  # number of cells in x direction
    cfg.ny = 200  # number of cells in y direction
    cfg.nr = 720  # number of cells on cylinder surface

    # whether the model and graph should have the variable `t`
    unsteady = casename not in steadycases

    # get the computational graph from file
    print(f"[Rank {rank}] Reading model from {modelfile.name}")
    graph, dtype = get_graph_from_checkpoint(cfg, modelfile, dim=2, unsteady=unsteady, mtype=mtype, device="cpu")

    # put everything to a single data object
    out = OmegaConf.create({
        "cfg": cfg,
        "dtype": str(dtype),
        "unsteady": unsteady,
    })

    return out, graph


def get_snapshots(casedata, graph, fields, rank=0):  # pylint: disable=too-many-locals
    """Get snapshots data.
    """

    xbg, xed = process_domain(casedata.cfg.custom.x)
    ybg, yed = process_domain(casedata.cfg.custom.y)

    # get a subset in the computational graph that gives us desired quantities
    if casedata.unsteady:
        model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))
    else:
        model = Graph(graph, Key.convert_list(["x", "y"]), Key.convert_list(fields))

    dtype = next(model.parameters()).dtype

    # gridlines (vertices)
    npx = numpy.linspace(xbg, xed, casedata.cfg.nx+1)
    npy = numpy.linspace(ybg, yed, casedata.cfg.ny+1)

    # gridlines (cell centers)
    npx = (npx[1:] + npx[:-1]) / 2
    npy = (npy[1:] + npy[:-1]) / 2

    # mesh
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # torch version of gridlines; reshape to N by 1
    kwargs = {"dtype": dtype, "device": casedata.cfg.device, "requires_grad": True}
    invars = {
        "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
        "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    }

    # snapshot data holder (for contour plotting)
    snapshots = {"x": npx, "y": npy}

    if casedata.unsteady:
        for time in casedata.cfg.eval_times:
            print(f"[Rank {rank}] Predicting time = {time}")
            invars["t"] = torch.full_like(invars["x"], time)  # pylint: disable=no-member
            preds = model(invars)
            snapshots[time] = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}
    else:
        print(f"[Rank {rank}] Predicting steady solution")
        preds = model(invars)
        snapshots["steady"] = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

    return snapshots


def cd_cl_kernel(norms, data, radius, nu):
    """Kernel for calculating C_D and C_L (also the viscosity and pressure effects).
    """
    # pylint: disable=too-many-locals, invalid-name

    # expand variables
    normx, normy = norms

    tau_xx = - data["p"] + 2 * nu * data["u_x"]
    tau_xy = nu * (data["u_y"] + data["v_x"])
    tau_yx = nu * (data["u_y"] + data["v_x"])
    tau_yy = - data["p"] + 2 * nu * data["v_y"]

    # viscosity force and pressure in x direction
    fd = 2 * radius * numpy.pi * numpy.mean(tau_xx * normx + tau_xy * normy)
    fdp = - 2 * radius * numpy.pi * numpy.mean(data["p"] * normx)
    fdv = fd - fdp

    cdv = 2 * fdv  # viscosity effect
    cdp = 2 * fdp  # pressure effect
    cd = 2 * fd

    # viscosity force and pressure in y direction
    fl = 2 * radius * numpy.pi * numpy.mean(tau_yx * normx + tau_yy * normy)
    flp = - 2 * radius * numpy.pi * numpy.mean(data["p"] * normy)
    flv = fl - flp

    clv = 2 * flv  # viscosity effect
    clp = 2 * flp  # pressure effect
    cl = 2 * fl

    return cdv, cdp, cd, clv, clp, cl


def get_drag_lift_coefficients(casedata, graph, rank=0):
    """Get drag and lift coefficients.
    """
    # pylint: disable=too-many-locals, invalid-name

    # required fields for calculating forces
    fields = ["u", "v", "p", "u_x", "u_y", "v_x", "v_y"]

    # get a subset in the computational graph that gives us desired quantities
    if casedata.unsteady:
        model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))
    else:
        model = Graph(graph, Key.convert_list(["x", "y"]), Key.convert_list(fields))

    dtype = next(model.parameters()).dtype

    # other parameters
    radius = float(sympify(casedata.cfg.custom.radius).evalf())

    # descretized angles
    theta = numpy.linspace(0., 2*numpy.pi, casedata.cfg.nr, False)

    # components of normal vectors (outward)
    normx = numpy.cos(theta)
    normy = numpy.sin(theta)

    # components of coordinates on cylinder surface
    npx = numpy.cos(theta) * float(radius)
    npy = numpy.sin(theta) * float(radius)

    # torch version of gridlines; reshape to N by 1
    kwargs = {"dtype": dtype, "device": casedata.cfg.device, "requires_grad": True}
    invars = {
        "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
        "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    }

    if not casedata.unsteady:

        # data holder for lift and drag coefficients
        out = {}

        print(f"[Rank {rank}] Making prediction for steady solution")
        preds = model(invars)
        preds = {k: v.detach().cpu().numpy().flatten() for k, v in preds.items()}

        out["cdv"], out["cdp"], out["cd"], out["clv"], out["clp"], out["cl"] = cd_cl_kernel(
            [normx, normy], preds, radius, casedata.cfg.custom.nu)

    else:
        # time
        _, ted = process_domain(casedata.cfg.custom.t)
        nt = int(ted) + 1
        times = numpy.linspace(0, ted, nt)

        # data holder for lift and drag coefficients
        out = {
            "times": times,
            "cdv": numpy.zeros_like(times),  # viscosity effect
            "cdp": numpy.zeros_like(times),  # pressure effect
            "cd": numpy.zeros_like(times),
            "clv": numpy.zeros_like(times),  # viscosity effect
            "clp": numpy.zeros_like(times),  # pressure effect
            "cl": numpy.zeros_like(times),
        }

        for i, time in enumerate(times):
            print(f"[Rank {rank}] Making prediction for time = {time}")
            invars["t"] = torch.full_like(invars["x"], time)  # pylint: disable=no-member
            preds = model(invars)
            preds = {k: v.detach().cpu().numpy().flatten() for k, v in preds.items()}

            out["cdv"][i], out["cdp"][i], out["cd"][i], \
                out["clv"][i], out["clp"][i], out["cl"][i] = \
                cd_cl_kernel([normx, normy], preds, radius, casedata.cfg.custom.nu)

    return out


def get_surface_pressure_coefficients(casedata, graph, rank=0):
    """Get pressure coefficients on the cylinder surface.
    """
    # pylint: disable=too-many-locals

    # required fields
    fields = ["p"]

    # get a subset in the computational graph that gives us desired quantities
    if casedata.unsteady:
        model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))
    else:
        model = Graph(graph, Key.convert_list(["x", "y"]), Key.convert_list(fields))

    dtype = next(model.parameters()).dtype

    # domain dimensions
    xbg, xed = process_domain(casedata.cfg.custom.x)
    ybg, yed = process_domain(casedata.cfg.custom.y)
    radius = float(sympify(casedata.cfg.custom.radius).evalf())

    # get a high resolution pressure field to calculate mean pressure
    kwargs = {"dtype": dtype, "device": casedata.cfg.device, "requires_grad": True}
    invars = {
        "x": torch.linspace(xbg, xed, casedata.cfg.nx*2+1, **kwargs),  # pylint: disable=no-member
        "y": torch.linspace(ybg, yed, casedata.cfg.ny*2+1, **kwargs)  # pylint: disable=no-member
    }
    invars = {k: (v[1:] + v[:-1]) / 2. for k, v in invars.items()}
    invars["x"], invars["y"] = torch.meshgrid(invars["x"], invars["y"], indexing="ij")
    invars = {k: v.reshape(-1, 1) for k, v in invars.items()}

    if casedata.unsteady:
        _, ted = process_domain(casedata.cfg.custom.t)
        print(f"[Rank {rank}] Making high-res pressure prediction for time = {ted}")
        invars["t"] = torch.full_like(invars["x"], ted)  # pylint: disable=no-member
    else:
        print(f"[Rank {rank}] Making high-res pressure prediction for steady solution")

    preds = model(invars)
    preds = {k: v.detach().cpu().numpy().flatten() for k, v in preds.items()}
    pref = preds["p"].mean()

    # descretized angles
    theta = numpy.linspace(0., 2*numpy.pi, casedata.cfg.nr, False)

    # components of coordinates on cylinder surface
    npx = numpy.cos(theta) * float(radius)
    npy = numpy.sin(theta) * float(radius)

    # torch version of gridlines; reshape to N by 1
    invars = {
        "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
        "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    }

    if casedata.unsteady:
        print(f"[Rank {rank}] Making surf. pressure prediction for time = {ted}")
        invars["t"] = torch.full_like(invars["x"], ted)  # pylint: disable=no-member
    else:
        print(f"[Rank {rank}] Making surf. pressure prediction for steady solution")

    preds = model(invars)
    preds = {k: v.detach().cpu().numpy().flatten() for k, v in preds.items()}

    return {"degrees": theta*180./numpy.pi, "cp": (preds["p"]-pref)*2}


def process_single_case(
    workdir, casename, mtype, snapshots=True, coeffs=True, rank=0
):  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    """Deal with one single case.
    """

    if not any([snapshots, coeffs]):
        print(f"[Rank {rank}] Nothing to do with {casename}. Skipping.")
        return 0

    # determine file mode
    fmode = "w" if all([snapshots, coeffs]) else "a"

    # files, directories, and paths
    workdir.joinpath("outputs").mkdir(exist_ok=True)
    h5filename = workdir.joinpath("outputs", f"{casename}-{mtype}.h5")

    # get configuration, computational graph, and misc. parameters
    casedata, graph = get_casedata(workdir, casename, mtype, rank)

    # write to a HDF5 file
    with h5open(h5filename, fmode) as h5file:
        print(f"[Rank {rank}] Writing data to {h5file.filename}")

        h5file.attrs["cfg"] = OmegaConf.to_yaml(casedata.cfg)
        h5file.attrs["dtype"] = casedata.dtype
        h5file.attrs["unsteady"] = casedata.unsteady

        if snapshots:
            fields = ["u", "v", "p", "vorticity_z"]  # fields of our interest
            snapshots = get_snapshots(casedata, graph, fields, rank)

            try:
                del h5file["fields"]
            except KeyError:
                pass
            h5file.create_dataset("fields/x", data=snapshots["x"], compression="gzip")
            h5file.create_dataset("fields/y", data=snapshots["y"], compression="gzip")

            if casedata.unsteady:
                h5file.create_dataset(
                    "fields/times", data=casedata.cfg.eval_times, compression="gzip")

                for time, field in itertools.product(casedata.cfg.eval_times, fields):
                    h5file.create_dataset(
                        f"fields/{time}/{field}", data=snapshots[time][field], compression="gzip")
            else:
                for field in fields:
                    h5file.create_dataset(
                        f"fields/steady/{field}", data=snapshots["steady"][field],
                        compression="gzip")

        if coeffs:
            coeffs = get_drag_lift_coefficients(casedata, graph, rank)

            try:
                del h5file["coeffs"]
            except KeyError:
                pass

            for key, val in coeffs.items():
                h5file.create_dataset(f"coeffs/{key}", data=val)

    return 0


def worker(inpq: multiprocessing.JoinableQueue, rank: int):
    """Thread worker."""
    while True:
        args = inpq.get(block=True, timeout=None)
        process_single_case(*args, rank=rank)
        inpq.task_done()


if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "4"  # limit threads per process
    ctx = multiprocessing.get_context('fork')  # specific spawing method

    # point workdir to the correct folder
    topdir = projdir.joinpath("cylinder-2d-re200")

    # the input queue
    inps = ctx.JoinableQueue()

    # cases
    inps.put((topdir, "nl6-nn512-npts6400-steady", "raw", False, False))
    inps.put((topdir, "nl6-nn512-npts6400-steady", "swa", False, False))
    inps.put((topdir, "nl6-nn512-npts6400-unsteady", "raw", False, False))
    inps.put((topdir, "nl6-nn512-npts6400-unsteady", "swa", False, False))
    inps.put((topdir, "nl6-nn512-npts6400-no-pinn", "raw", False, False))
    inps.put((topdir, "nl6-nn512-npts6400-no-pinn", "swa", False, False))
    inps.put((topdir, "nl8-nn512-npts6400-no-pin", "raw", False, False))
    inps.put((topdir, "nl8-nn512-npts6400-no-pin", "swa", False, False))
    inps.put((topdir, "nl8-nn512-npts50000-no-pinn", "raw", True, True))
    inps.put((topdir, "nl8-nn512-npts50000-no-pinn", "swa", True, True))

    # spawning processes
    procs = []
    for m in range(ctx.cpu_count()//4):
        proc = ctx.Process(target=worker, args=(inps, m))
        proc.start()
        procs.append(proc)

    # wait until the input queue is empty
    inps.join()

    # terminate all child processes
    for proc in procs:
        proc.terminate()
        proc.join()
        proc.close()

    # close all sub-processes if any has not been correctly closed
    multiprocessing.active_children()
