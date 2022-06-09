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
import multiprocessing
import sys
import datetime
import pathlib
from h5py import File as h5open
from modulus.key import Key
from modulus.graph import Graph
from numpy import array as nparray
from numpy import concatenate as npconcatenate
from numpy import cos as npcos
from numpy import cumsum as npcumsum
from numpy import float32 as npfloat32
from numpy import full as npfull
from numpy import exp as npexp
from numpy import linspace as nplinspace
from numpy import logical_or as nplogicalor
from numpy import meshgrid as npmeshgrid
from numpy import pi as nppi
from numpy import sin as npsin
from numpy import sort as npsort
from numpy import sqrt as npsqrt
from omegaconf import OmegaConf
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Series
from sympy import sympify
from torch import tensor as torchtensor
from torch import full_like as torchfulllike
from torch import from_numpy as torchfromnumpy

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.utils import get_model_from_file  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def analytical_solution(x, y, t, nu, field, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    if field == "u":
        return V0 * npcos(x/L) * npsin(y/L) * npexp(-2.*nu*t/L**2)
    elif field == "v":
        return - V0 * npsin(x/L) * npcos(y/L) * npexp(-2.*nu*t/L**2)
    elif field == "p":
        return - rho * V0**2 * npexp(-4.*nu*t/L**2) * (npcos(2.*x/L) + npcos(2.*y/L)) / 4.
    elif field == "wz":
        return - 2. * V0 * npcos(x/L) * npcos(y/L) * npexp(-2.*nu*t/L**2) / L
    elif field == "KE":  # kinetic energy
        return nppi**2 * L**2 * V0**2 * rho * npexp(-4.*nu*t/L**2)
    elif field == "KEDR":  # kinetic energy dissipation rate
        return 4. * nppi**2 * V0**2 * nu * rho * npexp(-4.*nu*t/L**2)
    elif field == "enstrophy":  # enstrophy
        return 2. * nppi**2 * V0**2 * nu * rho * npexp(-4.*nu*t/L**2)
    else:
        raise ValueError


def get_run_time(timestamps):
    """Process timestamps, eliminate gaps betwenn Slurm job submissions, and return accumulated run time.

    The returned run times are in seconds.
    """

    timestamps = nparray(timestamps)
    diff = timestamps[1:] - timestamps[:-1]
    truncated = npsort(diff)[5:-5]

    avg = truncated.mean()
    std = truncated.std(ddof=1)
    diff[nplogicalor(diff < avg-2*std, diff > avg+2*std)] = avg
    diff = npconcatenate((npfull((1,), avg), diff))  # add back the time for the first result

    return npcumsum(diff)


def get_case_data(cfg, workdir, fields=["u", "v", "p"]):
    """Get data from a single case.
    """

    # prepare periodicity
    xbg = ybg = - cfg.custom.scale * nppi
    xed = yed = cfg.custom.scale * nppi
    period = {"x": (float(xbg), float(xed)), "y": (float(ybg), float(yed))}

    # identify the last iteration
    mxstep = max([
        fname.stem.replace("flow-net-", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    _, _, graph, dtype = get_model_from_file(cfg, workdir.joinpath("inferencers", f"flow-net-{mxstep}.pth"), period)

    # get a subset in the computational graph that gives us desired quantities
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    # gridlines
    npx = nplinspace(-cfg.custom.scale*nppi, cfg.custom.scale*nppi, 513)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = nplinspace(-cfg.custom.scale*nppi, cfg.custom.scale*nppi, 513)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = npmeshgrid(npx, npy)
    shape = npx.shape

    # torch version of gridlines; reshape to N by 1
    torchx = torchtensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=False)
    torchy = torchtensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=False)

    # error data holder
    data = DataFrame(
        data=None,
        index=Index([], dtype=float, name="time"),
        columns=MultiIndex.from_product((["l1norm", "l2norm"], fields)),
    )

    # snapshot data holder (for contour plotting)
    snapshots = {"x": npx, "y": npy}

    for time in cfg.eval_times:

        preds = model({"x": torchx, "y": torchy, "t": torchfulllike(torchx, time)})
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        for key in fields:
            ans = analytical_solution(npx, npy, time, 0.01, key)
            err = abs(preds[key]-ans)
            data.loc[time, ("l1norm", key)] = 4 * nppi**2 * err.sum() / err.size
            data.loc[time, ("l2norm", key)] = 2 * nppi * npsqrt((err**2).sum()/err.size)

        # save the prediction data
        snapshots[time] = preds

    return data, snapshots


def get_error_vs_walltime(cfg, workdir, fields):
    """Get error v.s. walltime
    """

    # prepare periodicity
    xbg = ybg = - cfg.custom.scale * nppi
    xed = yed = cfg.custom.scale * nppi
    period = {"x": (float(xbg), float(xed)), "y": (float(ybg), float(yed))}

    # gridlines
    npx = nplinspace(-cfg.custom.scale*nppi, cfg.custom.scale*nppi, 513, dtype=npfloat32)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = nplinspace(-cfg.custom.scale*nppi, cfg.custom.scale*nppi, 513, dtype=npfloat32)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = [val.reshape(-1, 1) for val in npmeshgrid(npx, npy)]

    # a copy of torch version
    x, y = torchfromnumpy(npx), torchfromnumpy(npy)

    # initialize data holders
    data = DataFrame(
        data=None,
        index=Index([], dtype=int, name="iteration"),
        columns=MultiIndex.from_product(
            [["l1norm", "l2norm"], fields, cfg.eval_times]
        ).append(Index([("timestamp", "", "")])),
    )

    def single_process(rank, inputs, outputs):
        """A single workder in multi-processing setting.
        """
        while True:
            try:
                fname, period = inputs.get(True, 2)
            except multiprocessing.queues.Empty:
                inputs.close()
                outputs.close()
                return

            print(f"[Rank {rank}] processing {fname.name}")

            # initialize data holders
            temp = Series(
                data=None, dtype=float,
                index=MultiIndex.from_product(
                    [["l1norm", "l2norm"], fields, cfg.eval_times]
                ).append(Index([("timestamp", "", "")])),
            )

            # get the computational graph
            step, timestamp, graph, _ = get_model_from_file(cfg, fname, period)

            # convert to epoch time
            temp.loc["timestamp"] = datetime.datetime.fromisoformat(timestamp).timestamp()

            # get a subset in the computational graph that gives us desired quantities
            model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

            for time in cfg.eval_times:
                preds = model({"x": x, "y": y, "t": torchfulllike(x, time)})
                preds = {k: v.detach().cpu().numpy() for k, v in preds.items()}

                for key in fields:
                    ans = analytical_solution(npx, npy, time, 0.01, key)
                    err = abs(preds[key]-ans)
                    temp.loc[("l1norm", key, time)] = float(4 * nppi**2 * err.sum() / err.size)
                    temp.loc[("l2norm", key, time)] = float(2 * nppi * npsqrt((err**2).sum()/err.size))

            outputs.put((step, temp))
            inputs.task_done()
            print(f"[Rank {rank}] done processing {fname.name}")

    # collect all model snapshots
    files = multiprocessing.JoinableQueue()
    for i, file in enumerate(workdir.joinpath("inferencers").glob("flow-net-*.pth")):
        files.put((file, period))

    # initialize a queue for outputs
    results = multiprocessing.Queue()

    # workers
    procs = []
    for rank in range(multiprocessing.cpu_count()//2):
        proc = multiprocessing.Process(target=single_process, args=(rank, files, results))
        proc.start()
        procs.append(proc)

    # block until the queue is empty
    files.join()

    # extra from result queue
    while not results.empty():
        step, result = results.get(False)
        data.loc[step] = result

    # sort with iteration numbers
    data = data.sort_index()

    # get wall time using timestamps
    data["runtime"] = get_run_time(data["timestamp"])

    return data


def main(workdir):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("output").mkdir(exist_ok=True)

    # cases' names
    cases = [f"a100-{n}" for n in [1, 2, 4, 8]]

    # target fields
    fields = ["u", "v", "p"]

    # initialize a data holder for errors
    data = DataFrame(
        data=None, dtype=float,
        index=Index([], dtype=float, name="time"),
        columns=MultiIndex.from_product((cases, ["l1norm", "l2norm"], fields)),
    )

    # hdf5 file
    with h5open(workdir.joinpath("output", "snapshots.h5"), "w") as h5file:

        # read and process data case-by-case
        for job in cases:
            print(f"Handling {job}")

            jobdir = workdir.joinpath(job, "outputs")

            cfg = OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
            cfg.device = "cpu"
            cfg.custom.scale = float(sympify(cfg.custom.scale).evalf())
            cfg.eval_times = list(range(0, 101, 2))

            data[job], snapshots = get_case_data(cfg, jobdir, fields)

            h5file.create_dataset(f"{job}/x", data=snapshots["x"], compression="gzip")
            h5file.create_dataset(f"{job}/y", data=snapshots["y"], compression="gzip")
            for time, field in itertools.product(cfg.eval_times, fields):
                h5file.create_dataset(f"{job}/{time}/{field}", data=snapshots[time][field], compression="gzip")

    data.to_csv(workdir.joinpath("output", "sim-time-errors.csv"))

    # get error versus wall time from a100_8
    jobdir = workdir.joinpath("a100_8", "outputs")
    cfg = OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
    cfg.device = "cpu"
    cfg.custom.scale = float(sympify(cfg.custom.scale).evalf())
    cfg.eval_times = [2, 8, 32]
    data = get_error_vs_walltime(cfg, workdir.joinpath("a100_8", "outputs"), fields)
    data.to_csv(workdir.joinpath("output", "wall-time-errors.csv"))


if __name__ == "__main__":

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("modulus").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `modulus`.")

    root = root.joinpath("modulus", "taylor-green-vortex-2d-re100", "nl6-nn256")

    main(root)
