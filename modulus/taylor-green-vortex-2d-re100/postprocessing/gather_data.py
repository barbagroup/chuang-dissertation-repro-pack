#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Gathering data from each case to a single HDF file and csv files to speed visualization up.
"""
import collections
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
from pandas import concat as pdconcat
from torch import tensor as torchtensor
from torch import full_like as torchfulllike
from torch import from_numpy as torchfromnumpy

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.utils import get_model_from_file  # pylint: disable=import-error
        from helpers.utils import process_domain  # pylint: disable=import-error
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

    # folder for inference results
    inferdir = workdir.joinpath("inferencers")

    # domain bounds
    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

    # identify the last iteration
    mxstep = max([
        fname.stem.replace("flow-net-", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    graph = {}
    _, _, graph["orig"], dtype = get_model_from_file(cfg, inferdir.joinpath(f"flow-net-{mxstep}.pth"), 2)
    _, _, graph["swa"], _ = get_model_from_file(cfg, inferdir.joinpath(f"swa-model-{mxstep}.pth"), 2)

    # get a subset in the computational graph that gives us desired quantities
    model = {key: Graph(val, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields)) for key, val in graph.items()}

    # an empty error data holder
    data = DataFrame(
        data=None,
        index=Index([], dtype=float, name="time"),
        columns=MultiIndex.from_product((["orig", "swa"], ["l1norm", "l2norm"], fields)),
    )

    # an empty snapshot contour data holder
    snapshots = collections.defaultdict(lambda: collections.defaultdict(dict))

    # generate gridlines for contour
    snapshots.update({"x": nplinspace(xbg, xed, 513), "y": nplinspace(ybg, yed, 513)})
    snapshots.update({key: (val[1:] + val[:-1]) / 2 for key, val in snapshots.items()})
    snapshots["x"], snapshots["y"] = npmeshgrid(snapshots["x"], snapshots["y"])
    shape = snapshots["x"].shape

    # torch version of gridlines; reshape to N by 1
    torchx = torchtensor(snapshots["x"].reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=False)
    torchy = torchtensor(snapshots["y"].reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=False)

    for time in cfg.eval_times:
        for key in ["orig", "swa"]:
            preds = model[key]({"x": torchx, "y": torchy, "t": torchfulllike(torchx, time)})
            preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

            for field in fields:
                ans = analytical_solution(snapshots["x"], snapshots["y"], time, 0.01, field)
                err = abs(preds[field]-ans)

                # save the prediction data and error distribution
                snapshots[time][key][field] = preds[field]
                snapshots[time][key][f"err-{field}"] = err

                # norms
                data.loc[time, (key, "l1norm", field)] = 4 * nppi**2 * err.sum() / err.size
                data.loc[time, (key, "l2norm", field)] = 2 * nppi * npsqrt((err**2).sum()/err.size)

    return data, snapshots


def get_error_vs_walltime(cfg, workdir, fields):
    """Get error v.s. walltime
    """

    # domain bounds
    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

    # gridlines
    npx = nplinspace(xbg, xed, 513, dtype=npfloat32)  # vertices
    npx = (npx[1:] + npx[:-1]) / 2  # cell centers
    npy = nplinspace(ybg, yed, 513, dtype=npfloat32)  # vertices
    npy = (npy[1:] + npy[:-1]) / 2  # cell centers
    npx, npy = [val.reshape(-1, 1) for val in npmeshgrid(npx, npy)]

    # a copy of torch version
    x, y = torchfromnumpy(npx), torchfromnumpy(npy)

    def single_process(rank, inputs, outputs):
        """A single workder in multi-processing setting.
        """
        while True:
            try:
                rawfile, swafile = inputs.get(True, 2)
            except multiprocessing.queues.Empty:
                inputs.close()
                outputs.close()
                return

            # initialize data holders
            temp = Series(
                data=None, dtype=float,
                index=MultiIndex.from_product(
                    [["raw", "swa"], ["l1norm", "l2norm"], fields, cfg.eval_times]
                ).append(Index([("timestamp", "", "", "")])),
            )

            # get the computational graph
            print(f"[Rank {rank}] processing {rawfile.name}")
            step, timestamp, graph, _ = get_model_from_file(cfg, rawfile, 2)

            # get a subset in the computational graph that gives us desired quantities
            rawmodel = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

            # get swa model if it exists, otherwise, duplicate rawmodel
            print(f"[Rank {rank}] processing {swafile.name}")
            if swafile.is_file():
                swamodel = get_model_from_file(cfg, swafile, 2)
            else:
                swamodel = rawmodel

            # convert to epoch time
            temp.loc["timestamp"] = datetime.datetime.fromisoformat(timestamp).timestamp()

            for time in cfg.eval_times:
                rawpreds = rawmodel({"x": x, "y": y, "t": torchfulllike(x, time)})
                rawpreds = {k: v.detach().cpu().numpy() for k, v in rawpreds.items()}
                swapreds = swamodel({"x": x, "y": y, "t": torchfulllike(x, time)})
                swapreds = {k: v.detach().cpu().numpy() for k, v in swapreds.items()}

                for key in fields:
                    ans = analytical_solution(npx, npy, time, 0.01, key)
                    rawerr = abs(rawpreds[key]-ans)
                    swaerr = abs(swapreds[key]-ans)
                    temp.loc[("raw", "l1norm", key, time)] = float(4 * nppi**2 * rawerr.sum() / rawerr.size)
                    temp.loc[("raw", "l2norm", key, time)] = float(2 * nppi * npsqrt((rawerr**2).sum()/rawerr.size))
                    temp.loc[("swa", "l1norm", key, time)] = float(4 * nppi**2 * swaerr.sum() / swaerr.size)
                    temp.loc[("swa", "l2norm", key, time)] = float(2 * nppi * npsqrt((swaerr**2).sum()/swaerr.size))

            outputs.put((step, temp))
            inputs.task_done()
            print(f"[Rank {rank}] done processing {rawfile.name} and {swafile.name}")

    # collect all model snapshots
    files = multiprocessing.JoinableQueue()
    for i, rawfile in enumerate(workdir.joinpath("inferencers").glob("flow-net-*.pth")):
        files.put((rawfile, rawfile.with_name(rawfile.name.replace("flow-net", "swa-model"))))

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
    data = {}
    while not results.empty():
        step, result = results.get(False)
        data[step] = result

    # combine into a big table and with iteration/step as the indices
    data = pdconcat(data, axis=1).transpose().rename_axis("iterations")

    # sort with iteration numbers
    data = data.sort_index()

    # get wall time using timestamps
    data["runtime"] = get_run_time(data["timestamp"])

    return data


def main(workdir, force=False):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("outputs").mkdir(exist_ok=True)

    # cases' names
    cases = [f"nl{nl}-nn128-npts{npts}" for nl, npts in itertools.product([1, 2, 3], [2**i for i in range(10, 17)])]

    # target fields
    fields = ["u", "v", "p"]

    # initialize a data holder for errors vs simulation time
    sim_time_err = []

    # process the accuracy of the latest trained model
    with h5open(workdir.joinpath("outputs", "snapshots.h5"), "a") as h5file:

        # read and process data case-by-case
        for job in cases:

            if f"{job}" in h5file:
                if force:
                    del h5file[f"{job}"]
                else:
                    print(f"Skipping {job}")
                    continue

            print(f"Handling {job}")

            jobdir = workdir.joinpath(job, "outputs")

            # snapshot solutions and errors wrt simulation time
            cfg = OmegaConf.load(jobdir.joinpath(".hydra", "config.yaml"))
            cfg.device = "cpu"
            cfg.eval_times = list(range(0, 101, 10))

            err, snapshots = get_case_data(cfg, jobdir, fields)
            sim_time_err.append(err)

            h5file.create_dataset(f"{job}/x", data=snapshots["x"], compression="gzip")
            h5file.create_dataset(f"{job}/y", data=snapshots["y"], compression="gzip")
            for time, mtype, field in itertools.product(cfg.eval_times, ["orig", "swa"], fields):
                # predictions
                key = f"{job}/{time}/{mtype}/{field}"
                val = snapshots[time][mtype][field]
                h5file.create_dataset(key, data=val, compression="gzip")
                # spatial errors
                key = f"{job}/{time}/{mtype}/err-{field}"
                val = snapshots[time][mtype][f"err-{field}"]
                h5file.create_dataset(key, data=val, compression="gzip")

    # concat and export errors wrt simulation time
    sim_time_err = pdconcat(sim_time_err, axis=1, keys=cases)
    sim_time_err.to_csv(workdir.joinpath("outputs", "sim-time-errors.csv"))

    # # initialize a data holder for errors vs iteration/wall-time
    # wall_time_err = []

    # # process error v.s. wall time wrt. different model parameters during training
    # for job in cases:
    #     # errors wrt iterations/wall-time
    #     cfg.eval_times = [2, 8, 32]
    #     wall_time_err.append(get_error_vs_walltime(cfg, jobdir, fields))

    # # concat and export errors wrt simulation time
    # wall_time_err = pdconcat(wall_time_err, axis=1, keys=cases)
    # wall_time_err.to_csv(workdir.joinpath("output", "wall-time-errors.csv"))


if __name__ == "__main__":
    import argparse

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("taylor-green-vortex-2d-re100").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `modulus`.")

    root = root.joinpath("taylor-green-vortex-2d-re100")

    # cmd arguments
    parser = argparse.ArgumentParser(description="Post-processing Modulus Cylinder 2D Re200")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    # calling the main function
    sys.exit(main(root, args.force))
