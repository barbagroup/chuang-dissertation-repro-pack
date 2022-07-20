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
from pandas import read_csv
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


def _get_snapshots(workdir, eval_times, fields=["u", "v", "p"], device="cpu"):
    """Get snapshot data for contour plotting
    """

    # configuration file
    cfg = OmegaConf.load(workdir.joinpath(".hydra", "config.yaml"))

    # folder for inference results
    inferdir = workdir.joinpath("inferencers")

    # domain bounds
    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

    # identify the last iteration
    mxstep = max([fname.stem.replace("flow-net-", "") for fname in inferdir.glob("flow-net-*.pth")], key=int)

    # get the computational graph
    graph = {}
    _, _, graph["orig"], dtype = get_model_from_file(cfg, inferdir.joinpath(f"flow-net-{mxstep}.pth"), 2, device)
    _, _, graph["swa"], _ = get_model_from_file(cfg, inferdir.joinpath(f"swa-model-{mxstep}.pth"), 2, device)

    # get a subset in the computational graph that gives us desired quantities
    model = {key: Graph(val, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields)) for key, val in graph.items()}

    # an empty snapshot contour data holder
    snapshots = collections.defaultdict(lambda: collections.defaultdict(dict))

    # generate gridlines for contour
    snapshots.update({"x": nplinspace(xbg, xed, 513), "y": nplinspace(ybg, yed, 513)})
    snapshots.update({key: (val[1:] + val[:-1]) / 2 for key, val in snapshots.items()})
    snapshots["x"], snapshots["y"] = npmeshgrid(snapshots["x"], snapshots["y"])
    shape = snapshots["x"].shape

    # torch version of gridlines; reshape to N by 1
    torchx = torchtensor(snapshots["x"].reshape(-1, 1), dtype=dtype, device=device, requires_grad=False)
    torchy = torchtensor(snapshots["y"].reshape(-1, 1), dtype=dtype, device=device, requires_grad=False)

    for time in eval_times:
        for key in ["orig", "swa"]:
            preds = model[key]({"x": torchx, "y": torchy, "t": torchfulllike(torchx, time)})
            preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

            for field in fields:
                ans = analytical_solution(snapshots["x"], snapshots["y"], time, 0.01, field)
                err = abs(preds[field]-ans)

                # save the prediction data and error distribution
                snapshots[time][key][field] = preds[field]
                snapshots[time][key][f"err-{field}"] = err

    return snapshots


def get_snapshots(workdir, cases, eval_times, fields, force=False, device="cpu"):
    """Get snapshot data of all cases.
    """

    def worker(_rank, _inputs, _force):
        while True:
            _dir, _eval_times, _fields, _device, _outfile = _inputs.get(block=True, timeout=None)

            if _outfile.is_file() and not _force:
                print(f"[rank {_rank:2d}] Skipping {_outfile.name}.")
                _inputs.task_done()
                continue

            print(f"[rank {_rank:2d}] Handling {_outfile.name}")
            _result = _get_snapshots(_dir, _eval_times, _fields, _device)

            with h5open(_outfile, "w") as h5file:
                h5file.create_dataset("x", data=_result["x"], compression="gzip")
                h5file.create_dataset("y", data=_result["y"], compression="gzip")
                for _time, _mtype, _field in itertools.product(_eval_times, ["orig", "swa"], _fields):
                    # predictions
                    key = f"{_time}/{_mtype}/{_field}"
                    val = _result[_time][_mtype][_field]
                    h5file.create_dataset(key, data=val, compression="gzip")
                    # spatial errors
                    key = f"{_time}/{_mtype}/err-{_field}"
                    val = _result[_time][_mtype][f"err-{_field}"]
                    h5file.create_dataset(key, data=val, compression="gzip")

            print(f"[rank {_rank:2d}] Saved to {_outfile.name}")
            _inputs.task_done()

    # an empty input queues
    inputs = multiprocessing.JoinableQueue()

    # threads
    procs = []
    for rank in range(multiprocessing.cpu_count()//2):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=worker, args=(rank, inputs, force))
        proc.start()
        procs.append(proc)

    # fill things to the input queues
    for job in cases:
        target = workdir.joinpath(job, "outputs")
        outfile = workdir.joinpath("outputs", f"{job}.snapshot.h5")
        inputs.put((target, eval_times, fields, device, outfile))

    # wait until the input queue is empty
    inputs.join()

    # terminate all ranks
    for rank, proc in enumerate(procs):
        print(f"Closing rank {rank}")
        proc.terminate()


def _get_sim_time_errs(workdir, eval_times, fields=["u", "v", "p"], device="cpu"):
    """Get errors vs simulation time.
    """

    # configuration file
    cfg = OmegaConf.load(workdir.joinpath(".hydra", "config.yaml"))

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
    _, _, graph["orig"], dtype = get_model_from_file(cfg, inferdir.joinpath(f"flow-net-{mxstep}.pth"), 2, device)
    _, _, graph["swa"], _ = get_model_from_file(cfg, inferdir.joinpath(f"swa-model-{mxstep}.pth"), 2, device)

    # get a subset in the computational graph that gives us desired quantities
    model = {key: Graph(val, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields)) for key, val in graph.items()}

    # an empty error data holder
    data = DataFrame(
        data=None,
        index=Index([], dtype=float, name="time"),
        columns=MultiIndex.from_product((["orig", "swa"], ["l1norm", "l2norm"], fields)),
    )

    # generate gridlines for contour
    x = nplinspace(xbg, xed, 513)
    y = nplinspace(ybg, yed, 513)
    x = (x[1:] + x[:-1]) / 2
    y = (y[1:] + y[:-1]) / 2
    x, y = npmeshgrid(x, y)
    shape = x.shape

    # torch version of gridlines; reshape to N by 1
    torchx = torchtensor(x.reshape(-1, 1), dtype=dtype, device=device, requires_grad=False)
    torchy = torchtensor(y.reshape(-1, 1), dtype=dtype, device=device, requires_grad=False)

    for time in eval_times:
        for key in ["orig", "swa"]:
            preds = model[key]({"x": torchx, "y": torchy, "t": torchfulllike(torchx, time)})
            preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

            for field in fields:
                ans = analytical_solution(x, y, time, 0.01, field)
                err = abs(preds[field]-ans)
                data.loc[time, (key, "l1norm", field)] = 4 * nppi**2 * err.sum() / err.size
                data.loc[time, (key, "l2norm", field)] = 2 * nppi * npsqrt((err**2).sum()/err.size)

    return data


def get_sim_time_errs(workdir, cases, eval_times, fields, force=False, device="cpu"):
    """Get errors vs simulation time.
    """

    def worker(_rank, _inputs, _force):
        while True:
            _dir, _eval_times, _fields, _device, _outfile = _inputs.get(block=True, timeout=None)

            if _outfile.is_file() and not _force:
                print(f"[rank {_rank:2d}] Skipping {_outfile.name}.")
                _inputs.task_done()
                continue

            print(f"[rank {_rank:2d}] Handling {_outfile.name}")
            _result = _get_sim_time_errs(_dir, _eval_times, _fields, _device)
            _result.to_csv(_outfile)
            print(f"[rank {_rank:2d}] Saved to {_outfile.name}")
            _inputs.task_done()

    # an empty input queues
    inputs = multiprocessing.JoinableQueue()

    # threads
    procs = []
    for rank in range(multiprocessing.cpu_count()//2):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=worker, args=(rank, inputs, force))
        proc.start()
        procs.append(proc)

    # fill things to the input queues
    for job in cases:
        target = workdir.joinpath(job, "outputs")
        outfile = workdir.joinpath("outputs", f"{job}.simtime.csv")
        inputs.put((target, eval_times, fields, device, outfile))

    # wait until the input queue is empty
    inputs.join()

    # terminate all ranks
    for rank, proc in enumerate(procs):
        print(f"Closing rank {rank}")
        proc.terminate()


def _get_wall_time_errs(workdir, eval_times, fields, device="cpu"):
    """Get error v.s. walltime
    """

    # configuration file
    cfg = OmegaConf.load(workdir.joinpath(".hydra", "config.yaml"))

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
                    [["raw", "swa"], ["l1norm", "l2norm"], fields, eval_times]
                ).append(Index([("timestamp", "", "", "")])),
            )

            # get the computational graph
            print(f"[Rank {rank}] processing {rawfile.name}")
            step, timestamp, graph, _ = get_model_from_file(cfg, rawfile, 2, device)

            # get a subset in the computational graph that gives us desired quantities
            rawmodel = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

            # get swa model if it exists, otherwise, duplicate rawmodel
            if swafile.is_file():
                print(f"[Rank {rank}] processing {swafile.name}")
                _, _, swagraph, _ = get_model_from_file(cfg, swafile, 2, device)
                swamodel = Graph(swagraph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))
            else:
                print(f"[Rank {rank}] using {rawfile.name} as {swafile.name}")
                swamodel = rawmodel

            # convert to epoch time
            temp.loc["timestamp"] = datetime.datetime.fromisoformat(timestamp).timestamp()

            for time in eval_times:
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

    return data


def get_wall_time_errs(workdir, cases, eval_times, fields, force=False, device="cpu"):
    """Get error v.s. walltime
    """

    for job in cases:
        outfile = workdir.joinpath("outputs", f"{job}.walltime.csv")
        if outfile.is_file() and not force:
            print(f"Skipping {outfile.name}")
            continue
        data = _get_wall_time_errs(workdir.joinpath(job, "outputs"), eval_times, fields, device)
        data.to_csv(outfile)


def main(workdir, force=False):
    """Main function.
    """

    # save all post-processed data here
    workdir.joinpath("outputs").mkdir(exist_ok=True)

    # cases' names
    cases = [
        f"nl{nl}-nn{nn}-npts{npts}"
        for nl, nn, npts in itertools.product(range(1, 4), [2**i for i in range(4, 9)], [2**i for i in range(10, 17)])
    ]

    # target fields
    fields = ["u", "v", "p"]

    # gather snapshot data
    get_snapshots(workdir, cases, list(range(0, 101, 20)), fields, force, "cpu")

    # gather errors wrt simulation time
    get_sim_time_errs(workdir, cases, list(range(0, 101, 10)), fields, force, "cpu")

    # gather errors wrt walltime
    get_wall_time_errs(workdir, cases, list(range(0, 101, 20)), fields, force, "cpu")

    # combine all walltime errors
    data = []
    for job in cases:
        data.append(read_csv(workdir.joinpath("outputs", f"{job}.walltime.csv"), index_col=0, header=[0, 1, 2, 3]))
        print(job, len(data[-1].index))
    data = pdconcat(data, axis=1, keys=cases)
    print(len(data.index))
    data.to_csv(workdir.joinpath("outputs", "walltime-errors.csv"))


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
