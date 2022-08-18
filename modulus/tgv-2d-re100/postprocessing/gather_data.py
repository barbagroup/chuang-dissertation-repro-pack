#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
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

# find helpers and locate workdir
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        projdir = parent
        sys.path.insert(0, str(projdir))
        from helpers.utils import get_graph_from_file  # pylint: disable=import-error
        from helpers.utils import update_graph_with_file  # pylint: disable=import-error
        from helpers.utils import process_domain  # pylint: disable=import-error
        from helpers.utils import log_parser  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def analytical_solution(x, y, t, nu, field, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    if field == "u":
        return V0 * numpy.cos(x/L) * numpy.sin(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "v":
        return - V0 * numpy.sin(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "p":
        return - rho * V0**2 * numpy.exp(-4.*nu*t/L**2) * (numpy.cos(2.*x/L) + numpy.cos(2.*y/L)) / 4.
    elif field in ["wz", "vorticity_z"]:
        return - 2. * V0 * numpy.cos(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2) / L
    elif field == "KE":  # kinetic energy
        return numpy.pi**2 * L**2 * V0**2 * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "KEDR":  # kinetic energy dissipation rate
        return 4. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "enstrophy":  # enstrophy
        return 2. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    else:
        raise ValueError(f"Unknown field: {field}")


def get_casedata(casedir, mtype, rank=0):
    """Based on the type of a case, use different ways to load cfg and graph.
    """

    # files, directories, and paths
    datadir = casedir.joinpath("outputs")

    # case configuration
    cfg = OmegaConf.load(datadir.joinpath(".hydra", "config.yaml"))

    # get a dict to convert from iteration to elapsed wall time
    cfg.elapsedtimes = log_parser(casedir)["time elapsed"].to_dict()

    # get a list of flow-net model file dividable by 10
    cfg.rawfiles = [
        str(v) for v in datadir.glob("inferencers/flow-net-*.pth")
        if int(v.stem.replace("flow-net-", "")) % 10 == 0
    ]
    cfg.rawfiles.sort(key=lambda inp: int(pathlib.Path(inp).stem.replace("flow-net-", "")))

    # get a list of swa-net model files dividable by 10
    cfg.swafiles = [
        str(v) for v in datadir.glob("inferencers/swa-model-*.pth")
        if int(v.stem.replace("swa-model-", "")) % 10 == 0
    ]
    cfg.swafiles.sort(key=lambda inp: int(pathlib.Path(inp).stem.replace("swa-model-", "")))

    # identify the last iteration
    mxstep = max([pathlib.Path(fname).stem.replace("flow-net-", "") for fname in cfg.rawfiles], key=int)

    # will get the computational graph from the latest checkpoint
    if mtype == "raw":
        modelfile = datadir.joinpath("inferencers", f"flow-net-{mxstep}.pth")
    elif mtype == "swa":
        modelfile = datadir.joinpath("inferencers", f"swa-model-{mxstep}.pth")
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # extra configurations
    cfg.device = "cpu"
    cfg.eval_times = [0., 40., 80.]
    cfg.nx = 512  # number of cells in x direction
    cfg.ny = 512  # number of cells in y direction

    # whether the model and graph should have the variable `t`
    unsteady = True

    # get the computational graph from file
    print(f"[Rank {rank}] Reading model from {modelfile.name}")
    _, _, graph, _ = get_graph_from_file(cfg, modelfile, dim=2, unsteady=unsteady, device="cpu")

    return cfg, graph


def get_snapshots(cfg, graph, fields, h5file, h5kwargs, rank=0):  # pylint: disable=too-many-locals
    """Get snapshots data and write to a HDF5 group immediately to save memory.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

    # get a subset in the computational graph that gives us desired quantities
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    dtype = next(model.parameters()).dtype

    # gridlines (vertices)
    npx = numpy.linspace(xbg, xed, cfg.nx+1)
    npy = numpy.linspace(ybg, yed, cfg.ny+1)

    # gridlines (cell centers)
    npx = (npx[1:] + npx[:-1]) / 2
    npy = (npy[1:] + npy[:-1]) / 2

    # mesh
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # torch version of gridlines; reshape to N by 1
    kwargs = {"dtype": dtype, "device": cfg.device, "requires_grad": True}
    invars = {
        "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
        "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    }

    # snapshot data holder (for contour plotting)
    h5file.create_dataset("field/x", data=npx, **h5kwargs)
    h5file.create_dataset("field/y", data=npy, **h5kwargs)
    h5file.create_dataset("field/times", data=cfg.eval_times, **h5kwargs)

    for time in cfg.eval_times:
        print(f"[Rank {rank}] Predicting time = {time}")

        # make and get a subgroup using the time as its name
        grp = h5file.create_group(f"field/{time}")

        invars["t"] = torch.full_like(invars["x"], time)  # pylint: disable=no-member
        preds = model(invars)
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        for field in fields:
            ans = analytical_solution(npx, npy, time, 0.01, field)
            err = abs(preds[field]-ans)
            grp.create_dataset(f"{field}", data=preds[field], **h5kwargs)
            grp.create_dataset(f"err-{field}", data=err, **h5kwargs)

        h5file.flush()

    return h5file


def get_simtime_errs(cfg, graph, fields, h5file, h5kwargs, rank=0):  # pylint: disable=too-many-locals
    """Get err versus simulation time and write to a HDF5 group immediately to save memory.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

    # get a subset in the computational graph that gives us desired quantities
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    dtype = next(model.parameters()).dtype

    # gridlines (vertices)
    npx = numpy.linspace(xbg, xed, cfg.nx+1)
    npy = numpy.linspace(ybg, yed, cfg.ny+1)

    # gridlines (cell centers)
    npx = (npx[1:] + npx[:-1]) / 2
    npy = (npy[1:] + npy[:-1]) / 2

    # mesh
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # tensor props; don't need vorticity here, so no need for autograd
    kwargs = {"dtype": dtype, "device": cfg.device, "requires_grad": False}

    # torch version of gridlines; reshape to N by 1
    invars = {
        "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
        "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    }

    # snapshot data holder (for contour plotting)
    h5file.create_dataset("simtime/times", data=cfg.eval_times, **h5kwargs)

    l1norms = {k: [] for k in fields}
    l2norms = {k: [] for k in fields}
    for time in cfg.eval_times:
        print(f"[Rank {rank}] Predicting time = {time}")
        invars["t"] = torch.full_like(invars["x"], time)  # pylint: disable=no-member
        preds = model(invars)
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        for field in fields:
            ans = analytical_solution(npx, npy, time, 0.01, field)
            err = abs(preds[field]-ans)
            l1norms[field].append(err.sum() / err.size)
            l2norms[field].append(numpy.sqrt((err**2).sum() / err.size))

    for field in fields:
        h5file.create_dataset(f"simtime/l1norms/{field}", data=l1norms[field], **h5kwargs)
        h5file.create_dataset(f"simtime/l2norms/{field}", data=l2norms[field], **h5kwargs)
    h5file.flush()

    return h5file


def get_walltime_errs(cfg, graph, fields, mtype, h5file, h5kwargs, rank=0):  # pylint: disable=too-many-locals
    """Get err versus walltime and write to a HDF5 group immediately to save memory.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)

    # gridlines (vertices)
    npx = numpy.linspace(xbg, xed, cfg.nx+1)
    npy = numpy.linspace(ybg, yed, cfg.ny+1)

    # gridlines (cell centers)
    npx = (npx[1:] + npx[:-1]) / 2
    npy = (npy[1:] + npy[:-1]) / 2

    # mesh
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # determine dtype
    for node in graph:
        if node.name == "flow-net":
            dtype = next(node.evaluate.parameters()).dtype
            break

    # tensor props; don't need vorticity here, so no need for autograd
    kwargs = {"dtype": dtype, "device": cfg.device, "requires_grad": False}

    # torch version of gridlines; reshape to N by 1
    invars = {
        "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
        "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    }

    # get corresponding list of files
    assert mtype in ["raw", "swa"]
    tgtfiles = cfg.rawfiles if mtype == "raw" else cfg.swafiles

    # data holders
    steps = []
    elapsedtimes = []
    l1norms = {k: {time: [] for time in cfg.eval_times} for k in fields}
    l2norms = {k: {time: [] for time in cfg.eval_times} for k in fields}

    for tgt in tgtfiles:  # assumed tgtfiles are sorted from small to big
        tgt = pathlib.Path(tgt)
        step, _, graph, _ = update_graph_with_file(cfg, tgt, graph, cfg.device)

        steps.append(step)
        elapsedtimes.append(cfg.elapsedtimes[step])

        model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

        for time in cfg.eval_times:
            print(f"[Rank {rank}] Walltime errs: {tgt.stem}, time={time}")
            invars["t"] = torch.full_like(invars["x"], time)  # pylint: disable=no-member
            preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in model(invars).items()}

            for field in fields:
                ans = analytical_solution(npx, npy, time, 0.01, field)
                err = abs(preds[field]-ans)
                l1norms[field][time].append(err.sum() / err.size)
                l2norms[field][time].append(numpy.sqrt((err**2).sum() / err.size))

    # hdf5
    grp = h5file.create_group("walltime")
    grp.create_dataset("steps", data=steps, **h5kwargs)
    grp.create_dataset("elapsedtimes", data=elapsedtimes, **h5kwargs)
    for time, field in itertools.product(cfg.eval_times, fields):
        grp.create_dataset(f"l1norms/{field}/{time}", data=l1norms[field][time], **h5kwargs)
        grp.create_dataset(f"l2norms/{field}/{time}", data=l2norms[field][time], **h5kwargs)
    h5file.flush()

    return h5file


def process_single_case(
    workdir, casedir, mtype, snapshots=True, simtimeerr=True, walltimeerr=True, rank=0
):  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    """Deal with one single case.
    """

    # determine file mode
    fmode = "w" if all([snapshots, simtimeerr, walltimeerr]) else "a"

    # files, directories, and paths
    casename = casedir.name
    midpath = casedir.parent.relative_to(workdir)
    workdir.joinpath("outputs", midpath).mkdir(parents=True, exist_ok=True)
    h5filename = workdir.joinpath("outputs", midpath, f"{casename}-{mtype}.h5")

    # nothing to do
    if not any([snapshots, simtimeerr, walltimeerr]):
        print(f"[Rank {rank}] Nothing to do with {casename}. Skipping.")
        return 0

    # get configuration, computational graph, and misc. parameters
    cfg, graph = get_casedata(casedir, mtype, rank)

    # write to a HDF5 file
    h5kwargs = {"compression": "gzip"}
    with h5open(h5filename, fmode) as h5file:
        print(f"[Rank {rank}] Writing data to {h5file.filename}")

        h5file.attrs["cfg"] = OmegaConf.to_yaml(cfg)

        if snapshots:
            fields = ["u", "v", "p", "vorticity_z"]  # fields of our interest

            try:
                del h5file["fields"]
            except KeyError:
                pass

            h5file = get_snapshots(cfg, graph, fields, h5file, h5kwargs, rank)

        if simtimeerr:
            fields = ["u", "v"]  # fields of our interest
            cfg.eval_times = list(map(float, numpy.linspace(0., 80., 81)))

            try:
                del h5file["simtime"]
            except KeyError:
                pass

            h5file = get_simtime_errs(cfg, graph, fields, h5file, h5kwargs, rank)

        if walltimeerr:
            fields = ["u", "v"]  # fields of our interest
            cfg.eval_times = [0., 40., 80.]

            try:
                del h5file["walltime"]
            except KeyError:
                pass

            h5file = get_walltime_errs(cfg, graph, fields, mtype, h5file, h5kwargs, rank)

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
    topdir = projdir.joinpath("tgv-2d-re100")

    # the input queue
    inps = ctx.JoinableQueue()

    # base cases
    layers = [1, 2, 3]
    neurons = [16, 32, 64, 128, 256]
    nbss = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    mtypes = ["raw"]
    for nl, nn, nbs, mtype in itertools.product(layers, neurons, nbss, mtypes):
        inps.put((
            topdir, topdir.joinpath("base-cases", f"nl{nl}-nn{nn}-npts{nbs}"),
            mtype, True, True, True
        ))

    # # other tests
    # schers = ["exponential", "cyclic"]
    # aggs = ["sum", "annealing"]
    # ngpus = [1, 2, 4, 8]
    # for scher, agg, ngpus, mtype in itertools.product(schers, aggs, ngpus, mtypes):
    #     inps.put((
    #         topdir, topdir.joinpath(f"{scher}-{agg}-tests", f"nl3-nn128-npts8192-gpus{ngpus}"),
    #         mtype, True, True, True
    #     ))

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
