#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
"""
# pylint: disable=invalid-name, too-many-arguments, too-many-return-statements, too-many-arguments
# pylint: disable=too-many-locals
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
    if field == "v":
        return - V0 * numpy.sin(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2)
    if field == "p":
        return - rho * V0**2 * numpy.exp(-4.*nu*t/L**2) * \
            (numpy.cos(2.*x/L) + numpy.cos(2.*y/L)) / 4.
    if field in ["wz", "vorticity_z"]:
        return - 2. * V0 * numpy.cos(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2) / L
    if field == "KE":  # kinetic energy
        return numpy.pi**2 * L**2 * V0**2 * rho * numpy.exp(-4.*nu*t/L**2)
    if field == "KEDR":  # kinetic energy dissipation rate
        return 4. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    if field == "enstrophy":  # enstrophy
        return 2. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)

    raise ValueError(f"Unknown field: {field}")


def get_casedata(casedir, mtype, rank=0):
    """Based on the type of a case, use different ways to load cfg and graph.
    """

    # files, directories, and paths
    datadir = casedir.joinpath("outputs")

    # case configuration
    cfg = OmegaConf.load(datadir.parent.joinpath("config.yaml"))

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
    mxstep = max([
        pathlib.Path(fname).stem.replace("flow-net-", "")
        for fname in cfg.rawfiles], key=int
    )

    # will get the computational graph from the latest checkpoint
    if mtype == "raw":
        modelfile = datadir.joinpath("inferencers", f"flow-net-{mxstep}.pth")
    elif mtype == "swa":
        modelfile = datadir.joinpath("inferencers", f"swa-model-{mxstep}.pth")
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # extra configurations
    cfg.device = "cpu"
    cfg.eval_times = [0., 40.]
    cfg.nx = 512  # number of cells in x direction
    cfg.ny = 512  # number of cells in y direction
    cfg.nt = 100  # number of cells in y direction

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

    # snapshot data holder (for contour plotting)
    h5file.create_dataset("field/x", data=npx, **h5kwargs)
    h5file.create_dataset("field/y", data=npy, **h5kwargs)
    h5file.create_dataset("field/times", data=cfg.eval_times, **h5kwargs)

    for time in cfg.eval_times:
        print(f"[Rank {rank}] Predicting time = {time}")

        # make and get a subgroup using the time as its name
        grp = h5file.create_group(f"field/{time}")

        invars = {
            "x": torch.tensor(npx.reshape(-1, 1), **kwargs),  # pylint: disable=no-member
            "y": torch.tensor(npy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
        }
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


def get_spatial_temporal_errs(cfg, graph, fields, h5file, h5kwargs, rank=0):
    """Get err versus simulation time and write to a HDF5 group immediately to save memory.
    """
    # pylint: disable=unused-argument

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    tbg, ted = process_domain(cfg.custom.t)
    ncells =  cfg.nx * cfg.ny * cfg.nt # number of spatial-temporal cells

    # get a subset in the computational graph that gives us desired quantities
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))
    dtype = next(model.parameters()).dtype

    # spatial-temporal gridlines (vertices)
    npx = numpy.linspace(xbg, xed, cfg.nx+1)
    npy = numpy.linspace(ybg, yed, cfg.ny+1)
    npt = numpy.linspace(tbg, ted, cfg.nt+1 )

    # spatial-temporal gridlines (cell centers)
    npx = (npx[1:] + npx[:-1]) / 2
    npy = (npy[1:] + npy[:-1]) / 2
    npt = (npt[1:] + npt[:-1]) / 2

    # spatial-temporal mesh
    npx, npy, npt = numpy.meshgrid(npx, npy, npt)
    npx, npy, npt = npx.reshape(-1, 1), npy.reshape(-1, 1), npt.reshape(-1, 1)

    # batching
    bs = cfg.nx * cfg.ny * cfg.nt // 10
    nb = npx.size // 65536 + int(bool(npx.size % 65536))

    # tensor props; don't need vorticity here, so no need for autograd
    kwargs = {"dtype": dtype, "device": cfg.device, "requires_grad": False}

    print(f"[Rank {rank}] Predicting spatial-temporal errors")
    l1norms = {k: 0.0 for k in fields}
    l2norms = {k: 0.0 for k in fields}
    for i in range(nb):
        bg, ed = i * bs, (i + 1) * bs  # ed will > the actual size, but numpy is smart enough
        invars = {
            "x": torch.tensor(npx[bg:ed], **kwargs),  # pylint: disable=no-member
            "y": torch.tensor(npy[bg:ed], **kwargs),  # pylint: disable=no-member
            "t": torch.tensor(npt[bg:ed], **kwargs),  # pylint: disable=no-member
        }
        preds = model(invars)
        preds = {k: v.detach().cpu().numpy() for k, v in preds.items()}

        for field in fields:
            ans = analytical_solution(npx[bg:ed], npy[bg:ed], npt[bg:ed], 0.01, field)
            err = abs(preds[field]-ans)
            l1norms[field] += err.sum()
            l2norms[field] += (err**2).sum()

    for field in fields:
        l1norm = l1norms[field] / ncells
        l2norm = numpy.sqrt(l2norms[field] / ncells)
        h5file.create_dataset(f"sterrs/{field}/l1norm", data=l1norm)
        h5file.create_dataset(f"sterrs/{field}/l2norm", data=l2norm)

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
    workdir, casedir, mtype,
    snapshots=True, simtimeerr=True, walltimeerr=True, sterrs=True,
    rank=0
):  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    """Deal with one single case.
    """

    def _del_key(key):
        try:
            del h5file[key]
        except KeyError:
            pass

    # determine file mode
    fmode = "w" if all([snapshots, simtimeerr, walltimeerr, sterrs]) else "a"

    # files, directories, and paths
    casename = casedir.name
    midpath = casedir.parent.relative_to(workdir)
    workdir.joinpath("outputs", midpath).mkdir(parents=True, exist_ok=True)
    h5filename = workdir.joinpath("outputs", midpath, f"{casename}-{mtype}.h5")

    # nothing to do
    if not any([snapshots, simtimeerr, walltimeerr, sterrs]):
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
            _del_key("field")
            h5file = get_snapshots(cfg, graph, fields, h5file, h5kwargs, rank)

        if simtimeerr:
            fields = ["u", "v"]  # fields of our interest
            cfg.eval_times = list(map(float, numpy.linspace(0., 80., 81)))
            _del_key("simtime")
            h5file = get_simtime_errs(cfg, graph, fields, h5file, h5kwargs, rank)

        if walltimeerr:
            fields = ["u", "v"]  # fields of our interest
            cfg.eval_times = [0., 40.]
            _del_key("walltime")
            h5file = get_walltime_errs(cfg, graph, fields, mtype, h5file, h5kwargs, rank)

        if sterrs:
            fields = ["u", "v"]  # fields of our interest
            _del_key("sterrs")
            h5file = get_spatial_temporal_errs(cfg, graph, fields, h5file, h5kwargs, rank)

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
    inps.put((topdir, topdir.joinpath("base-cases", "nl3-nn128-npts8192"), "raw", False, False, True, False))
    inps.put((topdir, topdir.joinpath("base-cases", "nl3-nn256-npts4096"), "raw", True, False, False, False))
    inps.put((topdir, topdir.joinpath("cyclic-sum", "nl3-nn128-npts8192"), "raw", False, False, True, False))

    # spawning processes
    procs = []
    # for m in range(ctx.cpu_count()//4):
    for m in range(1):  # not parallelizng because out-of-memory on the personal latptop
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
