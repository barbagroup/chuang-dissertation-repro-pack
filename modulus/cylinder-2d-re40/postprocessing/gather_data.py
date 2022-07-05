#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of Taylor-Green vortex 2D Re100 w/ single network mode.
"""
import collections
import itertools
import multiprocessing
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


def _get_snapshots(workdir, eval_times, fields=["u", "v", "p"], device="cpu"):
    """Get data from a single case.
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

    print(f"Using flow-net-{mxstep}.pth as the final model.")

    # get the computational graph
    graph = {}
    _, _, graph["orig"], dtype = get_model_from_file(cfg, inferdir.joinpath(f"flow-net-{mxstep}.pth"), 2, device)
    _, _, graph["swa"], _ = get_model_from_file(cfg, inferdir.joinpath(f"swa-model-{mxstep}.pth"), 2, device)

    # get a subset in the computational graph that gives us desired quantities
    model = {key: Graph(val, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields)) for key, val in graph.items()}

    # an empty snapshot contour data holder
    snapshots = collections.defaultdict(lambda: collections.defaultdict(dict))

    # generate gridlines for contour
    snapshots.update({"x": numpy.linspace(xbg, xed, 401), "y": numpy.linspace(ybg, yed, 201)})
    snapshots.update({key: (val[1:] + val[:-1]) / 2 for key, val in snapshots.items()})
    snapshots["x"], snapshots["y"] = numpy.meshgrid(snapshots["x"], snapshots["y"])
    shape = snapshots["x"].shape

    # torch version of gridlines; reshape to N by 1
    torchx = torch.tensor(snapshots["x"].reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
    torchy = torch.tensor(snapshots["y"].reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)

    for time in eval_times:
        for key in ["orig", "swa"]:
            preds = model[key]({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
            preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}
            snapshots[time][key] = preds

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
    workdir.joinpath("outputs").mkdir(exist_ok=True)

    # cases' names
    cases = ["nl5-nn128-npts81920", "nl5-nn256-npts81920"]

    # target fields
    fields = ["u", "v", "p", "vorticity_z"]

    # gather snapshot data
    get_snapshots(workdir, cases, [float(val) for val in range(0, 21)], fields, force, "cpu")

    return 0


if __name__ == "__main__":
    import argparse

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("cylinder-2d-re40").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `cylinder-2d-re40`.")

    root = root.joinpath("cylinder-2d-re40")

    # cmd arguments
    parser = argparse.ArgumentParser(description="Post-processing Modulus Cylinder 2D Re40")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    # calling the main function
    sys.exit(main(root, args.force))
