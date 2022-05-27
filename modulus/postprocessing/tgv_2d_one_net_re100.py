#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of Taylor-Green vortex 2D Re100 w/ single network mode.
"""
import lzma
import multiprocessing
import sys
import numpy
import torch
from io import BytesIO
from datetime import datetime
from pathlib import Path
import pandas
from sympy import sympify
from matplotlib import pyplot
from modulus.key import Key
from modulus.graph import Graph
from modulus.architecture.fully_connected import FullyConnectedArch

# find helpers
for parent in Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.pdes import IncompNavierStokes  # pylint: disable=import-error
        from helpers.pdes import Vorticity  # pylint: disable=import-error
        from helpers.pdes import QCriterion  # pylint: disable=import-error
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
})


def analytical_solution(x, y, t, nu, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    with torch.no_grad():
        u =   V0 * torch.cos(x/L) * torch.sin(y/L) * torch.exp(-2.*nu*t/L**2)
        v = - V0 * torch.sin(x/L) * torch.cos(y/L) * torch.exp(-2.*nu*t/L**2)
        p = - rho * V0**2 * (torch.cos(2*x/L) + torch.cos(2*y/L)) * torch.exp(-4.*nu*t/L**2) / 4
    return {"u": u, "v": v, "p": p}


def create_graph(cfg, params=None):
    """Create a computational graph without proper parameters in the model.
    """

    xbg = ybg = - cfg.custom.scale * numpy.pi
    xed = yed = cfg.custom.scale * numpy.pi

    # network
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p"),],
        periodicity={"x": (float(xbg), float(xed)), "y": (float(ybg), float(yed)),},
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    # update parameters
    if params is not None:
        net.load_state_dict(params)

    # navier-stokes equation
    nseq = IncompNavierStokes(cfg.custom.nu, cfg.custom.rho, 2, True)

    # vorticity
    vorticity = Vorticity(dim=2)

    # q-criterion
    qcriterion = QCriterion(dim=2)

    nodes = \
        nseq.make_nodes() + vorticity.make_nodes() + qcriterion.make_nodes() + \
        [net.make_node(name="flow-net", jit=cfg.jit)]

    dtype = next(net.parameters()).dtype

    return nodes, dtype


def get_subgraph(graph, name):
    """Get a subset in the computational graph.
    """

    if name == "uvwp":
        invars = ["x", "y", "t"]
        outvars = ["u", "v", "p"]
    elif name == "vorticity":
        invars = ["x", "y", "t"]
        outvars = ["vorticity_x", "vorticity_x", "vorticity_x"]
    elif name == "qcriterion":
        invars = ["x", "y", "t"]
        outvars = ["qcriterion",]
    else:
        raise NotImplementedError(f"{name} not implemented")

    model = Graph(graph, Key.convert_list(invars), Key.convert_list(outvars))

    return model


def get_run_time(timestamps):
    """Process timestamps, eliminate gaps betwenn Slurm job submissions, and return accumulated run time.

    The returned run times are in seconds.
    """

    timestamps = numpy.array(timestamps)
    diff = timestamps[1:] - timestamps[:-1]
    truncated = numpy.sort(diff)[5:-5]

    avg = truncated.mean()
    std = truncated.std(ddof=1)
    diff[numpy.logical_or(diff < avg-2*std, diff > avg+2*std)] = avg
    diff = numpy.concatenate((numpy.full((1,), avg), diff))  # add back the time for the first result

    return numpy.cumsum(diff)


def get_snapshot(cfg, filename):
    """Read snapshot data.
    """

    # load the snatshot data
    with lzma.open(filename, "rb") as obj:  # open the file
        snapshot = torch.load(obj, map_location=cfg.device)

    # load the model parameters form the snapshot data
    with BytesIO(snapshot["model"]) as obj:  # load the model
        params = torch.jit.load(obj, map_location=cfg.device).state_dict()

    # get a computational graph
    graph, dtype = create_graph(cfg, params)

    # timestamp and training iteration
    timestamp = snapshot["time"]
    step = snapshot["step"]

    return step, timestamp, graph, dtype


def plot_latest_contours(cfg, workdir):
    """Plot a snatshop from an inferencer.
    """

    # helper function
    def plot_one_contour(x, y, val, xlab, ylab, title, fpath):
        """Plot single one contour
        """

        pyplot.figure(figsize=(4, 3), dpi=166, constrained_layout=True)
        pyplot.contourf(x, y, val, 128)
        pyplot.xlabel(xlab)
        pyplot.ylabel(xlab)
        pyplot.title(title)
        pyplot.colorbar()
        pyplot.savefig(fpath, dpi=166)
        pyplot.close("all")

    mxstep = max([
        fname.name.replace("flow-net-", "").replace(".pth", "") for
        fname in workdir.joinpath("inferencers").glob("flow-net-*.pth")
    ], key=int)

    # get the computational graph
    step, timestamp, graph, dtype = get_snapshot(cfg, workdir.joinpath("inferencers", f"flow-net-{mxstep}.pth"))

    # coordinates to infer
    npx = numpy.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, 129)  # vertices
    npy = numpy.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, 129)  # vertices
    npx, npy = numpy.meshgrid(npx, npy)
    shape = npx.shape

    # reshape to N by 1 vectors and create torch vectors (sharing the same memory space)
    torchx = torch.tensor(npx.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)
    torchy = torch.tensor(npy.reshape(-1, 1), dtype=dtype, device=cfg.device, requires_grad=True)

    # make sure the figures' prefix exists
    workdir.joinpath("figures").mkdir(exist_ok=True)

    # get subset in the computational graph that gives us desired quantities
    print("\tgetting model")
    outvars = ["u", "v", "p", "vorticity_z", "qcriterion"]
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(outvars))

    # plot time frame by time frame
    for time in cfg.eval_times:

        print("\tpredicting")
        preds = model({"x": torchx, "y": torchy, "t": torch.full_like(torchx, time)})
        preds = {k: v.detach().cpu().numpy().reshape(shape) for k, v in preds.items()}

        for key in outvars:
            print(f"\tplotting {key.replace('_', '-')}")
            title = f"Step {step:07d}; Prediction; {key.replace('_', '-')}; T={time}"
            fname = f"step{step:07d}-pred-{key.replace('_', '-')}-t{time}.png"
            plot_one_contour(npx, npy, preds[key], "x", "y", title, workdir.joinpath("figures", fname))

        ans = analytical_solution(torchx, torchy, torch.full_like(torchx, time), cfg.custom.nu)
        ans = {k: v.detach().cpu().numpy().reshape(shape) for k, v in ans.items()}

        for key in ["u", "v", "p"]:
            print(f"\tplotting {key}-error")
            err = abs(preds[key] - ans[key])
            title = f"Step {step:07d}; Error; {key}; T={time}"
            fname = f"step{step:07d}-error-{key}-t{time}.png"
            plot_one_contour(npx, npy, err, "x", "y", title, workdir.joinpath("figures", fname))


def get_snapshot_err(cfg, filename, x, y, times):
    """Get the error of one snapshot.
    """
    # cell area
    area = (2*cfg.custom.scale*numpy.pi)**2 / x.shape[0] / x.shape[1]

    # get the computational graph
    step, timestamp, graph, dtype = get_snapshot(cfg, filename)
    timestamp = datetime.fromisoformat(timestamp).timestamp()  # convert to epoch time in seconds

    # get subset in the computational graph that gives us desired quantities
    outvars = ["u", "v", "p"]
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(outvars))

    # make them nx*ny by 1, required by the model
    x = x.reshape(-1, 1).to(dtype).to(cfg.device)
    y = y.reshape(-1, 1).to(dtype).to(cfg.device)

    # data holder
    l1norms = {key: numpy.zeros(len(times), dtype=float) for key in ["u", "v", "p"]}
    l2norms = {key: numpy.zeros(len(times), dtype=float) for key in ["u", "v", "p"]}

    for i, time in enumerate(times):
        t = torch.full_like(x, time)
        preds = model({"x": x, "y": y, "t": t})
        ans = analytical_solution(x, y, t, cfg.custom.nu)

        for key in ["u", "v", "p"]:
            err = abs(preds[key]-ans[key])
            l1norms[key][i]= float(err.sum() * area)  # convert torch array to native float type
            l2norms[key][i]= float(torch.sqrt((err**2).sum() * area))

    return step, timestamp, l1norms, l2norms


def get_norm_history(cfg, workdir):
    """Plot l2norm v.s. iterations.
    """

    files = list(workdir.joinpath("inferencers").glob("flow-net-*.pth"))

    nx = ny = 128  # cell centers
    area = (2*cfg.custom.scale*numpy.pi)**2 / nx / ny

    # independent variables; we only need pytorch's versions in this function
    torchx = torch.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, nx+1, dtype=torch.float32)
    torchy = torch.linspace(-cfg.custom.scale*numpy.pi, cfg.custom.scale*numpy.pi, ny+1, dtype=torch.float32)
    torchx = (torchx[:-1] + torchx[1:]) / 2.
    torchy = (torchy[:-1] + torchy[1:]) / 2.
    torchx, torchy = torch.meshgrid(torchx, torchy, indexing="xy")

    # initialize data holders
    data = pandas.DataFrame(
        data=None,
        index=pandas.Index([], dtype=int, name="iteration"),
        columns=pandas.MultiIndex.from_product([
            ["l1norm", "l2norm"],
            ["u", "v", "p"],
            cfg.eval_times
        ]).append(pandas.Index([("timestamp", "", ""),])),
    )

    # jobs
    jobs = multiprocessing.JoinableQueue()
    for i, filename in enumerate(files):
        jobs.put((i, filename))

    # results
    results = multiprocessing.Queue()

    def worker(inputs, outputs, cfg, x, y, times):
        while True:
            try:
                i, fname = inputs.get(True, 2)
            except multiprocessing.queues.Empty:
                inputs.close()
                outputs.close()
                return
            print(f"\tprocessing {fname.name}")
            result = get_snapshot_err(cfg, fname, torchx, torchy, times)
            outputs.put((i, result))
            inputs.task_done()

    procs = []
    for _ in range(multiprocessing.cpu_count()//2):
        proc = multiprocessing.Process(target=worker, args=(jobs, results, cfg, torchx, torchy, cfg.eval_times))
        proc.start()
        procs.append(proc)

    jobs.join()

    while not results.empty():
        _, (step, timestamp, _l1norm, _l2norm) = results.get(False)
        data.loc[step, "timestamp"] = timestamp
        for key in ["u", "v", "p"]:
            print(len(data.loc[step, ("l1norm", key)]), len(_l1norm[key]))
            data.loc[step, ("l1norm", key)] = _l1norm[key]
            data.loc[step, ("l2norm", key)] = _l2norm[key]

    data = data.sort_index()
    data["runtime"] = get_run_time(data["timestamp"])
    data.to_csv(workdir.joinpath("norms.csv"))

    return data


def plot_norm_history(cfg, workdir):
    """Plot error norm history.
    """

    # helper function
    def plot_one_history(x, y, llabs, xlab, ylab, title, tickstyle, fname):
        pyplot.figure(figsize=(4, 3), dpi=166, constrained_layout=True)
        for i, label in enumerate(llabs):
            pyplot.semilogy(x, y[:, i], lw=1, alpha=0.7, label=f"t={label}")
        pyplot.xlabel(xlab)
        pyplot.ylabel(ylab)
        pyplot.title(title)
        pyplot.legend(loc=0, ncol=2)
        pyplot.ticklabel_format(axis="x", style=tickstyle, scilimits=(0, 0))
        pyplot.savefig(workdir.joinpath("figures", fname), dpi=166)
        pyplot.close("all")

    workdir.joinpath("figures").mkdir(exist_ok=True)

    if workdir.joinpath("norms.csv").is_file():
        data = pandas.read_csv(workdir.joinpath("norms.csv"), header=[0, 1, 2], index_col=0)
        data = data.sort_index()
    else:
        data = get_norm_history(cfg, workdir)

    for key in ["u", "v", "p"]:
        plot_one_history(
            data.index, data[("l1norm", key)].values, cfg.eval_times, "Iterations", "L1-norm",
            f"{key}: L1 error v.s. iterations", "sci", f"l1norm-hist-{key}.png"
        )

        plot_one_history(
            data.index, data[("l2norm", key)].values, cfg.eval_times, "Iterations", "L2-norm",
            f"{key}: L2 error v.s. iterations", "sci", f"l2norm-hist-{key}.png",
        )

        plot_one_history(
            data["runtime"]/3600, data[("l1norm", key)].values, cfg.eval_times, "Run time (hours)",
            "L1-norm", f"{key}: L1 error v.s. run time", "plain", f"l1norm-hist-run-time-{key}.png",
        )

        plot_one_history(
            data["runtime"]/3600, data[("l2norm"), key].values, cfg.eval_times, "Run time (hours)",
            "L2-norm", f"{key}: L2 error v.s. run time", "plain", f"l2norm-hist-run-time-{key}.png",
        )


def plot_residual_history(cfg, workdir):
    """Plot the PDE residual history from Modulus monitor.
    """
    workdir.joinpath("figures").mkdir(exist_ok=True)

    continuity = numpy.loadtxt(workdir.joinpath("monitors", "continuity_res.csv"), float, delimiter=",", skiprows=1)
    momentum_x = numpy.loadtxt(workdir.joinpath("monitors", "momentum_x_res.csv"), float, delimiter=",", skiprows=1)
    momentum_y = numpy.loadtxt(workdir.joinpath("monitors", "momentum_y_res.csv"), float, delimiter=",", skiprows=1)

    # moving average (against previous values)
    wsize = 5
    window = numpy.concatenate((numpy.zeros(int(wsize-1)), numpy.ones(wsize)/wsize))
    continuity[wsize-1:, 1] = numpy.convolve(continuity[:, 1], window, "same")[wsize-1:]
    momentum_x[wsize-1:, 1] = numpy.convolve(momentum_x[:, 1], window, "same")[wsize-1:]
    momentum_y[wsize-1:, 1] = numpy.convolve(momentum_y[:, 1], window, "same")[wsize-1:]

    pyplot.figure(figsize=(4, 3), dpi=166, constrained_layout=True)
    pyplot.semilogy(continuity[:, 0], continuity[:, 1], lw=1, alpha=0.7, label="Continuity")
    pyplot.semilogy(momentum_x[:, 0], momentum_x[:, 1], lw=1, alpha=0.7, label="Momentum x")
    pyplot.semilogy(momentum_y[:, 0], momentum_y[:, 1], lw=1, alpha=0.7, label="Momentum y")
    pyplot.xlabel(r"Iterations")
    pyplot.ylabel(r"Residual")
    pyplot.title(r"PDE residual v.s. training iteration")
    pyplot.legend(loc=0)
    pyplot.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    pyplot.savefig(workdir.joinpath("figures", "pde-residual-hist.png"), dpi=166)


def plot_residual_history_comp(rootdir):
    """Plot history comparison of different cases.
    """

    rootdir.joinpath("figures").mkdir(exist_ok=True)

    for hist in ["continuity", "momentum_x", "momentum_y"]:
        pyplot.figure(figsize=(4, 3), dpi=166, constrained_layout=True)

        for job in [f"a100_{ngpus}" for ngpus in (1, 2, 4 ,8)]:
            data = numpy.loadtxt(
                rootdir.joinpath(job, "outputs", "monitors", f"{hist}_res.csv"), float,
                delimiter=",", skiprows=1
            )

            wsize = 10
            window = numpy.concatenate((numpy.zeros(int(wsize-1)), numpy.ones(wsize)/wsize))
            data[wsize-1:, 1] = numpy.convolve(data[:, 1], window, "same")[wsize-1:]

            pyplot.semilogy(data[:, 0], data[:, 1], lw=1, alpha=0.7, label=f"{job.replace('a100_', '')} A100")

        pyplot.xlabel("Iterations")
        pyplot.ylabel("Residual")
        pyplot.title(f"{hist.capitalize().replace('_', ' ')}: PDE residual v.s. training iteration")
        pyplot.legend(loc=0)
        pyplot.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        pyplot.savefig(rootdir.joinpath("figures", f"pde-residual-hist-{hist}.png"), dpi=166)


def main_case_by_case(rootdir):
    """Process cases one by one, dealing w/ data within each case.
    """

    for job in [f"a100_{ngpus}" for ngpus in (1, 2, 4 ,8)]:

        joboutput = rootdir.joinpath(job, "outputs")

        cfg = OmegaConf.load(joboutput.joinpath(".hydra", "config.yaml"))
        cfg.device = "cpu"
        cfg.custom.scale = float(sympify(cfg.custom.scale).evalf())
        cfg.eval_times = [0.0, 0.5, 2.0, 8.0, 32.0]

        print(f"Handling {joboutput}")
        plot_latest_contours(cfg, joboutput)
        plot_residual_history(cfg, joboutput)
        plot_norm_history(cfg, joboutput)


def main_comparison(rootdir):
    """Comparing data across different cases.
    """
    plot_residual_history_comp(rootdir)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    # find the root of the folder `modulus`
    for root in Path(__file__).resolve().parents:
        if root.joinpath("cases").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `cases`.")

    root = root.joinpath("cases", "taylor-green-vortex-2d", "one-net", "re100")

    main_case_by_case(root)
    # main_comparison(root)
