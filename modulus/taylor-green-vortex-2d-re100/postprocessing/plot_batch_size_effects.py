#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plotting function related to batch size effects.
"""
import sys
import itertools
import pathlib
import multiprocessing
import re
import numpy
import pandas
from matplotlib import pyplot


# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
    "figure.dpi": 166,
})


def timestamps_to_elapsed_times(timestamps, noutliers=10, offset=0.):
    """Converts timestamps to elapsed times and eliminates outliers due to restart.

    Notes
    -----
    Timestamps from different training stages must be processed separately.
    """

    timestamps = numpy.array(timestamps)
    diff = timestamps[1:] - timestamps[:-1]
    truncated = numpy.sort(diff)[noutliers:-noutliers]

    assert len(truncated) != 0, f"{diff}\n{truncated}"
    avg = truncated.mean()
    std = truncated.std(ddof=1)
    diff[numpy.logical_or(diff < avg-2*std, diff > avg+2*std)] = avg
    diff = numpy.concatenate((numpy.full((1,), 0), diff))  # add back the time for the first result

    return numpy.cumsum(diff) + offset  # cumulative times


def err_vs_iter_plotter(rank: int, force: bool, inputs: multiprocessing.JoinableQueue):
    """Plot err v.s. training iterations under the same arch but different batch sizes.
    """

    while True:
        iters, errs, nbss, time, nl, nn, field, mtype, fname = inputs.get(block=True, timeout=None)

        if fname.is_file() and not force:
            print(f"[rank {rank:2d}] {fname.name} exists. Skipping.")
            inputs.task_done()
            continue

        print(f"[rank {rank:2d}] Plotting {fname.name}")
        fig, axs = pyplot.subplots(
            3, 1, constrained_layout=True, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 0.3, 1]}
        )
        args = {"lw": 1.5, "ls": "-", "alpha": 0.9}

        lines = []
        for nbs in nbss:

            # adam stage
            lines.append(axs[0].semilogy(iters["adam"][nbs], errs["adam"][nbs], label=rf"$N_{{bs}}={nbs}$", **args)[0])
            axs[0].set_title(r"Training stage: Adam", fontsize=10)
            axs[0].set_xlabel("Iterations")
            axs[0].set_ylabel(r"$l_2$-norm")

            # ncg stage
            axs[2].semilogy(iters["ncg"][nbs], errs["ncg"][nbs], color=lines[-1].get_color(), **args)
            axs[2].set_title(r"Training stage: nonlinear CG", fontsize=10)
            axs[2].set_xlabel("Iterations")
            axs[2].set_ylabel(r"$l_2$-norm")

            # put legend in between
            axs[1].legend(
                handles=lines, title="Batch size", loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=8, ncol=4
            )
            axs[1].axis("off")

        fig.suptitle(
            rf"2D TGV, $Re=100$, $N_n={nn}$, $N_l={nl}$, field: ${field}$, model: {mtype}, $t={time}$, "
            "\n"
            rf"$l_2$-norm v.s. iteration",
            fontsize=12
        )
        fig.savefig(fname, bbox_inches="tight")
        pyplot.close(fig)

        inputs.task_done()
        print(f"[rank {rank:2d}] Saved to {fname.name}")


def plot_err_vs_iter(cases, nbss, times, workdir, figdir, force=False):
    """Errors v.s. iterations.
    """

    # input queues
    inputs = multiprocessing.JoinableQueue(multiprocessing.cpu_count())  # limit the max size to avoid OOM

    # spawning workers
    procs = []
    for rank in range(multiprocessing.cpu_count()):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=err_vs_iter_plotter, args=(rank, force, inputs))
        proc.start()
        procs.append(proc)

    # adding tasks to the queue
    for job in cases:

        figdir.joinpath(job).mkdir(exist_ok=True)

        iters = {
            mtype: {
                field: {
                    time: {optimizer: {} for optimizer in ["adam", "ncg"]} for time in times
                } for field in ["u", "v"]
            } for mtype in ["orig", "swa"]
        }

        errs = {
            mtype: {
                field: {
                    time: {optimizer: {} for optimizer in ["adam", "ncg"]} for time in times
                } for field in ["u", "v"]
            } for mtype in ["orig", "swa"]
        }

        nn = re.search(r"nn(\d+)", job).group(1)
        nl = re.search(r"nl(\d+)", job).group(1)

        for nbs in nbss:
            data = pandas.read_csv(workdir.joinpath(f"{job}-npts{nbs}.walltime.csv"), index_col=0, header=[0, 1, 2, 3])

            # earlier data use "raw" rather than "orig"
            data = data.rename(columns=lambda val: "orig" if val == "raw" else val, level=0)

            # clear the sub-column names that contain "Unnamed"
            for lv in [1, 2, 3]:
                data = data.rename(columns=lambda val: "" if "Unnamed" in val else val, level=lv)

            for time, mtype, field in itertools.product(times, ["orig", "swa"], ["u", "v"]):
                series = data.xs((mtype, "l2norm", field, time), axis=1)
                adam = series[series.index <= 100000]
                ncg = series[series.index >= 100000]
                iters[mtype][field][time]["adam"][nbs] = adam.index
                errs[mtype][field][time]["adam"][nbs] = adam.array
                iters[mtype][field][time]["ncg"][nbs] = ncg.index
                errs[mtype][field][time]["ncg"][nbs] = ncg.array

        for time, mtype, field in itertools.product(times, ["orig", "swa"], ["u", "v"]):
            fname = figdir.joinpath(job, f"{job}-{mtype}-{field}-t={time.zfill(3)}.png")

            # this `put` waits until an available slot to avoid OOM
            inputs.put(
                (iters[mtype][field][time], errs[mtype][field][time], nbss, time, nl, nn, field, mtype, fname),
                block=True, timeout=None
            )

    # wait until the queue is empty
    inputs.join()

    # terminate workers
    for rank, proc in enumerate(procs):
        print(f"Terminating rank {rank}")
        proc.terminate()


def err_vs_runtime_plotter(rank: int, force: bool, inputs: multiprocessing.JoinableQueue):
    """Plot err v.s. elapsed walltime under the same arch but different batch sizes.
    """

    while True:
        wts, errs, nbss, time, nl, nn, field, mtype, fname = inputs.get(block=True, timeout=None)

        if fname.is_file() and not force:
            print(f"[rank {rank:2d}] {fname.name} exists. Skipping.")
            inputs.task_done()
            continue

        print(f"[rank {rank:2d}] Plotting {fname.name}")
        fig, axs = pyplot.subplots(
            3, 1, constrained_layout=True, figsize=(6, 6), gridspec_kw={"height_ratios": [1, 0.3, 1]}
        )
        args = {"lw": 1.5, "ls": "-", "alpha": 0.9}

        lines = []
        for nbs in nbss:

            # adam stage
            lines.append(
                axs[0].semilogy(wts["adam"][nbs]/3600, errs["adam"][nbs], label=rf"$N_{{bs}}={nbs}$", **args)[0]
            )
            axs[0].set_title(r"Training stage: Adam", fontsize=10)
            axs[0].set_xlabel("Elapsed wall time (hours)")
            axs[0].set_ylabel(r"$l_2$-norm")

            # ncg stage
            axs[2].semilogy(wts["ncg"][nbs]/3600, errs["ncg"][nbs], color=lines[-1].get_color(), **args)
            axs[2].set_title(r"Training stage: nonlinear CG", fontsize=10)
            axs[2].set_xlabel("Elapsed wall time (hours)")
            axs[2].set_ylabel(r"$l_2$-norm")

            # put legend in between
            axs[1].legend(
                handles=lines, title="Batch size", loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=8, ncol=4
            )
            axs[1].axis("off")

        fig.suptitle(
            rf"2D TGV, $Re=100$, $N_n={nn}$, $N_l={nl}$, field: ${field}$, model: {mtype}, $t={time}$, "
            "\n"
            rf"$l_2$-norm v.s. elapsed wall time",
            fontsize=12
        )
        fig.savefig(fname, bbox_inches="tight")
        pyplot.close(fig)

        inputs.task_done()
        print(f"[rank {rank:2d}] Saved to {fname.name}")


def plot_err_vs_runtime(cases, nbss, times, workdir, figdir, force=False):
    """Errors v.s. elapsed walltime.
    """

    # input queues
    inputs = multiprocessing.JoinableQueue(multiprocessing.cpu_count())  # limit the max size to avoid OOM

    # spawning workers
    procs = []
    for rank in range(multiprocessing.cpu_count()):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=err_vs_runtime_plotter, args=(rank, force, inputs))
        proc.start()
        procs.append(proc)

    # adding tasks to the queue
    for job in cases:

        figdir.joinpath(job).mkdir(exist_ok=True)

        runtimes = {
            mtype: {
                field: {
                    time: {optimizer: {} for optimizer in ["adam", "ncg"]} for time in times
                } for field in ["u", "v"]
            } for mtype in ["orig", "swa"]
        }

        errs = {
            mtype: {
                field: {
                    time: {optimizer: {} for optimizer in ["adam", "ncg"]} for time in times
                } for field in ["u", "v"]
            } for mtype in ["orig", "swa"]
        }

        nn = re.search(r"nn(\d+)", job).group(1)
        nl = re.search(r"nl(\d+)", job).group(1)

        for nbs in nbss:
            data = pandas.read_csv(workdir.joinpath(f"{job}-npts{nbs}.walltime.csv"), index_col=0, header=[0, 1, 2, 3])

            # earlier data use "raw" rather than "orig"
            data = data.rename(columns=lambda val: "orig" if val == "raw" else val, level=0)

            # clear the sub-column names that contain "Unnamed"
            for lv in [1, 2, 3]:
                data = data.rename(columns=lambda val: "" if "Unnamed" in val else val, level=lv)

            for time, mtype, field in itertools.product(times, ["orig", "swa"], ["u", "v"]):
                series = data.xs((mtype, "l2norm", field, time), axis=1)
                errs[mtype][field][time]["adam"][nbs] = series[series.index <= 100000].array
                errs[mtype][field][time]["ncg"][nbs] = series[series.index >= 100000].array

                timestamps = data.xs("timestamp", axis=1)
                adam_t = timestamps_to_elapsed_times(timestamps[timestamps.index <= 100000], 1)
                ncg_t = timestamps_to_elapsed_times(timestamps[timestamps.index >= 100000], 3, adam_t[-1])
                runtimes[mtype][field][time]["adam"][nbs] = adam_t
                runtimes[mtype][field][time]["ncg"][nbs] = ncg_t

        for time, mtype, field in itertools.product(times, ["orig", "swa"], ["u", "v"]):
            fname = figdir.joinpath(job, f"{job}-{mtype}-{field}-t={time.zfill(3)}.png")

            # this `put` waits until an available slot to avoid OOM
            inputs.put(
                (runtimes[mtype][field][time], errs[mtype][field][time], nbss, time, nl, nn, field, mtype, fname),
                block=True, timeout=None
            )

    # wait until the queue is empty
    inputs.join()

    # terminate workers
    for rank, proc in enumerate(procs):
        print(f"Terminating rank {rank}")
        proc.terminate()


def err_vs_simtime_plotter(rank: int, force: bool, inputs: multiprocessing.JoinableQueue):
    """Plot err v.s. simulation time under the same arch from the last model parameters and w/ different batch sizes.
    """

    while True:
        simtimes, errs, nbss, nl, nn, field, mtype, fname = inputs.get(block=True, timeout=None)

        if fname.is_file() and not force:
            print(f"[rank {rank:2d}] {fname.name} exists. Skipping.")
            inputs.task_done()
            continue

        print(f"[rank {rank:2d}] Plotting {fname.name}")
        fig, ax = pyplot.subplots(1, 1, constrained_layout=True, figsize=(6, 3))
        args = {"lw": 1.5, "ls": "-", "alpha": 0.9}

        for nbs in nbss:
            ax.semilogy(simtimes[nbs], errs[nbs], label=rf"$N_{{bs}}={nbs}$", **args)[0]
            ax.set_xlabel(r"Time in simulation ($t$ in seconds)")
            ax.set_ylabel(r"$l_2$-norm")
            ax.legend(loc=0, title="Batch size", fontsize=8, ncol=4)

        fig.suptitle(
            rf"2D TGV, $Re=100$, $N_n={nn}$, $N_l={nl}$, field: ${field}$, model: {mtype}, "
            "\n"
            rf"$l_2$-norm v.s. $t$",
            fontsize=12
        )
        fig.savefig(fname, bbox_inches="tight")
        pyplot.close(fig)

        inputs.task_done()
        print(f"[rank {rank:2d}] Saved to {fname.name}")


def plot_err_vs_simtime(cases, nbss, workdir, figdir, force=False):
    """Errors v.s. time in simulations.
    """

    # input queues
    inputs = multiprocessing.JoinableQueue(multiprocessing.cpu_count())  # limit the max size to avoid OOM

    # spawning workers
    procs = []
    for rank in range(multiprocessing.cpu_count()):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=err_vs_simtime_plotter, args=(rank, force, inputs))
        proc.start()
        procs.append(proc)

    # adding tasks to the queue
    for job in cases:

        figdir.joinpath(job).mkdir(exist_ok=True)

        simtimes = {mtype: {field: {} for field in ["u", "v"]} for mtype in ["orig", "swa"]}

        errs = {mtype: {field: {} for field in ["u", "v"]} for mtype in ["orig", "swa"]}

        nn = re.search(r"nn(\d+)", job).group(1)
        nl = re.search(r"nl(\d+)", job).group(1)

        for nbs in nbss:
            data = pandas.read_csv(workdir.joinpath(f"{job}-npts{nbs}.simtime.csv"), index_col=0, header=[0, 1, 2])

            # earlier data use "raw" rather than "orig"
            data = data.rename(columns=lambda val: "orig" if val == "raw" else val, level=0)

            # clear the sub-column names that contain "Unnamed"
            for lv in [1, 2]:
                data = data.rename(columns=lambda val: "" if "Unnamed" in val else val, level=lv)

            for mtype, field in itertools.product(["orig", "swa"], ["u", "v"]):
                series = data.xs((mtype, "l2norm", field), axis=1)
                errs[mtype][field][nbs] = series.array
                simtimes[mtype][field][nbs] = series.index

        for mtype, field in itertools.product(["orig", "swa"], ["u", "v"]):
            fname = figdir.joinpath(job, f"{job}-{mtype}-{field}.png")

            # this `put` waits until an available slot to avoid OOM
            inputs.put(
                (simtimes[mtype][field], errs[mtype][field], nbss, nl, nn, field, mtype, fname),
                block=True, timeout=None
            )

    # wait until the queue is empty
    inputs.join()

    # terminate workers
    for rank, proc in enumerate(procs):
        print(f"Terminating rank {rank}")
        proc.terminate()


def main(rootdir, force: bool = False):
    """Main function.
    """

    # folders
    workdir = rootdir.joinpath("outputs")

    # cases
    cases = [f"nl{nl}-nn{nn}" for nl, nn in itertools.product([1, 2, 3], [2**i for i in range(4, 9)])]
    nbss = [2**i for i in range(10, 17)]
    times = ["0", "40"]

    # plot errors v.s. iterations
    figdir = rootdir.joinpath("figures", "err-vs-iters")
    figdir.mkdir(exist_ok=True)
    plot_err_vs_iter(cases, nbss, times, workdir, figdir, force)

    # plot errors v.s. elapsed walltime
    figdir = rootdir.joinpath("figures", "err-vs-walltimes")
    figdir.mkdir(exist_ok=True)
    plot_err_vs_runtime(cases, nbss, times, workdir, figdir, force)

    # plot errors v.s. time in simulation
    figdir = rootdir.joinpath("figures", "err-vs-simtimes")
    figdir.mkdir(exist_ok=True)
    plot_err_vs_simtime(cases, nbss, workdir, figdir, force)


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
    parser = argparse.ArgumentParser(description="Post-processing Modulus TGV 2D Re100")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    # calling the main function
    sys.exit(main(root, args.force))
