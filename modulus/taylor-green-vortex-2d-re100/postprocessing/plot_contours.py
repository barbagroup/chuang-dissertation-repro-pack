#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot contours of each case for visual comparisons.
"""
import sys
import collections
import itertools
import pathlib
import multiprocessing
import re
import numpy
import h5py
from matplotlib import pyplot
from matplotlib import cm
from matplotlib import colors


# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
    "figure.dpi": 166,
})


def analytical_solution(x, y, t, nu, field, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    if field == "u":
        return V0 * numpy.cos(x/L) * numpy.sin(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "v":
        return - V0 * numpy.sin(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "p":
        return - rho * V0**2 * numpy.exp(-4.*nu*t/L**2) * (numpy.cos(2.*x/L) + numpy.cos(2.*y/L)) / 4.
    elif field == "wz":
        return - 2. * V0 * numpy.cos(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2) / L
    elif field == "KE":  # kinetic energy
        return numpy.pi**2 * L**2 * V0**2 * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "KEDR":  # kinetic energy dissipation rate
        return 4. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "enstrophy":  # enstrophy
        return 2. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    else:
        raise ValueError


def single_snapshot_plotter(rank: int, force: bool, inputs: multiprocessing.JoinableQueue):
    """Plot contours.
    """
    while True:
        x, y, val, errval, time, field, mtype, fname = inputs.get(block=True, timeout=None)

        if fname.is_file() and not force:
            print(f"[rank {rank:2d}] {fname.name} exists. Skipping.")
            inputs.task_done()
            continue

        print(f"[rank {rank:2d}] Plotting {fname.name}")
        ans = analytical_solution(x, y, float(time), 0.01, field)
        fig = pyplot.figure(constrained_layout=True, figsize=(6, 3))
        fig.suptitle(f"Taylor-Green 2D, Re=100, field: {field}, model: {mtype}, t={time}")

        gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[0.95, 0.05])

        # analytical solutions
        normalizer = colors.Normalize(vmin=ans.min(), vmax=ans.max(), clip=True)
        ax = fig.add_subplot(gs[0, 0])
        ct = ax.contourf(x, y, ans, 32, cmap="turbo", norm=normalizer)
        ax.set_title("Analytical")

        # predictions solutions
        ax = fig.add_subplot(gs[0, 1])
        ax.contourf(x, y, val, 32, cmap="turbo", norm=normalizer)
        ax.set_title("PINN")

        # shared colorbar for analytical and predicted solutions
        ax = fig.add_subplot(gs[1, :2])
        fig.colorbar(ct, cax=ax, orientation="horizontal")

        # absolute error
        normalizer = colors.BoundaryNorm(range(-7, 1), cm.turbo.N)
        ax = fig.add_subplot(gs[0, 2])
        ct = ax.contourf(x, y, numpy.log10(errval), 9, cmap="turbo", norm=normalizer)
        ax.set_title("Absolute error")

        # colorbar for absolute error
        ax = fig.add_subplot(gs[1, 2])
        cbar = fig.colorbar(cm.ScalarMappable(normalizer, cm.turbo), cax=ax, orientation='horizontal')
        cbar.ax.set_xticks(range(-7, 1, 2), [rf"$10^{{{i}}}$" for i in range(-7, 1, 2)])

        fig.savefig(fname)
        pyplot.close(fig)

        inputs.task_done()
        print(f"[rank {rank:2d}] Saved to {fname.name}")


def plot_single_snapshot(cases, workdir, figdir, force=False):
    """Plot figures for each single case and single snapshot.
    """

    # input queues
    inputs = multiprocessing.JoinableQueue(multiprocessing.cpu_count()*2)  # limit the max size to avoid OOM

    # spawning workers
    procs = []
    for rank in range(multiprocessing.cpu_count()):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=single_snapshot_plotter, args=(rank, force, inputs))
        proc.start()
        procs.append(proc)

    # adding tasks to the queue
    for job in cases:
        figdir.joinpath(job).mkdir(exist_ok=True)

        with h5py.File(workdir.joinpath(f"{job}.snapshot.h5"), "r") as dsets:
            x = dsets["x"][...]
            y = dsets["y"][...]

            for time, dset in dsets.items():
                if time in ["x", "y"]:
                    continue

                for field, mtype in itertools.product(["u", "v"], ["orig", "swa"]):
                    fname = figdir.joinpath(job, f"{job}-{mtype}-{field}-t{time.zfill(3)}.png")
                    val = dset[mtype][field][...]
                    errval = dset[mtype][f"err-{field}"][...]

                    # this `put` waits until an available slot to avoid OOM
                    inputs.put((x, y, val, errval, time, field, mtype, fname), block=True, timeout=None)

    # wait until the queue is empty
    inputs.join()

    # terminate workers
    for rank, proc in enumerate(procs):
        print(f"Terminating rank {rank}")
        proc.terminate()


def single_arch_err_plotter(rank: int, force: bool, inputs: multiprocessing.JoinableQueue):
    """Plot comparisons within an architecture.
    """
    while True:
        x, y, errvals, field, mtype, nptss, times, fname, nn, nl = inputs.get(block=True, timeout=None)

        if fname.is_file() and not force:
            print(f"[rank {rank:2d}] {fname.name} exists. Skipping.")
            inputs.task_done()
            continue

        print(f"[rank {rank:2d}] Plotting {fname.name}")
        fig = pyplot.figure(constrained_layout=True, figsize=(6.5, 9))
        gs = fig.add_gridspec(nrows=len(nptss), ncols=len(times)+1, width_ratios=[1]*len(times)+[0.1])

        normalizer = colors.BoundaryNorm(range(-4, 2), cm.turbo.N, extend="both")  # 10^{-4} to 10^{1}
        for j, t in enumerate(times):
            for i, nbs in enumerate(nptss):
                ax = fig.add_subplot(gs[i, j])
                ax.contourf(x, y, numpy.log10(errvals[nbs][t]), 6, cmap="turbo", norm=normalizer)
                ax.set_title(rf"$t={t}$, $N_{{bs}}={nbs}$", fontsize=7)
                ax.tick_params(axis='x', labelsize=7)
                ax.tick_params(axis='y', labelsize=7)

                if i == len(nptss) - 1:
                    ax.set_xlabel(r"$x$", fontsize=7)
                else:
                    pyplot.setp(ax.get_xticklabels(), visible=False)

                if j == 0:
                    ax.set_ylabel(r"$y$", fontsize=7)
                else:
                    pyplot.setp(ax.get_yticklabels(), visible=False)

        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(cm.ScalarMappable(normalizer, cm.turbo), cax=cax, orientation='vertical')
        cax.set_yticks(range(-4, 2), [rf"$10^{{{i}}}$" for i in range(-4, 2)])
        cax.tick_params(axis='x', labelsize=7)
        cax.tick_params(axis='y', labelsize=7)

        fig.suptitle(
            rf"2D TGV, $Re=100$, absolute errors, $N_n={nn}$, $N_l={nl}$, field: ${field}$, model: {mtype}",
            fontsize=12
        )
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.1)
        fig.savefig(fname, bbox_inches="tight")
        pyplot.close(fig)

        inputs.task_done()
        print(f"[rank {rank:2d}] Saved to {fname.name}")


def plot_single_arch_err(cases, nbss, workdir, figdir, force=False):
    """Plot figures for each single case and single snapshot.
    """

    # input queues
    inputs = multiprocessing.JoinableQueue(multiprocessing.cpu_count())  # limit the max size to avoid OOM

    # spawning workers
    procs = []
    for rank in range(multiprocessing.cpu_count()):
        print(f"Spawning rank {rank}")
        proc = multiprocessing.Process(target=single_arch_err_plotter, args=(rank, force, inputs))
        proc.start()
        procs.append(proc)

    # adding tasks to the queue
    for job in cases:

        figdir.joinpath(job).mkdir(exist_ok=True)

        data = {
            "orig": {"u": collections.defaultdict(dict), "v": collections.defaultdict(dict)},
            "swa": {"u": collections.defaultdict(dict), "v": collections.defaultdict(dict)},
        }

        nn = re.search(r"nn(\d+)", job).group(1)
        nl = re.search(r"nl(\d+)", job).group(1)

        for nbs in nbss:
            with h5py.File(workdir.joinpath(f"{job}-npts{nbs}.snapshot.h5"), "r") as dsets:

                # now we just assume all cases have the same x and y points
                x = dsets["x"][...]
                y = dsets["y"][...]

                for time, dset in dsets.items():
                    if time not in ["20", "60", "100"]:
                        continue

                    for field, mtype in itertools.product(["u", "v"], ["orig", "swa"]):
                        data[mtype][field][nbs][time] = dset[mtype][f"err-{field}"][...]

        for field, mtype in itertools.product(["u", "v"], ["orig", "swa"]):
            fname = figdir.joinpath(job, f"{job}-{mtype}-{field}.png")

            # this `put` waits until an available slot to avoid OOM
            inputs.put(
                (x, y, data[mtype][field], field, mtype, nbss, ["20", "60", "100"], fname, nn, nl),
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
    figdir = rootdir.joinpath("figures", "contours")
    figdir.mkdir(exist_ok=True)

    # cases
    cases = [
        f"nl{nl}-nn{nn}-npts{npts}"
        for nl, nn, npts in itertools.product([1, 2, 3], [2**i for i in range(4, 9)], [2**i for i in range(10, 17)])
    ]

    # plot case by case
    plot_single_snapshot(cases, workdir, figdir, force)

    # cases
    cases = [f"nl{nl}-nn{nn}" for nl, nn in itertools.product([1, 2, 3], [2**i for i in range(4, 9)])]
    nbss = [2**i for i in range(10, 17)]

    # plot case by case
    plot_single_arch_err(cases, nbss, workdir, figdir, force)


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
