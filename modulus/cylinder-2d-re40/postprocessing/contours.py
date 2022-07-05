#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot contours.
"""
import multiprocessing
import pathlib
import numpy
import h5py
from matplotlib import pyplot
from matplotlib import ticker

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})


def plotter(_t, _x, _y, _u, _v, _p, _wz, _fname):

    # fields
    fields = [_u, _v, _p, _wz]

    # titles
    titles = [r"$u$-velocity", r"$v$-velocity", r"pressure", r"$z$-vorticity"]

    # colored contour levels
    lvl1 = [
        numpy.linspace(-0.1, 1.1, 13),
        numpy.linspace(-0.5, 0.5, 21),
        numpy.linspace(-0.5, 0.5, 21),
        numpy.linspace(-3.0, 3.0, 13),
    ]

    # contour line levels
    lvl2 = [
        numpy.linspace(0.0, 1.0, 6),
        numpy.linspace(-0.5, 0.5, 11),
        numpy.linspace(-0.5, 0.5, 11),
        numpy.linspace(-3.0, 3.0, 13),
    ]

    # subplot locations
    locs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # plot
    fig, axs = pyplot.subplots(2, 2, sharex=True, sharey=False, figsize=(8, 6), dpi=166)
    fig.suptitle(rf"Flow distribution, $Re=40$ at $t={_t}$, PINN")

    for i in range(4):
        ct1 = axs[locs[i]].contourf(_x, _y, fields[i], lvl1[i], cmap="cividis", extend="both")
        ct2 = axs[locs[i]].contour(_x, _y, fields[i], lvl2[i], colors='black', linewidths=0.5)
        axs[locs[i]].clabel(ct2, lvl2[i], fmt="%1.1f", inline_spacing=0.25, fontsize="small")
        axs[locs[i]].add_artist(pyplot.Circle((0., 0.), 0.5, color="w", zorder=10))
        axs[locs[i]].set_xlim(-3, 4)
        axs[locs[i]].set_ylim(-2.5, 2.5)
        axs[locs[i]].set_aspect("equal", "box")
        axs[locs[i]].set_ylabel("y")
        axs[locs[i]].set_title(titles[i])

        fmt1 = ticker.ScalarFormatter(useOffset=True, useMathText=True)
        fmt1.set_powerlimits((0, 0))
        cbar = fig.colorbar(ct1, ax=axs[locs[i]], ticks=lvl2[i], format=fmt1)
        cbar.ax.get_yaxis().set_offset_position("left")

    axs[0, 0].set_ylabel("y")
    axs[1, 0].set_ylabel("y")
    axs[1, 0].set_xlabel("x")
    axs[1, 1].set_xlabel("x")
    fig.savefig(_fname, bbox_inches="tight", dpi=166)
    pyplot.close(fig)


def plot_single_case(workdir, figdir, jobname, force=False):
    """Plot for a single case.
    """

    def worker(_rank: int, _inputs: multiprocessing.JoinableQueue):
        while True:
            _t, _x, _y, _u, _v, _p, _wz, _fname = _inputs.get(block=True, timeout=None)

            if _fname.is_file() and not force:
                print(f"[Rank {_rank:2d}] Skipping {_fname}")
            else:
                print(f"[Rank {_rank:2d}] Plotting {_fname}")
                plotter(_t, _x, _y, _u, _v, _p, _wz, _fname)
            _inputs.task_done()
            print(f"[Rank {_rank:2d}] Done plotting {_fname}")

    # initialize an empty queue
    inputs = multiprocessing.JoinableQueue()

    # spawn processes
    procs = []
    for rank in range(multiprocessing.cpu_count()//2):
        proc = multiprocessing.Process(target=worker, args=(rank, inputs))
        proc.start()
        procs.append(proc)

    # read data
    with h5py.File(workdir.joinpath("outputs", f"{jobname}.snapshot.h5"), "r") as dset:
        x = dset["x"][...]
        y = dset["y"][...]

        for time, data in dset.items():

            if time in ["x", "y"]:
                continue

            for mtype, vals in data.items():
                inputs.put((
                    time, x, y, vals["u"][...], vals["v"][...], vals["p"][...], vals["vorticity_z"][...],
                    figdir.joinpath(f"{jobname}-{mtype}-t={time.zfill(4)}.png")
                ))

    # wait until all tasks are done, and the queue is empty
    inputs.join()

    # shut down all child processes
    for rank, proc in enumerate(procs):
        print(f"Closing rank {rank}")
        proc.terminate()


def main(workdir, figdir, force=False):
    """Main function.
    """

    # cases
    cases = ["nl5-nn128-npts81920", "nl5-nn256-npts81920"]

    for job in cases:
        print(f"Plotting {job}")
        plot_single_case(workdir, figdir, job, force)

    return 0


if __name__ == "__main__":
    import sys
    import argparse

    # directories
    rootdir = pathlib.Path(__file__).resolve().parents[1]
    figdir = rootdir.joinpath("figures")
    figdir.mkdir(exist_ok=True)

    # cmd arguments
    parser = argparse.ArgumentParser(description="Post-processing Modulus Cylinder 2D Re200")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    sys.exit(main(rootdir, figdir, args.force))
