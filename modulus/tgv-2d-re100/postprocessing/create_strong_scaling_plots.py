#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
"""
import pathlib
import sys
from h5py import File as h5open
from cycler import cycler
from matplotlib import pyplot

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def create_strong_scaling_plots(arch, simdir, outdir, figdir):
    """Create strong scaling plots.
    """

    gpus = [1, 2, 4, 8]
    logdata = []
    h5data = []

    # expand arch
    nl, nn, nbs = arch

    for ngpus in gpus:

        cname = f"nl{nl}-nn{nn}-npts{nbs//ngpus}-ngpus{ngpus}"
        logdata.append(log_parser(simdir.joinpath(cname)))

        with h5open(outdir.joinpath(f"{cname}-raw.h5"), "r") as h5file:
            h5data.append({
                "steps": h5file["walltime/steps"][...],
                "elapsedtimes": h5file["walltime/elapsedtimes"][...],
                "l2norms": h5file["walltime/l2norms/u/40.0"][...]
            })

    styles_def = (
        cycler("ls", ["solid", "dotted", "dashed", "dashdot"]*5) +
        cycler("color", pyplot.get_cmap("tab10").colors*2) +
        cycler("lw", [1.5]*20) +
        cycler("alpha", [0.75]*20)
    )

    styles = styles_def()

    fig = pyplot.figure(figsize=(6.5, 3.3))
    fig.suptitle("Aggregated loss and elapsed time v.s. iteration")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.83, 0.17])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss or $L_2$ error of $u$")
    ax.grid()
    tax = ax.twinx()
    tax.set_ylabel("Run time (hours)")

    l1s, l2s, l3s = [], [], []
    for ilogdata, ih5data in zip(logdata, h5data):
        l1s.append(ax.semilogy(ilogdata.index, ilogdata.loss.rolling(window=30).min(), **next(styles))[0])
        l2s.append(ax.semilogy(ih5data["steps"], ih5data["l2norms"], **next(styles))[0])
        l3s.append(tax.plot(ilogdata.index, ilogdata["time elapsed"], **next(styles))[0])

    # put legend in the second and invicid axes
    labels = [f"{ngpus} GPUs" for ngpus in gpus]
    lax = fig.add_subplot(gs[0, 1])
    lgds = [
        lax.legend(l1s, labels, title="Loss", loc="upper right", bbox_to_anchor=(1.0, 1.05)),
        lax.legend(l2s, labels, title=r"$L_2$ err. of $u$", loc="center right", bbox_to_anchor=(1.0, 0.5)),
        lax.legend(l3s, labels, title="Run time", loc="lower right", bbox_to_anchor=(1.0, -0.05))
    ]
    lax.add_artist(lgds[0])
    lax.add_artist(lgds[1])
    lax.add_artist(lgds[2])
    lax.axis("off")

    # save
    figdir.joinpath("scaling-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("scaling-tests", f"nl{nl}-nn{nn}-npts{nbs}-strong-scaling.png"))

    return 0


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _outdir = _projdir.joinpath("outputs", "exp-sum-scaling")
    _simdir = _projdir.joinpath("exp-sum-scaling")

    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    create_strong_scaling_plots((2, 32, 65536), _simdir, _outdir, _figdir)
    create_strong_scaling_plots((3, 128, 65536), _simdir, _outdir, _figdir)
