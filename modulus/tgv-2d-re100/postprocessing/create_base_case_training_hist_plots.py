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
from matplotlib import pyplot

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def create_base_case_training_hist(simdir, outdir, sterrdir, figdir, arch, ws=10):
    """Plot figures related to training loss and spatial-temporal errors.
    """

    nl, nn, nbs = arch
    data = log_parser(simdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}"))

    with h5open(outdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        errsteps = h5file["walltime/steps"][...]
        errtimes = h5file["walltime/elapsedtimes"][...]
        err = [
            h5file["walltime/l2norms/u/0.0"][...],
            h5file["walltime/l2norms/u/40.0"][...],
            h5file["walltime/l2norms/u/80.0"][...],
        ]

    with h5open(sterrdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        sterr = h5file["walltime/l2norms/u"][...]

    # plot according to optimizer type
    fig = pyplot.figure(figsize=(6.5, 3.3))
    fig.suptitle(rf"2D TGV, $Re=100$, $(N_l, N_n, N_{{bs}})=({nl}, {nn}, {nbs})$")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 0.05])

    # axes for loss/err against steps
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Training loss and solution errors v.s. iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss or $L_2$ error")
    ax.grid()

    # axes for timeline (run time)
    axerr = fig.add_subplot(gs[1, 0])
    axerr.semilogy(errtimes, err[0], lw=0)  # dummy line to set x-axis limits
    axerr.spines["top"].set_visible(False)
    axerr.spines["left"].set_visible(False)
    axerr.spines["right"].set_visible(False)
    axerr.spines["bottom"].set_visible(True)
    axerr.set_xlabel("Run time (hours)")
    axerr.xaxis.set_label_position("bottom")
    axerr.xaxis.set_ticks_position("bottom")
    axerr.xaxis.set_ticks_position("bottom")
    axerr.yaxis.set_visible(False)

    l1, = ax.semilogy(data.index, data.loss.rolling(window=ws).min(), lw=1.5, label="Aggregated loss")
    l2, = ax.semilogy(errsteps, sterr, lw=2, ls="--", label=r"Overall spatial-temporal error")

    l4, = ax.semilogy(errsteps, err[0], lw=1.5, alpha=0.8, ls="--", label=r"$t=0$")
    l5, = ax.semilogy(errsteps, err[1], lw=1.5, alpha=0.8, ls="--", label=r"$t=40$")
    l6, = ax.semilogy(errsteps, err[2], lw=1.5, alpha=0.8, ls="--", label=r"$t=80$")

    # customized  legend locations
    if arch == (2, 32, 16384):
        lloc1 = (0.55, 0.6)
        lloc2 = (0.99, 0.6)
    if arch == (2, 32, 65536):
        lloc1 = (0.55, 0.6)
        lloc2 = (0.99, 0.6)
    else:
        lloc1 = (0.55, 0.99)
        lloc2 = (0.99, 0.99)

    # legends
    lgds = [
        ax.legend(handles=[l1, l2], loc="upper right", bbox_to_anchor=lloc1),
        ax.legend(handles=[l4, l5, l6], title=r"Spatial error of $u$", loc="upper right", bbox_to_anchor=lloc2, ncol=3)
    ]
    ax.add_artist(lgds[0])
    ax.add_artist(lgds[1])

    # save
    figdir.joinpath("training-hist").mkdir(parents=True, exist_ok=True)
    print(figdir.joinpath("training-hist", f"nl{nl}-nn{nn}-npts{nbs}.png"))
    fig.savefig(figdir.joinpath("training-hist", f"nl{nl}-nn{nn}-npts{nbs}.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).parents[1]
    _simdir = _projdir.joinpath("base-cases")
    _outdir = _projdir.joinpath("outputs", "base-cases")
    _sterrdir = _projdir.joinpath("old.nogit", "outputs.sterrs", "base-cases")

    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    create_base_case_training_hist(_simdir, _outdir, _sterrdir, _figdir, (1, 32, 16384))
    create_base_case_training_hist(_simdir, _outdir, _sterrdir, _figdir, (2, 32, 65536))
    create_base_case_training_hist(_simdir, _outdir, _sterrdir, _figdir, (3, 256, 4096))
