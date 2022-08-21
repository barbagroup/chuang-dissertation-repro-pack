#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
"""
import itertools
import pathlib
import sys
from h5py import File as h5open
from matplotlib import pyplot

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def plot_error_loss_scatter(simdir, outdir, figdir, field, nls, nns, nbss):
    """Plot error vs loss.
    """

    loss = []
    errs = []
    ccodes = []
    cmap = {(nl, nn): i for i, (nl, nn) in enumerate(itertools.product(nls, nns))}
    invcmap = {i: (nl, nn) for i, (nl, nn) in enumerate(itertools.product(nls, nns))}

    for (nl, nn, nbs) in itertools.product(nls, nns, nbss):
        loss.append(log_parser(simdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}")).iloc[-1]["loss"])

        with h5open(outdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))

        ccodes.append(cmap[(nl, nn)])

    fig = pyplot.figure()
    fig.suptitle(rf"Error v.s. aggregated loss, ${field}$")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel("Aggregate loss")
    ax.set_xscale("log")
    ax.set_ylabel(rf"$L_2$ error of ${field}$")
    ax.set_yscale("log")

    scatter = ax.scatter(loss, errs, c=ccodes, s=75, cmap="tab20", alpha=0.6, marker="o")

    # legend
    handles, labels = scatter.legend_elements(prop="colors", num=None, fmt="{x:d}", func=lambda x: x.astype(int))
    labels = [invcmap[int(key)] for key in labels]
    ax.legend(handles, labels, loc=0, title=r"$(N_l, N_n)$", ncol=2)

    figdir.joinpath("err-vs-loss").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-loss", f"err-loss-{field}.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _outdir = _projdir.joinpath("outputs", "base-cases")
    _simdir = _projdir.joinpath("base-cases")

    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    _nls = [1, 2, 3]
    _nns = [16, 32, 64, 128, 256]
    _nbss = [2**i for i in range(10, 17)]

    plot_error_loss_scatter(_simdir, _outdir, _figdir, "u", _nls, _nns, _nbss)
    plot_error_loss_scatter(_simdir, _outdir, _figdir, "v", _nls, _nns, _nbss)
