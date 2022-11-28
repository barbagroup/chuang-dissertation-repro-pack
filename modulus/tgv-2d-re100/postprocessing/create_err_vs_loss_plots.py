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
import re
import sys
from h5py import File as h5open
from matplotlib import pyplot

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def plot_error_loss_scatter(figdir):
    """Plot error vs loss.
    """

    projdir = pathlib.Path(__file__).resolve().parents[1]
    outdir = projdir.joinpath("outputs")
    nls = [1, 2, 3]
    nns = [16, 32, 64, 128, 256]

    loss = []
    errs = []
    ccodes = []
    cmap = {(nl, nn): i for i, (nl, nn) in enumerate(itertools.product(nls, nns))}
    invcmap = {i: (nl, nn) for i, (nl, nn) in enumerate(itertools.product(nls, nns))}

    def _get_data(_prefix, _cmap, _loss, _errs, _ccodes):
        _cases = projdir.joinpath(_prefix).glob("nl*-nn*-npts*")
        for _case, field in itertools.product(_cases, ["u", "v"]):
            _nl, _nn = re.search(r"nl(\d)-nn(\d+)-npts.*", _case.name).groups()

            _loss.append(log_parser(_case).iloc[-1]["loss"])
            with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-raw.h5")) as h5file:
                _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
            _ccodes.append(_cmap[(int(_nl), int(_nn))])

            if outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5").is_file():
                _loss.append(log_parser(_case).iloc[-1]["loss"])
                with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5")) as h5file:
                    _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
                _ccodes.append(_cmap[(int(_nl), int(_nn))])

        return _loss, _errs, _ccodes

    loss, errs, ccodes = _get_data("base-cases", cmap, loss, errs, ccodes)
    loss, errs, ccodes = _get_data("exp-sum-scaling", cmap, loss, errs, ccodes)
    loss, errs, ccodes = _get_data("exp-annealing", cmap, loss, errs, ccodes)
    loss, errs, ccodes = _get_data("cyclic-sum", cmap, loss, errs, ccodes)
    loss, errs, ccodes = _get_data("cyclic-annealing", cmap, loss, errs, ccodes)
    loss, errs, ccodes = _get_data("ncg-sum", cmap, loss, errs, ccodes)

    loss = [val**0.5 for val in loss]

    fig = pyplot.figure(figsize=(6.5, 4))
    fig.suptitle(r"$L_{2,sp-t}$ error v.s. aggregated loss")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(r"$\sqrt{\mathrm{Aggregated\ loss}}$")
    ax.set_xlim(1e-3, 2e0)
    ax.set_xscale("log")
    ax.set_ylabel(r"$L_{2,sp-t}$ error")
    ax.set_ylim(7e-3, 1e0)
    ax.set_yscale("log")
    ax.set_aspect("equal")

    scatter = ax.scatter(loss, errs, c=ccodes, s=75, cmap="tab20", alpha=0.6, marker="o")

    # legend
    handles, labels = scatter.legend_elements(prop="colors", num=None, fmt="{x:d}", func=lambda x: x.astype(int))
    labels = [invcmap[int(key)] for key in labels]
    ax.legend(handles, labels, loc=0, title=r"$(N_l, N_n)$", ncol=2)

    figdir.joinpath("err-vs-loss").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-loss", "err-loss.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    plot_error_loss_scatter(_figdir)
