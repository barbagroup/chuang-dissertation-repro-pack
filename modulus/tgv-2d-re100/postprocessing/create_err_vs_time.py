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
from h5py import File as h5open
from matplotlib import pyplot


# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def plot_error_loss_scatter(figdir):
    """Plot error vs loss.
    """

    projdir = pathlib.Path(__file__).resolve().parents[1]
    outdir = projdir.joinpath("outputs")
    nls = [1, 2, 3]
    nns = [16, 32, 64, 128, 256]
    cmap = {(nl, nn): i for i, (nl, nn) in enumerate(itertools.product(nls, nns))}
    invcmap = {i: (nl, nn) for i, (nl, nn) in enumerate(itertools.product(nls, nns))}

    def _get_data(_prefix, _cmap, _times, _errs, _ccodes):
        _cases = projdir.joinpath(_prefix).glob("nl*-nn*-npts*")
        for _case, field in itertools.product(_cases, ["u", "v"]):
            _nl, _nn = re.search(r"nl(\d)-nn(\d+)-npts.*", _case.name).groups()

            with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-raw.h5")) as h5file:
                _times.append(float(h5file["walltime/elapsedtimes"][...][-1]))
                _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
            _ccodes.append(_cmap[(int(_nl), int(_nn))])

            if outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5").is_file():
                with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5")) as h5file:
                    _times.append(float(h5file["walltime/elapsedtimes"][...][-1]))
                    _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
                _ccodes.append(_cmap[(int(_nl), int(_nn))])

        return _times, _errs, _ccodes

    times, errs, ccodes = [], [], []
    times, errs, ccodes = _get_data("base-cases", cmap, times, errs, ccodes)
    times, errs, ccodes = _get_data("exp-annealing", cmap, times, errs, ccodes)
    times, errs, ccodes = _get_data("cyclic-sum", cmap, times, errs, ccodes)
    times, errs, ccodes = _get_data("cyclic-annealing", cmap, times, errs, ccodes)

    fig = pyplot.figure(figsize=(6.5, 4))
    fig.suptitle(r"$L_{2,sp-t}$ error v.s. run time")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel("Run time (hours)")
    ax.set_ylabel(r"$L_{2,sp-t}$ error")
    ax.set_yscale("log")

    scatter = ax.scatter(times, errs, c=ccodes, s=75, cmap="tab20", alpha=0.6, marker="o")

    # legend
    handles, labels = scatter.legend_elements(prop="colors", num=None, fmt="{x:d}", func=lambda x: x.astype(int))
    labels = [invcmap[int(key)] for key in labels]
    ax.legend(handles, labels, loc=0, title=r"$(N_l, N_n)$", ncol=2)

    figdir.joinpath("err-vs-time").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-time", "err-time.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    plot_error_loss_scatter(_figdir)
