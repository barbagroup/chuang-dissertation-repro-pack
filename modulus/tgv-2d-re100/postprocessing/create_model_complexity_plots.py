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
import numpy
import pandas
from h5py import File as h5open
from matplotlib import pyplot


# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def dof_calculator(nl, nn, dim=2, unsteady=True, periodic=True):
    """Degree of freedom calculator.
    """
    ninp = dim + int(unsteady) + 2 * int(periodic)
    dof = ninp * nn + 2 * nn
    dof += (nn**2 + 2 * nn) * (nl - 1)
    dof += (nn + 1) * (dim + 1)
    return dof


def create_err_arch_boxplot(outdir, figdir, field, nls, nns, nbss):
    """plot_err_arch_boxplot
    """
    data = {"nl": [], "nn": [], "nbs": [], "l2norm": []}
    for nl, nn, nbs in itertools.product(nls, nns, nbss):
        data["nl"].append(nl)
        data["nn"].append(nn)
        data["nbs"].append(nbs)
        with h5open(outdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            data["l2norm"].append(float(h5file[f"sterrs/{field}/l2norm"][...]))

    data = pandas.DataFrame(data)
    data = data.pivot(index="nbs", columns=["nl", "nn"], values="l2norm")
    data = data[data.mean().sort_values(ascending=False).index]

    fig = pyplot.figure()
    fig.suptitle(rf"Error distribution across network architectures, ${field}$")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.boxplot(
        data.values, labels=data.columns, showmeans=True,
        meanprops={"marker": ".", "mfc": "k", "mec": "k"},
        medianprops={"ls": "none"},
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel(r"$(N_l, N_n)$")
    ax.set_ylabel(rf"$L_2$ error of ${field}$")
    ax.set_yscale("log")

    figdir.joinpath("err-vs-model-complexity").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-model-complexity", f"err-arch-boxplot-{field}.png"))


def create_err_dof_boxplot(outdir, figdir, field, nls, nns, nbss):
    """plot_dof_err
    """
    data = {"dofs": [], "l2norm": [], "nbs": []}
    for nl, nn, nbs in itertools.product(nls, nns, nbss):
        data["nbs"].append(nbs)
        data["dofs"].append(dof_calculator(nl, nn, 2, True, True))
        with h5open(outdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            data["l2norm"].append(float(h5file[f"sterrs/{field}/l2norm"][...]))

    data = pandas.DataFrame(data)
    data = data.pivot(index="nbs", columns=["dofs"], values="l2norm")
    data = data.sort_index(axis=1)

    # box widths on the plot with log x axis
    width = lambda p, w: 10**(numpy.log10(p)+w/2.)-10**(numpy.log10(p)-w/2.)  # noqa: E731

    fig = pyplot.figure()
    fig.suptitle(rf"Error distribution v.s. degree of freedom, ${field}$")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.boxplot(
        data.values, labels=data.columns, positions=data.columns,
        showmeans=True, widths=width(data.columns, 0.1),
        meanprops={"marker": ".", "mfc": "k", "mec": "k"},
        medianprops={"ls": "none"},
    )

    ax.set_xlabel("Degree of freedom")
    ax.set_xscale("log")
    ax.set_ylabel(rf"$L_2$ error of ${field}$")
    ax.set_yscale("log")

    figdir.joinpath("err-vs-model-complexity").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-model-complexity", f"err-dof-boxplot-{field}.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _outdir = _projdir.joinpath("outputs", "base-cases")
    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    _nls = [1, 2, 3]
    _nns = [16, 32, 64, 128, 256]
    _nbss = [2**i for i in range(10, 17)]

    create_err_arch_boxplot(_outdir, _figdir, "u", _nls, _nns, _nbss)
    create_err_arch_boxplot(_outdir, _figdir, "v", _nls, _nns, _nbss)
    create_err_dof_boxplot(_outdir, _figdir, "u", _nls, _nns, _nbss)
    create_err_dof_boxplot(_outdir, _figdir, "v", _nls, _nns, _nbss)
