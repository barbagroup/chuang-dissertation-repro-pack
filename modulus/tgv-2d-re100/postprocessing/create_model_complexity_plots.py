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


def create_err_arch_boxplot(figdir):
    """plot_err_arch_boxplot
    """
    projdir = pathlib.Path(__file__).resolve().parents[1]
    outdir = projdir.joinpath("outputs")

    def _get_data(_prefix, _nls, _nns, _errs):
        _cases = projdir.joinpath(_prefix).glob("nl*-nn*-npts*")
        for _case, field in itertools.product(_cases, ["u", "v"]):
            _nl, _nn = re.search(r"nl(\d)-nn(\d+)-npts.*", _case.name).groups()

            with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-raw.h5")) as h5file:
                _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
            _nls.append(int(_nl))
            _nns.append(int(_nn))

            if outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5").is_file():
                with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5")) as h5file:
                    _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
                _nls.append(int(_nl))
                _nns.append(int(_nn))

        return _nls, _nns, _errs

    nls, nns, errs = [], [], []
    nls, nns, errs = _get_data("base-cases", nls, nns, errs)
    nls, nns, errs = _get_data("exp-sum-scaling", nls, nns, errs)
    nls, nns, errs = _get_data("exp-annealing", nls, nns, errs)
    nls, nns, errs = _get_data("cyclic-sum", nls, nns, errs)
    nls, nns, errs = _get_data("cyclic-annealing", nls, nns, errs)
    nls, nns, errs = _get_data("ncg-sum", nls, nns, errs)

    data = {"nl": nls, "nn": nns, "l2norm": errs}
    data = pandas.DataFrame(data)
    data = data.pivot(index=None, columns=["nl", "nn"], values="l2norm")
    data = data[data.mean().sort_values(ascending=False).index]  # automatically skips NaN

    fig = pyplot.figure(figsize=(6.5, 4))
    fig.suptitle(r"Error distribution v.s. network architectures")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax = data.boxplot(
        ax=ax, rot=45, showmeans=True, meanprops={"marker": ".", "mfc": "k", "mec": "k"},
        grid=False, medianprops={"ls": "none"}, boxprops={"color": "k"}, whiskerprops={"color": "k"}
    )
    ax.set_xlabel(r"$(N_l, N_n)$")
    ax.set_ylabel(r"$L_2$ error")
    ax.set_yscale("log")

    figdir.joinpath("err-vs-model-complexity").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-model-complexity", "err-arch-boxplot.png"))


def create_err_dof_boxplot(figdir):
    """plot_dof_err
    """
    projdir = pathlib.Path(__file__).resolve().parents[1]
    outdir = projdir.joinpath("outputs")

    def _get_data(_prefix, _dofs, _errs):
        _cases = projdir.joinpath(_prefix).glob("nl*-nn*-npts*")
        for _case, field in itertools.product(_cases, ["u", "v"]):
            _nl, _nn = re.search(r"nl(\d)-nn(\d+)-npts.*", _case.name).groups()

            with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-raw.h5")) as h5file:
                _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
            _dofs.append(dof_calculator(int(_nl), int(_nn), 2, True, True))

            if outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5").is_file():
                with h5open(outdir.joinpath(_case.relative_to(projdir)).with_name(_case.name+"-swa.h5")) as h5file:
                    _errs.append(float(h5file[f"sterrs/{field}/l2norm"][...]))
                _dofs.append(dof_calculator(int(_nl), int(_nn), 2, True, True))

        return _dofs, _errs

    dofs, errs = [], []
    dofs, errs = _get_data("base-cases", dofs, errs)
    dofs, errs = _get_data("exp-sum-scaling", dofs, errs)
    dofs, errs = _get_data("exp-annealing", dofs, errs)
    dofs, errs = _get_data("cyclic-sum", dofs, errs)
    dofs, errs = _get_data("cyclic-annealing", dofs, errs)
    dofs, errs = _get_data("ncg-sum", dofs, errs)

    data = {"dofs": dofs, "l2norm": errs}
    data = pandas.DataFrame(data)
    data = data.pivot(index=None, columns=["dofs"], values="l2norm")
    data = data.sort_index(axis=1)

    # box widths on the plot with log x axis
    width = lambda p, w: 10**(numpy.log10(p)+w/2.)-10**(numpy.log10(p)-w/2.)  # noqa: E731

    fig = pyplot.figure(figsize=(6.5, 4))
    fig.suptitle(r"Error distribution v.s. degree of freedom")
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax = data.boxplot(
        ax=ax, rot=45, positions=data.columns, widths=width(data.columns, 0.1),
        showmeans=True, grid=False, meanprops={"marker": ".", "mfc": "k", "mec": "k"},
        medianprops={"ls": "none"}, boxprops={"color": "k"}, whiskerprops={"color": "k"}
    )

    ax.set_xlabel("Degree of freedom")
    ax.set_xscale("log")
    ax.set_ylabel(r"$L_2$ error")
    ax.set_yscale("log")

    figdir.joinpath("err-vs-model-complexity").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("err-vs-model-complexity", "err-dof-boxplot.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    create_err_arch_boxplot(_figdir)
    create_err_dof_boxplot(_figdir)
