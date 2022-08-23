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
import pandas
from h5py import File as h5open
from matplotlib import pyplot
from cycler import cycler

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def _plot_loss_err(archs, nlogdata, alogdata, nh5data, ah5data, figdir):
    """internal func.
    """

    stylegen = cycler("color", pyplot.get_cmap("tab10").colors)

    fig = pyplot.figure(figsize=(6.5, 9))
    fig.suptitle(r"TGV 2D, $Re=100$, training hist., nonlinear-CG optimizer")
    gs = fig.add_gridspec(4, 2, height_ratios=[0.2, 1., 1., 1.])

    for i, arch in enumerate(archs):
        styles = stylegen()  # restart the style generator to unify styles
        nl, nn, nbs = arch

        ax = fig.add_subplot(gs[i+1, 0])
        ax.set_title(rf"$({nl}, {nn}, {nbs})$, err. v.s. iteration")
        ax.set_xlim(199800, 200200)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Loss or $L_2$ error of $u$")
        ax.grid()

        lloss = [
            ax.semilogy(
                alogdata[i].index, alogdata[i].loss.rolling(window=10).min(),
                ls="-", alpha=0.8, **next(styles))[0],
            ax.semilogy(
                nlogdata[i].index, nlogdata[i].loss.rolling(window=10).min(),
                ls="-.", alpha=0.8, **next(styles))[0]
        ]
        lu0 = [
            ax.semilogy(ah5data[i].steps, ah5data[i].u0, ls="-", alpha=0.8, **next(styles))[0],
            ax.semilogy(nh5data[i].steps, nh5data[i].u0, ls="-.", alpha=0.8, **next(styles))[0],
        ]
        lu40 = [
            ax.semilogy(ah5data[i].steps, ah5data[i].u40, ls="-", alpha=0.8, **next(styles))[0],
            ax.semilogy(nh5data[i].steps, nh5data[i].u40, ls="-.", alpha=0.8, **next(styles))[0],
        ]

    for i, arch in enumerate(archs):
        styles = stylegen()  # restart the style generator to unify styles
        nl, nn, nbs = arch

        ax = fig.add_subplot(gs[i+1, 1])
        ax.set_title(rf"$({nl}, {nn}, {nbs})$, err. v.s. run time")
        ax.set_xlim(nlogdata[i].loc[199800, "time elapsed"], nlogdata[i].loc[200200, "time elapsed"])
        ax.set_xlabel("Run time (hours)")
        ax.set_ylabel(r"Loss or $L_2$ error of $u$")
        ax.grid()

        aloss = alogdata[i].loss.rolling(window=10).min()
        nloss = list(nlogdata[i].loss.loc[:200000].rolling(window=10).min()) + list(nlogdata[i].loss.loc[200001:])
        lloss = [
            ax.semilogy(alogdata[i]["time elapsed"], aloss, ls="-", alpha=0.8, **next(styles))[0],
            ax.semilogy(nlogdata[i]["time elapsed"], nloss, ls="-.", alpha=0.8, **next(styles))[0]
        ]
        lu0 = [
            ax.semilogy(ah5data[i].elapsedtimes, ah5data[i].u0, ls="-", alpha=0.8, **next(styles))[0],
            ax.semilogy(nh5data[i].elapsedtimes, nh5data[i].u0, ls="-.", alpha=0.8, **next(styles))[0],
        ]
        lu40 = [
            ax.semilogy(ah5data[i].elapsedtimes, ah5data[i].u40, ls="-", alpha=0.8, **next(styles))[0],
            ax.semilogy(nh5data[i].elapsedtimes, nh5data[i].u40, ls="-.", alpha=0.8, **next(styles))[0],
        ]

    # put legend in the second and invicid axes
    lax = fig.add_subplot(gs[0, :])
    labels = ["Adam", "CG"]
    kwargs = {"ncol": 2, "columnspacing": 2}
    lgds = [
        lax.legend(lloss, labels, title="Loss", loc=2, bbox_to_anchor=(0.0, 1.0), **kwargs),
        lax.legend(lu0, labels, title=r"$L_2$ err., $u$, $t=0$", loc=9, bbox_to_anchor=(0.5, 1.0), **kwargs),
        lax.legend(lu40, labels, title=r"$L_2$ err., $u$, $t=40$", loc=1, bbox_to_anchor=(1.0, 1.0), **kwargs)
    ]

    lax.add_artist(lgds[0])
    lax.add_artist(lgds[1])
    lax.add_artist(lgds[2])
    lax.axis("off")

    # save
    figdir.joinpath("ncg-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("ncg-tests", "training-hist.png"))


def _plot_final_spatial_temporal_err(archs, noutdir, aoutdir, figdir):
    """internal use
    """

    ndatau, ndatav = [], []
    adatau, adatav = [], []
    for arch in archs:
        nl, nn, nbs = arch
        with h5open(noutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            ndatau.append(float(h5file["sterrs/u/l2norm"][...]))
            ndatav.append(float(h5file["sterrs/v/l2norm"][...]))

        with h5open(aoutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            adatau.append(float(h5file["sterrs/u/l2norm"][...]))
            adatav.append(float(h5file["sterrs/v/l2norm"][...]))

    fig = pyplot.figure(figsize=(6.5, 3))
    fig.suptitle(r"Nonlinear-CG: spatial-temporal error comparisons")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 0.1])

    axu = fig.add_subplot(gs[0, 0])
    axu.set_title(r"$u$")
    axu.set_xlabel(r"Architecture ($(N_l, N_n, N_{bs})$)")
    axu.set_ylabel(r"$L_2$ error")
    axu.set_xticks(range(len(archs)), [rf"$({nl}, {nn}, {nbs})$" for nl, nn, nbs in archs], rotation=-20)
    axu.set_yscale("log")
    axu.set_ylim(1e-2, 5e-1)
    bar1 = axu.bar([inp - 0.125 for inp in range(len(archs))], adatau, width=0.25)
    bar2 = axu.bar([inp + 0.125 for inp in range(len(archs))], ndatau, width=0.25)

    axv = fig.add_subplot(gs[0, 1], sharey=axu)
    axv.set_title(r"$v$")
    axv.set_xlabel(r"Architecture ($(N_l, N_n, N_{bs})$)")
    axv.set_ylabel(r"$L_2$ error")
    axv.set_xticks(range(len(archs)), [rf"$({nl}, {nn}, {nbs})$" for nl, nn, nbs in archs], rotation=-20)
    axv.set_yscale("log")
    bar1 = axv.bar([inp - 0.125 for inp in range(len(archs))], adatav, width=0.25)
    bar2 = axv.bar([inp + 0.125 for inp in range(len(archs))], ndatav, width=0.25)

    lax = fig.add_subplot(gs[1, :])
    lax.legend([bar1, bar2], ["Adam", "CG"], loc="center", ncol=3, columnspacing=5)
    lax.axis("off")

    # save
    figdir.joinpath("ncg-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("ncg-tests", "final-spatial-temporal-errors.png"))


def create_annealing_test_plots(nsimdir, noutdir, asimdir, aoutdir, figdir, archs):
    """create_annealing_test_plots
    """

    nlogdata = []
    alogdata = []
    nh5data = []
    ah5data = []

    for arch in archs:
        nl, nn, nbs = arch
        nlogdata.append(log_parser(nsimdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}")))
        alogdata.append(log_parser(asimdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}")))

        with h5open(noutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            nh5data.append(pandas.DataFrame({
                "steps": h5file["walltime/steps"][...],
                "elapsedtimes": h5file["walltime/elapsedtimes"][...],
                "u0": h5file["walltime/l2norms/u/0.0"][...],
                "u40": h5file["walltime/l2norms/u/40.0"][...],
                "u80": h5file["walltime/l2norms/u/80.0"][...],
            }))

        with h5open(aoutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            ah5data.append(pandas.DataFrame({
                "steps": h5file["walltime/steps"][...],
                "elapsedtimes": h5file["walltime/elapsedtimes"][...],
                "u0": h5file["walltime/l2norms/u/0.0"][...],
                "u40": h5file["walltime/l2norms/u/40.0"][...],
                "u80": h5file["walltime/l2norms/u/80.0"][...],
            }))

    _plot_loss_err(archs, nlogdata, alogdata, nh5data, ah5data, figdir)


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _nsimdir = _projdir.joinpath("ncg-sum")
    _noutdir = _projdir.joinpath("outputs", "ncg-sum")
    _asimdir = _projdir.joinpath("base-cases")
    _aoutdir = _projdir.joinpath("outputs", "base-cases")

    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    _archs = ((1, 16, 8192), (2, 32, 8192), (3, 128, 8192))
    create_annealing_test_plots(_nsimdir, _noutdir, _asimdir, _aoutdir, _figdir, _archs)
    _plot_final_spatial_temporal_err(_archs, _noutdir, _aoutdir, _figdir)
