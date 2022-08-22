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

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def _plot_loss_err_vs_steps(arch, alogdata, slogdata, ah5data, sh5data, figdir):
    """internal func.
    """

    nl, nn, nbs = arch

    fig = pyplot.figure(figsize=(6.5, 2.5))
    fig.suptitle(r"TGV 2D, $Re=100$, training hist., annealing loss aggregation")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.83, 0.17])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(rf"$(N_l, N_n, N_{{bs}})=({nl}, {nn}, {nbs})$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss or $L_2$ error of $u$")
    ax.grid()

    lloss = [
        ax.semilogy(slogdata.index, slogdata.loss.rolling(window=10).min(), ls="-", alpha=0.8)[0],
        ax.semilogy(alogdata.index, alogdata.loss.rolling(window=10).min(), ls="-.", alpha=0.8)[0]
    ]
    lu0 = [
        ax.semilogy(sh5data.steps, sh5data.u0, ls="-", alpha=0.8)[0],
        ax.semilogy(ah5data.steps, ah5data.u0, ls="-.", alpha=0.8)[0],
    ]
    lu40 = [
        ax.semilogy(sh5data.steps, sh5data.u40, ls="-", alpha=0.8)[0],
        ax.semilogy(ah5data.steps, ah5data.u40, ls="-.", alpha=0.8)[0],
    ]

    # put legend in the second and invicid axes
    labels = ["Naive sum", "Annealing"]
    lax = fig.add_subplot(gs[0, 1])
    lgds = [
        lax.legend(lloss, labels, title="Loss", loc="upper right", bbox_to_anchor=(1.0, 1.05)),
        lax.legend(lu0, labels, title=r"$L_2$ err., $u$, $t=0$", loc="center right", bbox_to_anchor=(1.0, 0.5)),
        lax.legend(lu40, labels, title=r"$L_2$ err., $u$, $t=40$", loc="lower right", bbox_to_anchor=(1.0, -0.05))
    ]
    lax.add_artist(lgds[0])
    lax.add_artist(lgds[1])
    lax.add_artist(lgds[2])
    lax.axis("off")

    # save
    figdir.joinpath("annealing-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("annealing-tests", f"nl{nl}-nn{nn}-npts{nbs}-steps.png"))


def _plot_loss_err_vs_time(arch, alogdata, slogdata, ah5data, sh5data, figdir):
    """internal func.
    """

    nl, nn, nbs = arch

    fig = pyplot.figure(figsize=(6.5, 2.5))
    fig.suptitle(r"TGV 2D, $Re=100$, training hist., annealing loss aggregation")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.83, 0.17])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(rf"$(N_l, N_n, N_{{bs}})=({nl}, {nn}, {nbs})$")
    ax.set_xlabel("Run time (hours)")
    ax.set_ylabel(r"Loss or $L_2$ error of $u$")
    ax.grid()

    lloss = [
        ax.semilogy(slogdata["time elapsed"], slogdata.loss.rolling(window=10).min(), ls="-", alpha=0.8)[0],
        ax.semilogy(alogdata["time elapsed"], alogdata.loss.rolling(window=10).min(), ls="-.", alpha=0.8)[0]
    ]
    lu0 = [
        ax.semilogy(sh5data.elapsedtimes, sh5data.u0, ls="-", alpha=0.8)[0],
        ax.semilogy(ah5data.elapsedtimes, ah5data.u0, ls="-.", alpha=0.8)[0],
    ]
    lu40 = [
        ax.semilogy(sh5data.elapsedtimes, sh5data.u40, ls="-", alpha=0.8)[0],
        ax.semilogy(ah5data.elapsedtimes, ah5data.u40, ls="-.", alpha=0.8)[0],
    ]

    # put legend in the second and invicid axes
    labels = ["Naive sum", "Annealing"]
    lax = fig.add_subplot(gs[0, 1])
    lgds = [
        lax.legend(lloss, labels, title="Loss", loc="upper right", bbox_to_anchor=(1.0, 1.05)),
        lax.legend(lu0, labels, title=r"$L_2$ err., $u$, $t=0$", loc="center right", bbox_to_anchor=(1.0, 0.5)),
        lax.legend(lu40, labels, title=r"$L_2$ err., $u$, $t=40$", loc="lower right", bbox_to_anchor=(1.0, -0.05))
    ]
    lax.add_artist(lgds[0])
    lax.add_artist(lgds[1])
    lax.add_artist(lgds[2])
    lax.axis("off")

    # save
    figdir.joinpath("annealing-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("annealing-tests", f"nl{nl}-nn{nn}-npts{nbs}-walltimes.png"))


def _plot_final_spatial_temporal_err(archs, aoutdir, soutdir, figdir):
    """internal use
    """

    adatau, adatav = [], []
    sdatau, sdatav = [], []
    for arch in archs:
        nl, nn, nbs = arch
        with h5open(aoutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            adatau.append(float(h5file["sterrs/u/l2norm"][...]))
            adatav.append(float(h5file["sterrs/v/l2norm"][...]))

        with h5open(soutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            sdatau.append(float(h5file["sterrs/u/l2norm"][...]))
            sdatav.append(float(h5file["sterrs/v/l2norm"][...]))

    fig = pyplot.figure(figsize=(6.5, 3))
    fig.suptitle(r"Naive sum and annealing loss: spatial-temporal error comparisons")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 0.1])

    axu = fig.add_subplot(gs[0, 0])
    axu.set_title(r"$u$")
    axu.set_xlabel(r"Architecture ($(N_l, N_n, N_{bs})$)")
    axu.set_ylabel(r"$L_2$ error")
    axu.set_xticks(range(len(archs)), [rf"$({nl}, {nn}, {nbs})$" for nl, nn, nbs in archs], rotation=-20)
    axu.set_yscale("log")
    bar1 = axu.bar([inp - 0.125 for inp in range(len(archs))], sdatau, width=0.25)
    bar2 = axu.bar([inp + 0.125 for inp in range(len(archs))], adatau, width=0.25)

    axv = fig.add_subplot(gs[0, 1], sharey=axu)
    axv.set_title(r"$v$")
    axv.set_xlabel(r"Architecture ($(N_l, N_n, N_{bs})$)")
    axv.set_ylabel(r"$L_2$ error")
    axv.set_xticks(range(len(archs)), [rf"$({nl}, {nn}, {nbs})$" for nl, nn, nbs in archs], rotation=-20)
    axv.set_yscale("log")
    bar1 = axv.bar([inp - 0.125 for inp in range(len(archs))], sdatav, width=0.25)
    bar2 = axv.bar([inp + 0.125 for inp in range(len(archs))], adatav, width=0.25)

    lax = fig.add_subplot(gs[1, :])
    lax.legend([bar1, bar2], ["Naive sum", "Annealing"], loc="center", ncol=2, columnspacing=5)
    lax.axis("off")

    # save
    figdir.joinpath("annealing-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("annealing-tests", "final-spatial-temporal-errors.png"))


def create_annealing_test_plots(asimdir, aoutdir, ssimdir, soutdir, figdir, arch):
    """create_annealing_test_plots
    """
    nl, nn, nbs = arch
    alogdata = log_parser(asimdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}"))
    slogdata = log_parser(ssimdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}"))

    with h5open(aoutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        ah5data = pandas.DataFrame({
            "steps": h5file["walltime/steps"][...],
            "elapsedtimes": h5file["walltime/elapsedtimes"][...],
            "u0": h5file["walltime/l2norms/u/0.0"][...],
            "u40": h5file["walltime/l2norms/u/40.0"][...],
            "u80": h5file["walltime/l2norms/u/80.0"][...],
        })

    with h5open(soutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        sh5data = pandas.DataFrame({
            "steps": h5file["walltime/steps"][...],
            "elapsedtimes": h5file["walltime/elapsedtimes"][...],
            "u0": h5file["walltime/l2norms/u/0.0"][...],
            "u40": h5file["walltime/l2norms/u/40.0"][...],
            "u80": h5file["walltime/l2norms/u/80.0"][...],
        })

    _plot_loss_err_vs_steps(arch, alogdata, slogdata, ah5data, sh5data, figdir)
    _plot_loss_err_vs_time(arch, alogdata, slogdata, ah5data, sh5data, figdir)


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _asimdir = _projdir.joinpath("exp-annealing")
    _aoutdir = _projdir.joinpath("outputs", "exp-annealing")
    _ssimdir = _projdir.joinpath("base-cases")
    _soutdir = _projdir.joinpath("outputs", "base-cases")

    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    _archs = ((1, 16, 8192), (2, 32, 8192), (3, 128, 8192))

    for _arch in _archs:
        create_annealing_test_plots(_asimdir, _aoutdir, _ssimdir, _soutdir, _figdir, _arch)

    _plot_final_spatial_temporal_err(_archs, _aoutdir, _soutdir, _figdir)
