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
import numpy
import pandas
from h5py import File as h5open
from matplotlib import pyplot
from cycler import cycler

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402
from helpers.lr_simulator import cyclic_exp_range  # pylint: disable=import-error # noqa: E402
from helpers.lr_simulator import tf_exponential  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def _plot_lr(figdir):
    """plot lr
    """
    steps = numpy.arange(0, 400000, 100)
    clr = cyclic_exp_range(steps, 0.999989, 0.0015, 0.000015, 5000)
    elr = tf_exponential(steps, 0.95, 5000, 1e-3)

    fig = pyplot.figure(figsize=(3.75, 1.5))
    ax = fig.gca()
    ax.semilogy(steps, clr, lw=1.5, alpha=0.8, label="Cyclical LR")
    ax.semilogy(steps, elr, lw=1.5, alpha=0.8, label="Exponential LR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate")
    ax.grid()
    ax.legend(loc=0)

    figdir.joinpath("cyclic-swa-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("cyclic-swa-tests", "learning-rate-hist.png"))


def _plot_loss_err_vs_steps(arch, clogdata, elogdata, crh5data, eh5data, figdir):
    """internal func.
    """

    stylegen = cycler("color", pyplot.get_cmap("tab10").colors)
    styles = stylegen()

    nl, nn, nbs = arch

    fig = pyplot.figure(figsize=(3.75, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 0.22])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss or $L_2$-error of $u$")
    ax.grid()

    lloss = [
        ax.semilogy(
            elogdata.index, elogdata.loss.rolling(window=10).min(),
            ls="-", lw=1, alpha=0.9, **next(styles))[0],
        ax.semilogy(
            clogdata.index, clogdata.loss.rolling(window=10).min(),
            ls="-.", lw=1, alpha=0.9, **next(styles))[0]
    ]
    lu0 = [
        ax.semilogy(eh5data.steps, eh5data.u0, ls="-", lw=1, alpha=0.9, **next(styles))[0],
        ax.semilogy(crh5data.steps, crh5data.u0, ls="-.", lw=1, alpha=0.9, **next(styles))[0],
    ]
    lu40 = [
        ax.semilogy(eh5data.steps, eh5data.u40, ls="-", lw=1, alpha=0.9, **next(styles))[0],
        ax.semilogy(crh5data.steps, crh5data.u40, ls="-.", lw=1, alpha=0.9, **next(styles))[0],
    ]

    # wall time
    tax = ax.twinx()
    tax.set_ylabel("Run time (hours)")
    times = [
        tax.plot(elogdata.index, elogdata["time elapsed"], ls="-", lw=1, alpha=0.9, **next(styles))[0],
        tax.plot(clogdata.index, clogdata["time elapsed"], ls="-.", lw=1, alpha=0.9, **next(styles))[0]
    ]

    # put legend in the second and invicid axes
    lax = fig.add_subplot(gs[1, 0])
    labels = ["Exponential", "Cyclical"]
    lgds = [
        lax.legend(
            lloss, labels, title="Loss", ncol=2,
            loc="lower right", bbox_to_anchor=(0.49, 0.525)),
        lax.legend(
            times, labels, title="Run time", ncol=2,
            loc="lower left", bbox_to_anchor=(0.51, 0.525)),
    ]

    # labels += ["Cyclical (SWA)"]
    lgds += [
        lax.legend(
            lu0, labels, title=r"$L_2$ err., $u$, $t=0$", ncol=2,
            loc="lower right", bbox_to_anchor=(0.49, 0.0)),
        lax.legend(
            lu40, labels, title=r"$L_2$ err., $u$, $t=40$", ncol=2,
            loc="lower left", bbox_to_anchor=(0.51, 0.0))
    ]

    lax.add_artist(lgds[0])
    lax.add_artist(lgds[1])
    lax.add_artist(lgds[2])
    lax.add_artist(lgds[3])
    lax.axis("off")

    # save
    figdir.joinpath("cyclic-swa-tests").mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir.joinpath("cyclic-swa-tests", f"nl{nl}-nn{nn}-npts{nbs}.png"))


def create_annealing_test_plots(csimdir, coutdir, esimdir, eoutdir, figdir, arch):
    """create_annealing_test_plots
    """
    nl, nn, nbs = arch
    clogdata = log_parser(csimdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}"))
    elogdata = log_parser(esimdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}"))

    with h5open(coutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        crh5data = pandas.DataFrame({
            "steps": h5file["walltime/steps"][...],
            "elapsedtimes": h5file["walltime/elapsedtimes"][...],
            "u0": h5file["walltime/l2norms/u/0.0"][...],
            "u40": h5file["walltime/l2norms/u/40.0"][...],
        })

    with h5open(eoutdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        eh5data = pandas.DataFrame({
            "steps": h5file["walltime/steps"][...],
            "elapsedtimes": h5file["walltime/elapsedtimes"][...],
            "u0": h5file["walltime/l2norms/u/0.0"][...],
            "u40": h5file["walltime/l2norms/u/40.0"][...],
        })

    _plot_loss_err_vs_steps(arch, clogdata, elogdata, crh5data, eh5data, figdir)


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _csimdir = _projdir.joinpath("cyclic-sum")
    _coutdir = _projdir.joinpath("outputs", "cyclic-sum")
    _esimdir = _projdir.joinpath("base-cases")
    _eoutdir = _projdir.joinpath("outputs", "base-cases")

    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    _arch = (3, 128, 8192)

    _plot_lr(_figdir)
    create_annealing_test_plots(_csimdir, _coutdir, _esimdir, _eoutdir, _figdir, _arch)
