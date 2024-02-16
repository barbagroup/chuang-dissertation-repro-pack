#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Postprocessing.
"""
import sys
import pathlib
from cycler import cycler
from matplotlib import pyplot

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error  # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def plot_training_history(workdir, figdir, ws=10):
    """Plot figures related to training loss.
    """

    cases = {
        "nl6-nn512-npts25600-large-cycle-unsteady": "Unsteady solver",
        "nl6-nn512-npts25600-large-cycle-steady": "Steady solver",
    }
    data = [log_parser(workdir.joinpath(case)) for case in cases.keys()]

    # fixed cycling kwargs
    styles = cycler("color", ["tab:red", "tab:blue"]) + cycler("label", cases.values())

    fig = pyplot.figure(figsize=(3.75, 3.1))
    gs = fig.add_gridspec(2, 1)

    # against steps
    kwargs = styles()
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Aggregated loss v.s. iterations")
    ax.semilogy(data[0].index, data[0].loss.rolling(window=ws).min(), lw=1, **next(kwargs))
    ax.semilogy(data[1].index, data[1].loss.rolling(window=ws).min(), lw=1, **next(kwargs))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Aggregated loss")
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e2])
    ax.legend(loc=0)

    # against runtime
    kwargs = styles()
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Aggregated loss v.s. run time")
    ax.semilogy(data[0]["time elapsed"], data[0].loss.rolling(window=ws).min(), lw=1, **next(kwargs))
    ax.semilogy(data[1]["time elapsed"], data[1].loss.rolling(window=ws).min(), lw=1, **next(kwargs))
    ax.set_xlabel("Run time (hours)")
    ax.set_ylabel("Aggregated loss")
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e2])
    ax.legend(loc=0)

    # save
    fig.savefig(workdir.joinpath("figures", "loss-hist.png"))


if __name__ == "__main__":

    # point workdir to the correct folder
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _petibmdir = pathlib.Path(__file__).resolve().parents[3].joinpath("petibm", "cylinder-2d-re40")
    _figdir = _workdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    plot_training_history(_workdir, _figdir)
