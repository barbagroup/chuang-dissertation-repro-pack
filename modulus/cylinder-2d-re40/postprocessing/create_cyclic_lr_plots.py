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
from matplotlib import pyplot

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.lr_simulator import cyclic_exp_range  # pylint: disable=import-error # noqa: E402

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def _plot_lr(figdir):
    """plot lr
    """
    steps = numpy.arange(0, 400000, 100)
    lr1 = cyclic_exp_range(steps, 0.999972, 1e-3, 1e-6, 2000)
    lr2 = cyclic_exp_range(steps, 0.99998, 1e-2, 1e-6, 5000)

    fig = pyplot.figure(figsize=(5, 2.5))
    fig.suptitle("Cyclic learning rate history")
    ax = fig.gca()
    ax.semilogy(steps, lr2, lw=1.5, alpha=0.7, label="Large cycle")
    ax.semilogy(steps, lr1, lw=1.5, alpha=0.7, label="Small cycle")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate")
    ax.grid()
    ax.legend(loc=0)

    fig.savefig(figdir.joinpath("learning-rate-hist.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)
    _plot_lr(_figdir)
