#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Total losses versus training iterations.
"""
import pathlib
import pandas
from matplotlib import pyplot

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})


def plot_single_case(data, figdir, jobname):
    """Plot a single case.
    """
    # initialize plot
    fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
    fig.suptitle(r"Convergence history: $L_2$-norm history at $t=2$")

    # raw model
    ax.semilogy(data.index[1:], data[("raw", "l2norm", "u", "2")].iloc[1:], label="u, raw")
    ax.semilogy(data.index[1:], data[("raw", "l2norm", "v", "2")].iloc[1:], label="v, raw")

    # swa model
    ax.semilogy(data.index[1:], data[("swa", "l2norm", "u", "2")].iloc[1:], label="u, raw")
    ax.semilogy(data.index[1:], data[("swa", "l2norm", "v", "2")].iloc[1:], label="v, raw")

    # finish up the plot
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$L_2$-norm")
    ax.legend(loc=0)
    fig.savefig(figdir.joinpath(f"tgv-pinn-l2norm-hist-{jobname}.png"), bbox_inches="tight", dpi=166)


def main(workdir, figdir):
    """The main function.
    """

    # cases
    cases = [f"nl1-nn{n}" for n in [32, 64, 128, 256, 512]]

    # big data
    data = pandas.read_csv(workdir.joinpath("output", "wall-time-errors.csv"), index_col=0, header=[0, 1, 2, 3, 4])

    for job in cases:
        print(f"Plotting {job}")
        plot_single_case(data[job], figdir, job)

    return 0


if __name__ == "__main__":
    import sys

    # directories
    rootdir = pathlib.Path(__file__).resolve().parents[1]
    modulusdir = rootdir.joinpath("modulus", "taylor-green-vortex-2d-re100")
    figdir = rootdir.joinpath("figures")
    figdir.mkdir(exist_ok=True)

    sys.exit(main(modulusdir, figdir))
