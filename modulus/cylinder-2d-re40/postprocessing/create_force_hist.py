#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Postprocessing.
"""
import pathlib
import numpy
from h5py import File as h5open
from cycler import cycler
from matplotlib import pyplot
from matplotlib.legend_handler import HandlerTuple

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def plot_force_hist(workdir, petibmdir, figdir):
    """Plot a field of all cases at a single time.
    """
    styles = (cycler("color", ["k"]+["tab:blue", "tab:red", "tab:gree"]) + cycler("ls", ["-", "-.", "--", ":"]))()

    cases = {
        "nl6-nn512-npts25600-large-cycle-steady": "Steady PINN solver",
        "nl6-nn512-npts25600-large-cycle-unsteady": "Unsteady PINN solver",
    }

    # line and label holders
    lines = []
    labels = []

    # add PetIBM results
    petibm = numpy.loadtxt(petibmdir.joinpath("output", "forces-0.txt"), dtype=float)

    fig = pyplot.figure(figsize=(3.75, 2))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    # plot petibm result
    style = next(styles)  # the two line will use the same style
    lines.append((
        ax.plot(petibm[:, 0], petibm[:, 1]*2, lw=1, **style)[0],
        ax.plot(petibm[:, 0], petibm[:, 2]*2, lw=1, **style)[0]
    ))
    labels.append("PetIBM")

    # add lines from each case
    for case, label in cases.items():
        with h5open(workdir.joinpath("outputs", f"{case}-raw.h5"), "r") as h5file:
            if h5file.attrs["unsteady"]:
                cd = h5file["coeffs/cd"][...]
                cl = h5file["coeffs/cl"][...]
                times = h5file["coeffs/times"][...]
            else:
                cd = numpy.full_like(petibm[:, 0], float(h5file["coeffs/cd"][...]))
                cl = numpy.full_like(petibm[:, 0], float(h5file["coeffs/cl"][...]))
                times = petibm[:, 0]

        style = next(styles)  # the two line will use the same style
        lines.append((
            ax.plot(times, cd, lw=1.25, alpha=0.95, **style)[0],
            ax.plot(times, cl, lw=1.25, alpha=0.95, **style)[0]
        ))
        labels.append(label)

    ax.set_xlim(0., 20.)
    ax.set_xlabel(r"$t$")
    ax.set_ylim(-0.5, 2.5)
    ax.set_ylabel("$C_D$ and $C_L$")
    ax.legend(lines, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=2, loc=0)

    # save
    fig.savefig(figdir.joinpath("drag-lift-coeffs.png"))


if __name__ == "__main__":

    # point workdir to the correct folder
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _petibmdir = pathlib.Path(__file__).resolve().parents[3].joinpath("petibm", "cylinder-2d-re40")
    _figdir = _workdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)
    plot_force_hist(_workdir, _petibmdir, _figdir)
