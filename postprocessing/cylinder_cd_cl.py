#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Drag and lift coefficients

"""
import pathlib
import numpy
import h5py
from matplotlib import pyplot

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})


def plot_single_case(workdir, figdir, petibmdata, jobname):
    """Plot a single case.
    """

    # read data
    with h5py.File(workdir.joinpath("snapshots.h5"), "r") as dset:
        t = dset[f"{jobname}/times"][...]
        cd = dset[f"{jobname}/cd"][...]
        cl = dset[f"{jobname}/cl"][...]

    # plot
    fig, ax = pyplot.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)
    fig.suptitle(r"Lift and drag coefficients, $Re=200$")

    # petibm
    ax.plot(petibmdata[:, 0], petibmdata[:, 1]*2, "k-", lw=1, alpha=0.8, label=r"$C_D$, PetIBM")
    ax.plot(petibmdata[:, 0], petibmdata[:, 2]*2, "k-.", lw=1, alpha=0.8, label=r"$C_L$, PetIBM")

    # pinn
    ax.plot(t, cd, c="tab:red", ls="-", lw=1, alpha=0.8, label=r"$C_D$, PINN")
    ax.plot(t, cl, c="tab:red", ls="-.", lw=1, alpha=0.8, label=r"$C_L$, PINN")

    ax.set_ylim(-1, 2)
    ax.set_xlabel(r"$T_{sim}$")
    ax.set_ylabel("$C_D$ and $C_L$")
    ax.legend(loc=0)
    fig.savefig(figdir.joinpath(f"cylinder-cd-cl-{jobname}.png"), bbox_inches="tight", dpi=166)


def main(workdir, figdir, petibmdata):
    """The main function.
    """

    # cases
    cases = [f"nn{n}" for n in [256, 512]]
    cases.extend([f"nn256-shedding-ic-t{n}" for n in [100, 130]])

    for job in cases:
        print(f"Plotting {job}")
        plot_single_case(workdir, figdir, petibmdata, job)

    return 0


if __name__ == "__main__":
    import sys

    # directories
    rootdir = pathlib.Path(__file__).resolve().parents[1]
    modulusdir = rootdir.joinpath("modulus", "cylinder-2d-re200", "output")
    figdir = rootdir.joinpath("figures")
    figdir.mkdir(exist_ok=True)

    # read in petibm data
    petibmdir = rootdir.joinpath("petibm", "cylinder-2d-re200", "output")
    petibmdata = numpy.loadtxt(petibmdir.joinpath("forces-0.txt"), dtype=float)

    # call the main function
    sys.exit(main(modulusdir, figdir, petibmdata))
