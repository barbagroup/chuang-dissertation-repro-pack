#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D cylinder flow Re200, T=200, Modulus/PINN
"""
import pathlib
import h5py
from matplotlib import pyplot
from matplotlib import cm
from matplotlib import colors

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})


def plot_single_case(workdir, figdir, jobname):
    """Plot for a single case.
    """

    # read data
    with h5py.File(workdir.joinpath("output", "snapshots.h5"), "r") as dset:
        x = dset[f"{jobname}/x"][...]
        y = dset[f"{jobname}/y"][...]
        u = dset[f"{jobname}/200.0/u"][...]
        wz = dset[f"{jobname}/200.0/vorticity_z"][...]

    # normalization for colormaps
    norms = {
        "u": colors.Normalize(-0.32, 1.4),
        "wz": colors.CenteredNorm(0., 5)
    }

    # plot
    fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)
    fig.suptitle(r"Flow distribution, $Re=200$ at $t=200$, PINN")

    axs[0].contourf(x, y, u, 256, norm=norms["u"], cmap="cividis")
    axs[0].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
    axs[0].set_xlim(-3, 14)
    axs[0].set_ylim(-3, 3)
    axs[0].set_aspect("equal", "box")
    axs[0].set_ylabel("y")
    axs[0].set_title(r"$u$ velocity")

    axs[1].contourf(x, y, wz, 512, norm=norms["wz"], cmap="cividis")
    axs[1].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
    axs[1].set_xlim(-3, 14)
    axs[1].set_ylim(-3, 3)
    axs[1].set_aspect("equal", "box")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(r"Vorticity")

    fig.colorbar(cm.ScalarMappable(norms["u"], "cividis"), ax=axs[0])
    fig.colorbar(cm.ScalarMappable(norms["wz"], "cividis"), ax=axs[1])
    fig.savefig(figdir.joinpath(f"cylinder-pinn-contour-t200-{jobname}.png"), bbox_inches="tight", dpi=166)


def main(workdir, figdir):
    """Main function.
    """

    # cases
    cases = [f"nn{n}" for n in [256, 512]]
    cases.extend([f"nn256-shedding-ic-t{n}" for n in [100, 130]])

    for job in cases:
        print(f"Plotting {job}")
        plot_single_case(workdir, figdir, job)

    return 0


if __name__ == "__main__":
    import sys

    # directories
    rootdir = pathlib.Path(__file__).resolve().parents[1]
    modulusdir = rootdir.joinpath("modulus", "cylinder-2d-re200")
    figdir = rootdir.joinpath("figures")
    figdir.mkdir(exist_ok=True)

    sys.exit(main(modulusdir, figdir))
