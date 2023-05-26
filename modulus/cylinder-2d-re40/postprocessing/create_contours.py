#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create contours.
"""
import pathlib
import numpy
from h5py import File as h5open
from matplotlib import pyplot
from matplotlib import ticker

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def read_petibm_snapshot(petibmdir, time):
    """Read snapshots from PetIBM simulations.
    """
    assert float(int(time)) == float(time), f"Only supports integer time. Got {time}."

    # hard-coded simulation parameters; should match the config.yaml in the PetIBM case
    dt = 0.01
    step = int(time/dt+0.5)

    # get gridline and calculate coordinates
    with h5open(petibmdir.joinpath("output", "grid.h5"), "r") as h5file:
        coords = {
            r"$u$": numpy.meshgrid(h5file["u"]["x"][...], h5file["u"]["y"][...]),
            r"$v$": numpy.meshgrid(h5file["v"]["x"][...], h5file["v"]["y"][...]),
            r"$p$": numpy.meshgrid(h5file["p"]["x"][...], h5file["p"]["y"][...]),
            r"$\omega_z$": numpy.meshgrid(h5file["wz"]["x"][...], h5file["wz"]["y"][...]),
        }

    # get data
    with h5open(petibmdir.joinpath("output", f"{step:07d}.h5"), "r") as h5file:
        vals = {
            r"$u$": h5file["u"][...],
            r"$v$": h5file["v"][...],
            r"$p$": h5file["p"][...],
            r"$\omega_z$": h5file["wz"][...],
        }

    return coords, vals


def read_pinn_snapshot(workdir, casename, time):
    """Read snapshots from PINN simulations.
    """
    with h5open(workdir.joinpath("outputs", f"{casename}-raw.h5"), "r") as h5file:
        coords = {
            r"$u$": (h5file["fields/x"][...], h5file["fields/y"][...]),
            r"$v$": (h5file["fields/x"][...], h5file["fields/y"][...]),
            r"$p$": (h5file["fields/x"][...], h5file["fields/y"][...]),
            r"$\omega_z$": (h5file["fields/x"][...], h5file["fields/y"][...]),
        }

        if not h5file.attrs["unsteady"]:
            time = "steady"

        vals = {
            r"$u$": h5file[f"fields/{time}/u"][...],
            r"$v$": h5file[f"fields/{time}/v"][...],
            r"$p$": h5file[f"fields/{time}/p"][...],
            r"$\omega_z$": h5file[f"fields/{time}/vorticity_z"][...],
        }

    return coords, vals


def three_cols_plot(workdir, petibmdir, figdir):
    """Plot contours from three cases side by side.
    """

    cases = {
        "PetIBM": "PetIBM",
        "nl6-nn512-npts25600-large-cycle-steady": "PINN, steady",
        "nl6-nn512-npts25600-large-cycle-unsteady": "PINN, unsteady",
    }

    # colored contour levels
    lvl1 = {
        r"$u$": numpy.linspace(-0.1, 1.1, 13),
        r"$v$": numpy.linspace(-0.5, 0.5, 21),
        r"$p$": numpy.linspace(-0.5, 0.5, 21),
        r"$\omega_z$": numpy.linspace(-3.0, 3.0, 13),
    }

    # contour line levels
    lvl2 = {
        r"$u$": numpy.linspace(0.0, 1.0, 6),
        r"$v$": numpy.linspace(-0.5, 0.5, 11),
        r"$p$": numpy.linspace(-0.5, 0.5, 11),
        r"$\omega_z$": numpy.linspace(-3.0, 3.0, 13),
    }

    # data
    data = []
    for case in cases.keys():
        if case == "PetIBM":
            data.append(read_petibm_snapshot(petibmdir, 20))
        else:
            data.append(read_pinn_snapshot(workdir, case, 20.0))

    # figure
    fig, axs = pyplot.subplots(4, 3, sharex=True, sharey=True, figsize=(6, 8))

    for col, ((case, label), (coords, vals)) in enumerate(zip(cases.items(), data)):
        for row, (field, val) in enumerate(vals.items()):
            ax = axs[row, col]
            ct1 = ax.contourf(*coords[field], val, lvl1[field], cmap="cividis", extend="both")
            ct2 = ax.contour(*coords[field], val, lvl2[field], colors='black', linewidths=0.5)
            ax.clabel(ct2, lvl2[field], fmt="%1.1f", inline_spacing=0.25, fontsize="small")
            ax.add_artist(pyplot.Circle((0., 0.), 0.5, color="w", zorder=10))

            ax.set_title(label)
            ax.set_xlim(-3, 4)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal", "box")

            if col == 2:  # only the last column will have colorbars
                fmt1 = ticker.ScalarFormatter(useOffset=True, useMathText=True)
                fmt1.set_powerlimits((0, 0))
                cbar = fig.colorbar(
                    ct1, ax=axs[row, :], format=fmt1, orientation="horizontal",
                    fraction=0.05, aspect=60,
                )
                cbar.ax.get_yaxis().set_offset_position("left")
                cbar.set_ticks(lvl2[field])
                cbar.ax.annotate(field, (-0.125, 0), xycoords="axes fraction")

    for row in range(4):
        axs[row, 0].set_ylabel("y")
        axs[row, 1].yaxis.set_visible(False)
        axs[row, 2].yaxis.set_visible(False)

    for col in range(3):
        axs[0, col].xaxis.set_visible(False)
        axs[1, col].xaxis.set_visible(False)
        axs[2, col].xaxis.set_visible(False)
        axs[3, col].set_xlabel("x")

    fig.set_constrained_layout_pads(w_pad=0, h_pad=0.03, hspace=0, wspace=0)

    # save
    fig.savefig(figdir.joinpath("contour-comparison.png"), bbox_inches="tight")


if __name__ == "__main__":

    # point workdir to the correct folder
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _petibmdir = pathlib.Path(__file__).resolve().parents[3].joinpath("petibm", "cylinder-2d-re40")
    _figdir = _workdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    three_cols_plot(_workdir, _petibmdir, _figdir)
