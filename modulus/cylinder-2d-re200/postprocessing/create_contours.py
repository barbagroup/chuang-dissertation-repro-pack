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
from matplotlib import pyplot
from matplotlib import ticker

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def read_petibm_snapshot(petibmdir, time):
    """Read snapshots from PetIBM simulations.
    """
    assert float(int(time)) == float(time), f"Only supports integer time. Got {time}."

    # hard-coded simulation parameters; should match the config.yaml in the PetIBM case
    dt = 0.005
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

        try:
            vals = {
                r"$u$": h5file[f"fields/{time}/u"][...],
                r"$v$": h5file[f"fields/{time}/v"][...],
                r"$p$": h5file[f"fields/{time}/p"][...],
                r"$\omega_z$": h5file[f"fields/{time}/vorticity_z"][...],
            }
        except KeyError:
            print(f"fields/{time}/u")
            raise

    return coords, vals


def steady_state_plot(workdir, figdir):
    """Plot contours from three cases side by side.
    """

    # colored contour levels
    lvl1 = {
        r"$u$": numpy.linspace(-0.3, 1.2, 16),
        r"$v$": numpy.linspace(-0.8, 0.8, 17),
        r"$p$": numpy.linspace(-1.0, 0.5, 16),
        r"$\omega_z$": numpy.linspace(-5.0, 5.0, 11),
    }

    # contour line levels
    lvl2 = {
        r"$u$": numpy.linspace(0.0, 1.0, 6),
        r"$v$": numpy.linspace(-0.5, 0.5, 6),
        r"$p$": numpy.linspace(-0.75, 0.25, 6),
        r"$\omega_z$": numpy.linspace(-5.0, 5.0, 6),
    }

    casename = "nl6-nn512-npts6400-steady"
    locs = ((0, 0), (0, 1), (2, 0), (2, 1))
    data = read_pinn_snapshot(workdir, casename, 0)

    gridspec = {"height_ratios": [1, 0.1, 1, 0.1]}
    fig, axs = pyplot.subplots(4, 2, sharex=False, sharey=False, figsize=(6.5, 4), gridspec_kw=gridspec)
    fig.suptitle(r"Steady PINN, 2D Cylinder Flow, $Re=200$")

    for loc, f in zip(locs, lvl1.keys()):
        ct1 = axs[loc].contourf(*data[0][f], data[1][f], lvl1[f], cmap="cividis", extend="both")
        ct2 = axs[loc].contour(*data[0][f], data[1][f], lvl2[f], colors='black', linewidths=0.5)
        axs[loc].clabel(ct2, lvl2[f], fmt="%1.1f", inline_spacing=0.25, fontsize="small")
        axs[loc].add_artist(pyplot.Circle((0., 0.), 0.5, color="w", zorder=10))
        axs[loc].set_xlim(-3, 14)
        axs[loc].set_ylim(-2.5, 2.5)
        axs[loc].set_aspect("equal", "box")
        axs[loc].set_title(f)

        fmt1 = ticker.ScalarFormatter(useOffset=True, useMathText=True)
        fmt1.set_powerlimits((0, 0))
        cbar = fig.colorbar(ct1, cax=axs[loc[0]+1, loc[1]], format=fmt1, orientation="horizontal")
        cbar.ax.get_yaxis().set_offset_position("left")
        cbar.set_ticks(lvl2[f])

    axs[2, 0].sharex(axs[0, 0])
    axs[2, 1].sharex(axs[0, 1])
    axs[0, 1].sharey(axs[0, 0])
    axs[2, 1].sharey(axs[2, 0])

    axs[2, 0].set_xlabel(r"$x$")
    axs[2, 1].set_xlabel(r"$x$")
    axs[0, 0].set_ylabel(r"$y$")
    axs[2, 0].set_ylabel(r"$y$")

    fig.savefig(figdir.joinpath("contour-comparison-steady.png"))


def three_cols_plot(workdir, petibmdir, figdir, time):
    """Plot contours from three cases side by side.
    """

    cases = {
        "PetIBM": "PetIBM",
        "nl6-nn512-npts6400-unsteady": r"Unsteady PINN",
        "nl6-nn512-npts6400-unsteady-petibm": r"Data-driven PINN",
    }

    # colored contour levels
    lvl1 = {
        r"$u$": numpy.linspace(-0.3, 1.2, 16),
        r"$v$": numpy.linspace(-0.8, 0.8, 17),
        r"$p$": numpy.linspace(-1.0, 0.5, 16),
        r"$\omega_z$": numpy.linspace(-5.0, 5.0, 11),
    }

    # contour line levels
    lvl2 = {
        r"$u$": numpy.linspace(0.0, 1.0, 6),
        r"$v$": numpy.linspace(-0.5, 0.5, 6),
        r"$p$": numpy.linspace(-0.75, 0.25, 6),
        r"$\omega_z$": numpy.linspace(-5.0, 5.0, 6),
    }

    fignames = ("uv", "pwz")

    # data
    data = []
    for case in cases.keys():
        if case == "PetIBM":
            data.append(read_petibm_snapshot(petibmdir, time))
        else:
            data.append(read_pinn_snapshot(workdir, case, time))

    # figure
    for fi, fields in enumerate(((r"$u$", r"$v$"), (r"$p$", r"$\omega_z$"))):
        gridspec = {"height_ratios": [1, 1, 1, 0.15]}
        fig, axs = pyplot.subplots(4, 2, sharex=False, sharey=False, figsize=(6.5, 4.5), gridspec_kw=gridspec)
        fig.suptitle(rf"Flow Field Comparison, 2D Cylinder Flow, $Re=200$, $t={int(time)}$")

        for j, f in enumerate(fields):
            for i in range(3):
                ct1 = axs[i, j].contourf(*data[i][0][f], data[i][1][f], lvl1[f], cmap="cividis", extend="both")
                ct2 = axs[i, j].contour(*data[i][0][f], data[i][1][f], lvl2[f], colors='black', linewidths=0.5)
                axs[i, j].clabel(ct2, lvl2[f], fmt="%1.1f", inline_spacing=0.25, fontsize="small")
                axs[i, j].add_artist(pyplot.Circle((0., 0.), 0.5, color="w", zorder=10))
                axs[i, j].set_xlim(-3, 14)
                axs[i, j].set_ylim(-2.5, 2.5)
                axs[i, j].set_aspect("equal", "box")

            axs[1, j].sharex(axs[0, j])
            axs[2, j].sharex(axs[0, j])

            fmt1 = ticker.ScalarFormatter(useOffset=True, useMathText=True)
            fmt1.set_powerlimits((0, 0))
            cbar = fig.colorbar(ct1, cax=axs[3, j], format=fmt1, orientation="horizontal")
            cbar.ax.get_yaxis().set_offset_position("left")
            cbar.set_ticks(lvl2[f])
            cbar.set_label(f)

        axs[0, 1].sharey(axs[0, 0])
        axs[1, 1].sharey(axs[1, 0])
        axs[2, 1].sharey(axs[2, 0])

        axs[0, 0].set_title(r"$u$")
        axs[0, 1].set_title(r"$v$")
        axs[0, 0].set_ylabel("PetIBM\n"+r"$y$")
        axs[1, 0].set_ylabel("Unsteady PINN\n"+r"$y$")
        axs[2, 0].set_ylabel("Data-driven PINN\n"+r"$y$")
        axs[2, 0].set_xlabel(r"$x$")
        axs[2, 1].set_xlabel(r"$x$")

        fig.savefig(figdir.joinpath(f"contour-comparison-{fignames[fi]}-t{int(time)}.png"))


if __name__ == "__main__":

    # point workdir to the correct folder
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _petibmdir = pathlib.Path(__file__).resolve().parents[3].joinpath("petibm", "cylinder-2d-re200")
    _figdir = _workdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    steady_state_plot(_workdir, _figdir)

    for t in [10., 50., 140., 141., 142., 143., 144., 145., 190.]:
        print(t)
        three_cols_plot(_workdir, _petibmdir, _figdir, t)
