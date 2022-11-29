#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot contours in refined regions surrounding the cylinder.
"""
# %% preprocessing
import pathlib
import numpy
import itertools
import h5py
from matplotlib import pyplot
from matplotlib import colors


# %% paths
projdir = pathlib.Path(__file__).resolve().parents[3]
casedir = projdir.joinpath("modulus", "cylinder-2d-re200", "nl6-nn512-npts6400-unsteady-petibm")
figdir = projdir.joinpath("modulus", "cylinder-2d-re200", "figures", "refined")
datafile = projdir.joinpath("modulus", "cylinder-2d-re200", "outputs", "refined_region.h5")

# unified figure style
pyplot.style.use(projdir.joinpath("resources", "figstyle"))
figdir.mkdir(exist_ok=True)

# custom cmap for q-criterion
cmap = pyplot.get_cmap("turbo").colors
cmap[0] = [1.0, 1.0, 1.0]
cmap = colors.ListedColormap(cmap)

# %% load time and coords
with h5py.File(datafile, "r") as dset:
    times = dset["times"][...]
    times = times[(times >= 140.) & (times <= 142.5)]

coords = {}
with h5py.File(datafile, "r") as dset:
    for field, orent in itertools.product(["u", "v", "p", "vorticity_z"], ["x", "y"]):
        coords[f"petibm/{field}/{orent}"] = dset[f"petibm/{field}/{orent}"][...]
        coords[f"pinn/{field}/{orent}"] = dset[f"pinn/{orent}"][...]

with h5py.File(datafile, "r+") as dset:
    coords["pinn/qcriterion/x"] = dset["pinn/x"][...]
    coords["pinn/qcriterion/y"] = dset["pinn/y"][...]
    coords["petibm/qcriterion/x"] = dset["pinn/x"][...]
    coords["petibm/qcriterion/y"] = dset["pinn/y"][...]
    for t in times:
        data = dset[f"pinn/qcriterion/{t-5}"][...]
        dset.require_dataset(
            f"petibm/qcriterion/{t}", data.shape, data.dtype, True
        )[...] = data

# %% contour configs
lvls = {
    "u": numpy.linspace(-0.3, 1.2, 31),
    "v": numpy.linspace(-0.8, 0.8, 33),
    "p": numpy.linspace(-0.8, 0.5, 40),
    "vorticity_z": numpy.linspace(-5, 5, 51),
    "qcriterion": numpy.linspace(0, 4, 21),
}

names = {
    "u": r"$u$ velocity", "v": r"$v$ velocity", "p": "pressure",
    "vorticity_z": r"Vorticity", "qcriterion": "Q-criterion",
    "pinn": "Data-driven PINN", "petibm": "PetIBM"
}

fields = ["u", "v", "p", "vorticity_z", "qcriterion"]

for field, t in itertools.product(fields, times):
    print(f"Plotting {field} at time={t}")

    fig = pyplot.figure(figsize=(6.5, 3.25))
    fig.suptitle(rf"{names[field]}, t={t}$s$")

    gs = fig.add_gridspec(1, 3, width_ratios=[10, 10, 0.5])

    for i, solv in enumerate(["petibm", "pinn"]):

        with h5py.File(datafile, "r") as dset:
            data = dset[f"{solv}/{field}/{t}"][...]

        ax = fig.add_subplot(gs[0, i])
        ax.set_title(names[solv])

        if field == "qcriterion":
            cs = ax.contourf(
                coords[f"{solv}/{field}/x"], coords[f"{solv}/{field}/y"],
                data, lvls[field], extend="max", cmap=cmap
            )
        else:
            cs = ax.contour(
                coords[f"{solv}/{field}/x"], coords[f"{solv}/{field}/y"],
                data, lvls[field], linewidths=0.5, cmap="turbo"
            )

        ax.set_xlim(-1, 3)
        ax.set_ylim(-2, 2)
        ax.set_xlabel(r"$x$ ($m$)")

        if i == 0:
            ax.set_ylabel(r"$y$ ($m$)")
        else:
            ax.get_yaxis().set_ticks([])

        ax.set_aspect("equal", "box")
        ax.add_artist(pyplot.Circle((0., 0.), 0.5, color="w", ec="k", lw=1, zorder=10))

    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(cs, cax=cax, label=names[field])
    fig.savefig(figdir.joinpath(f"{field}_t{t}.png"))
    pyplot.close(fig)
