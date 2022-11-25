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


# %% paths
projdir = pathlib.Path(__file__).resolve().parents[3]
casedir = projdir.joinpath("modulus", "cylinder-2d-re200", "nl6-nn512-npts6400-unsteady-petibm")
figdir = projdir.joinpath("modulus", "cylinder-2d-re200", "figures", "refined")
datafile = projdir.joinpath("modulus", "cylinder-2d-re200", "data", "refined_region.h5")

# unified figure style
pyplot.style.use(projdir.joinpath("resources", "figstyle"))
figdir.mkdir(exist_ok=True)

# %% load time and coords
with h5py.File(datafile, "r") as dset:
    times = dset["times"][...]

coords = {}
with h5py.File(datafile, "r") as dset:
    for field, orent in itertools.product(["u", "v", "p", "vorticity_z"], ["x", "y"]):
        coords[f"petibm/{field}/{orent}"] = dset[f"petibm/{field}/{orent}"][...]
        coords[f"pinn/{field}/{orent}"] = dset[f"pinn/{orent}"][...]

# %% contour configs
lvls = {
    "u": numpy.linspace(-0.3, 1.2, 31),
    "v": numpy.linspace(-0.8, 0.8, 33),
    "p": numpy.linspace(-0.8, 0.5, 40),
    "vorticity_z": numpy.linspace(-5, 5, 51),
    "qcriterion": numpy.linspace(-4, 4, 41),
}

names = {
    "u": r"$u$ velocity", "v": r"$v$ velocity", "p": "pressure", "vorticity_z": r"vorticity",
    "qcriterion": "Q-criterion",
    "pinn": "Data-driven PINN", "petibm": "PetIBM"
}

for solv, field, t in itertools.product(["pinn", "petibm"], ["u", "v", "p", "vorticity_z"], times):
    print(f"Plotting {field} at time={t} for {solv}")

    with h5py.File(datafile, "r") as dset:
        data = dset[f"{solv}/{field}/{t}"][...]

    pyplot.figure(figsize=(4, 4))
    pyplot.title(rf"{names[solv]}, {names[field]}, t={t}$s$")
    pyplot.contour(
        coords[f"{solv}/{field}/x"], coords[f"{solv}/{field}/y"], data,
        lvls[field], linewidths=0.5, cmap="turbo"
    )
    pyplot.xlim(-1, 3)
    pyplot.ylim(-2, 2)
    pyplot.xlabel(r"$x$ ($m$)")
    pyplot.ylabel(r"$y$ ($m$)")
    pyplot.gca().set_aspect("equal", "box")
    pyplot.gca().add_artist(pyplot.Circle((0., 0.), 0.5, color="w", ec="k", lw=1, zorder=10))
    pyplot.colorbar(orientation="vertical", extend="both", shrink=0.75)
    pyplot.savefig(figdir.joinpath(f"{solv}_{field}_t{t}.png"))
    pyplot.close()

# qcriterion
for t in times:
    print(f"Plotting qcriterion at time={t} for PINN")

    with h5py.File(datafile, "r") as dset:
        data = dset[f"pinn/qcriterion/{t}"][...]

    pyplot.figure(figsize=(4, 4))
    pyplot.title(rf"{names['pinn']}, {names['qcriterion']}, t={t}$s$")
    pyplot.contourf(
        coords["pinn/u/x"], coords["pinn/u/y"], data,
        lvls["qcriterion"], extend="both", cmap="turbo",
    )
    pyplot.colorbar(orientation="vertical", extend="both", shrink=0.75)
    pyplot.xlim(-1, 3)
    pyplot.ylim(-2, 2)
    pyplot.xlabel(r"$x$ ($m$)")
    pyplot.ylabel(r"$y$ ($m$)")
    pyplot.gca().set_aspect("equal", "box")
    pyplot.gca().add_artist(pyplot.Circle((0., 0.), 0.5, color="w", ec="k", lw=1, zorder=10))
    pyplot.savefig(figdir.joinpath(f"pinn-qcriterion-t{t}.png"))
    pyplot.close()
