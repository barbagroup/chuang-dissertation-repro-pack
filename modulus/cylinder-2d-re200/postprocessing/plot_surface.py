#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot figures related to the cylinder surface.
"""
# %% preprocessing
import pathlib
import h5py
from matplotlib import pyplot


# %% paths
projdir = pathlib.Path(__file__).resolve().parents[3]
casedir = projdir.joinpath("modulus", "cylinder-2d-re200", "nl6-nn512-npts6400-unsteady-petibm")
figdir = projdir.joinpath("modulus", "cylinder-2d-re200", "figures", "surface")
datafile = projdir.joinpath("modulus", "cylinder-2d-re200", "data", "surface_data.h5")

# unified figure style
pyplot.style.use(projdir.joinpath("resources", "figstyle"))
figdir.mkdir(exist_ok=True)

# settings
fields = ["u", "v", "p"]
names = {"u": r"$u$ velocity", "v": r"$v$ velocity", "p": "Pressure"}
units = {"u": r"$m\cdot s^{-1}$", "v": r"$m\cdot s^{-1}$", "p": r"$kg \cdot m^{-1}\cdot s^{-2}$"}

# %% load input data
with h5py.File(datafile, "r") as dset:
    times = dset["times"][...]
    thetas = dset["theta"][...]
    x = dset["x"][...]
    y = dset["x"][...]

# %%
for t in times:
    for field in fields:
        print(f"Plotting {field} at time={t}")

        # read pinn data
        with h5py.File(datafile, "r") as dset:
            pinndata = dset[f"pinn/{field}/{t}"][...]
            petibmdata = dset[f"petibm/{field}/{t}"][...]

        # create the plot
        pyplot.figure(figsize=(3.5, 3.5))
        pyplot.title(rf"{names[field]} on cylinder, t={t}$s$")
        pyplot.plot(thetas, petibmdata, c="k", lw=2, ls="-", label="PetIBM")
        pyplot.plot(thetas, pinndata, c="tab:red", lw=2, ls="--", alpha=0.8, label="Data-driven PINN")
        pyplot.xlabel(r"Angle ($\theta$) from back of the cylinder in radius")
        pyplot.ylabel(f"{names[field]} ({units[field]})")
        pyplot.legend(loc=0)
        pyplot.gca().get_yaxis().set_offset_position("left")
        pyplot.savefig(figdir.joinpath(f"surface-{field}-t{t}.png"))
        pyplot.close()
