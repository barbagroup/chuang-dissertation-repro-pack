#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Processing cylinder flow 2D Re200 from both PetIBM and Modulus.
"""
import sys
import pathlib
import numpy
import h5py
from matplotlib import pyplot
from matplotlib import cm
from matplotlib import colors

# find helpers
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].joinpath("modulus")))
from helpers.tbreader import read_tensorboard_data  # pylint: disable=import-error  # noqa: E402

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
})


# directories
rootdir = pathlib.Path(__file__).resolve().parents[1]
modulusdir = rootdir.joinpath("modulus", "cases", "cylinder-2d", "re200")
petibmdir = rootdir.joinpath("petibm", "cylinder-2d", "re200")
rootdir.joinpath("figures").mkdir(exist_ok=True)


# =================================================
# u velocity contourf from PetIBM for demonstration
# =================================================

with h5py.File(petibmdir.joinpath("output", "grid.h5"), "r") as dset:
    x = {"u": dset["u"]["x"][...], "wz": dset["wz"]["x"][...]}
    y = {"u": dset["u"]["y"][...], "wz": dset["wz"]["y"][...]}

with h5py.File(petibmdir.joinpath("output", "0020000.h5"), "r") as dset:
    u = dset["u"][...]
    wz = dset["wz"][...]

norms = {
    "u": colors.Normalize(-0.32, 1.4),
    "wz": colors.CenteredNorm(0., 5)
}

fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)

axs[0].contourf(x["u"], y["u"], u, 256, norm=norms["u"], cmap="cividis")
axs[0].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
axs[0].set_xlim(-3, 14)
axs[0].set_ylim(-3, 3)
axs[0].set_aspect("equal", "box")
axs[0].set_ylabel("y")
axs[0].set_title(r"$u$ velocity")

axs[1].contourf(x["wz"], y["wz"], wz, 512, norm=norms["wz"], cmap="cividis")
axs[1].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
axs[1].set_xlim(-3, 14)
axs[1].set_ylim(-3, 3)
axs[1].set_aspect("equal", "box")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_title(r"Vorticity")

fig.colorbar(cm.ScalarMappable(norms["u"], "cividis"), ax=axs[0])
fig.colorbar(cm.ScalarMappable(norms["wz"], "cividis"), ax=axs[1])
fig.suptitle(r"Flow distribution, $Re=200$ at $T_{sim}=100$, PetIBM")
fig.savefig(rootdir.joinpath("figures", "cylinder-petibm-contour-t100.png"), bbox_inches="tight", dpi=166)


# ================
# training history
# ================
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
for nn in [256, 512]:
    data = read_tensorboard_data(modulusdir.joinpath(f"nn_{nn}", "outputs"))
    ax.semilogy(data["step"], data["loss"].ewm(span=10).mean(), label=f"{nn} neurons per layer", alpha=0.8)
ax.set_xlabel("Iteration")
ax.set_ylabel("Total loss")
ax.legend(loc=0)
fig.suptitle("Convergence history: total loss v.s. iteration")
fig.savefig(rootdir.joinpath("figures", "cylinder-pinn-training-convergence.png"), bbox_inches="tight", dpi=166)


# =============================
# u velocity contourf from PINN
# =============================

with numpy.load(modulusdir.joinpath("nn_256", "outputs", "cylinder-re200-t100.0-u.npz"), "r") as dset:
    x = dset["x"]
    y = dset["y"]
    u = dset["val"]

with numpy.load(modulusdir.joinpath("nn_256", "outputs", "cylinder-re200-t100.0-vorticity_z.npz"), "r") as dset:
    wz = dset["val"]

norms = {
    "u": colors.Normalize(-0.32, 1.4),
    "wz": colors.CenteredNorm(0., 5)
}

fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)

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
fig.suptitle(r"Flow distribution, $Re=200$ at $T_{sim}=100$, PINN")
fig.savefig(rootdir.joinpath("figures", "cylinder-pinn-contour-t100.png"), bbox_inches="tight", dpi=166)


# =============================
# CD and CL
# =============================

fig, ax = pyplot.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)

# petibm
data = numpy.loadtxt(petibmdir.joinpath("output", "forces-0.txt"), dtype=float)
ax.plot(data[:, 0], data[:, 1]*2, "k-", lw=1, alpha=0.8, label=r"$C_D$, PetIBM")
ax.plot(data[:, 0], data[:, 2]*2, "k-.", lw=1, alpha=0.8, label=r"$C_L$, PetIBM")

# pinn
with numpy.load(modulusdir.joinpath("nn_256", "outputs", "cd_cl.npz")) as dset:
    data = numpy.zeros((dset["times"].size, 3), dtype=float)
    data[:, 0] = dset["times"]
    data[:, 1] = dset["cd"]
    data[:, 2] = dset["cl"]
ax.plot(data[:, 0], data[:, 1], c="tab:red", ls="-", lw=1, alpha=0.8, label=r"$C_D$, PINN")
ax.plot(data[:, 0], data[:, 2], c="tab:red", ls="-.", lw=1, alpha=0.8, label=r"$C_L$, PINN")

ax.set_ylim(-1, 2)
ax.set_xlabel(r"$T_{sim}$")
ax.set_ylabel("$C_D$ and $C_L$")
ax.legend(loc=0)
fig.suptitle(r"Lift and drag coefficients, $Re=200$")
fig.savefig(rootdir.joinpath("figures", "cylinder-cd-cl.png"), bbox_inches="tight", dpi=166)


# =============================
# u velocity contourf from PINN
# =============================

with numpy.load(modulusdir.joinpath("nn_256", "outputs", "cylinder-re200-t100.0-u.npz"), "r") as dset:
    x = dset["x"]
    y = dset["y"]
    u = dset["val"]

with numpy.load(modulusdir.joinpath("nn_256", "outputs", "cylinder-re200-t100.0-vorticity_z.npz"), "r") as dset:
    wz = dset["val"]

norms = {
    "u": colors.Normalize(-0.32, 1.4),
    "wz": colors.CenteredNorm(0., 5)
}

fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 4), dpi=166)

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
fig.suptitle(r"Flow distribution, $Re=200$ at $T_{sim}=100$, PINN")
fig.savefig(rootdir.joinpath("figures", "cylinder-pinn-contour-t100.png"), bbox_inches="tight", dpi=166)
