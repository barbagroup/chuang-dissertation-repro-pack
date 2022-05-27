#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""TGV 2D Re100, comparison, PetIBM vs Modulus
"""
import sys
import pathlib
import numpy
import pandas
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


def contour_data_reader(workdir, key):
    """helper"""
    with numpy.load(workdir.joinpath(f"tgv-re100-t32.0-{key}.npz")) as dset:
        values = {"x": dset["x"], "y": dset["y"], "val": dset["val"]}
    return values


# directories
rootdir = pathlib.Path(__file__).resolve().parents[1]
modulusdir = rootdir.joinpath("modulus", "cases", "taylor-green-vortex-2d", "one-net", "re100")
petibmdir = rootdir.joinpath("petibm", "taylor-green-vortex-2d", "re100")
rootdir.joinpath("figures").mkdir(exist_ok=True)

# ===========================================================
# plot example contour for TGV 2D Re100 from PetIBM 1024x1024
# ===========================================================
data = {key: contour_data_reader(petibmdir.joinpath("1024x1024", "output"), key) for key in ["u", "v"]}
levels = numpy.linspace(-0.6, 0.6, 13)  # contour levels
fig, axs = pyplot.subplots(1, 2, sharex=False, sharey=True, figsize=(6, 3), dpi=166)
axs[0].contourf(data["u"]["x"], data["u"]["y"], data["u"]["val"], 128, vmin=-0.6, vmax=0.6, cmap="cividis")
axs[0].contour(data["u"]["x"], data["u"]["y"], data["u"]["val"], levels, colors="k", linewidths=0.5)
axs[0].set_aspect("equal", "box")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("u-velocity")
axs[1].contourf(data["v"]["x"], data["v"]["y"], data["v"]["val"], 128, vmin=-0.6, vmax=0.6, cmap="cividis")
cs = axs[1].contour(data["v"]["x"], data["v"]["y"], data["v"]["val"], levels, colors="k", linewidths=0.5)
axs[1].set_aspect("equal", "box")
axs[1].set_xlabel("x")
axs[1].set_title("v-velocity")
cbar = fig.colorbar(cm.ScalarMappable(colors.Normalize(-0.6, 0.6), "cividis"), ax=axs)
cbar.add_lines(cs)
fig.suptitle(r"Taylor-Green vortex, Re=100, $T_{sim}=32$, PetIBM")
fig.savefig(rootdir.joinpath("figures", "tgv-petibm-contour-t32.png"), bbox_inches="tight", dpi=166)

# =======================================================
# plot contour for TGV 2D Re100 from PINN A100_8 at T=32
# =======================================================
data = {key: contour_data_reader(modulusdir.joinpath("a100_8", "outputs"), key) for key in ["u", "v"]}
levels = numpy.linspace(-0.6, 0.6, 13)  # contour levels
fig, axs = pyplot.subplots(1, 2, sharex=False, sharey=True, figsize=(6, 3), dpi=166)
axs[0].contourf(data["u"]["x"], data["u"]["y"], data["u"]["val"], 128, vmin=-0.6, vmax=0.6, cmap="cividis")
axs[0].contour(data["u"]["x"], data["u"]["y"], data["u"]["val"], levels, colors="k", linewidths=0.5)
axs[0].set_aspect("equal", "box")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("u-velocity")
axs[1].contourf(data["v"]["x"], data["v"]["y"], data["v"]["val"], 128, vmin=-0.6, vmax=0.6, cmap="cividis")
cs = axs[1].contour(data["v"]["x"], data["v"]["y"], data["v"]["val"], levels, colors="k", linewidths=0.5)
axs[1].set_aspect("equal", "box")
axs[1].set_xlabel("x")
axs[1].set_title("v-velocity")
cbar = fig.colorbar(cm.ScalarMappable(colors.Normalize(-0.6, 0.6), "cividis"), ax=axs)
cbar.add_lines(cs)
fig.suptitle(r"Taylor-Green vortex, Re=100, $T_{sim}=32$, PINN")
fig.savefig(rootdir.joinpath("figures", "tgv-pinn-contour-t32.png"), bbox_inches="tight", dpi=166)

# =====================================================================
# total losses of our networks to show our training processes converged
# =====================================================================
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
for ngpus in [1, 2, 4, 8]:
    data = read_tensorboard_data(modulusdir.joinpath(f"a100_{ngpus}", "outputs"))
    ax.semilogy(data["step"], data["loss"].ewm(span=10).mean(), label=f"{ngpus} A100")
ax.set_xlabel("Iteration")
ax.set_ylabel("Total loss")
ax.legend(loc=0)
fig.suptitle("Convergence history: total loss v.s. iteration")
fig.savefig(rootdir.joinpath("figures", "tgv-pinn-training-convergence.png"), bbox_inches="tight", dpi=166)

# ============================================================
# l2 error v.s. simulation time
# ============================================================
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)

data = pandas.read_csv(modulusdir.joinpath("a100_1", "outputs", "sim-time-errors.csv"), index_col=0, header=[0, 1])
ax.semilogy(data.index, data[("l2norm", "u")], label="PINN, 1 A100", alpha=0.6)

data = pandas.read_csv(modulusdir.joinpath("a100_8", "outputs", "sim-time-errors.csv"), index_col=0, header=[0, 1])
ax.semilogy(data.index, data[("l2norm", "u")], label="PINN, 8 A100", alpha=0.6)

data = pandas.read_csv(petibmdir.joinpath("16x16", "errors.csv"), index_col=0, header=[0, 1])
ax.semilogy(data.index, data[("l2norm", "u")], label="PetIBM, 16x16", ls="--", alpha=0.6)

data = pandas.read_csv(petibmdir.joinpath("32x32", "errors.csv"), index_col=0, header=[0, 1])
ax.semilogy(data.index, data[("l2norm", "u")], label="PetIBM, 32x32", ls="--", alpha=0.6)

data = pandas.read_csv(petibmdir.joinpath("1024x1024", "errors.csv"), index_col=0, header=[0, 1])
ax.semilogy(data.index, data[("l2norm", "u")], label="PetIBM, 1024x12024", ls="--", alpha=0.6)

ax.set_xlabel(r"t")
ax.set_ylabel(r"$L_2$-norm")
ax.set_ylim(1e-7, 2)
ax.set_yticks(numpy.power(10., numpy.arange(-7, 1)))
ax.legend(loc=0, ncol=3)
fig.suptitle(r"$L_2$ error in $u$ v.s. simulation time")
fig.savefig(rootdir.joinpath("figures", "tgv-sim-time-errors.png"), bbox_inches="tight", dpi=166)

# ============================================================
# error norms of velocity with respect to run time (wall time)
# ============================================================
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)

# load Modulus data
data = pandas.read_csv(modulusdir.joinpath("a100_8", "outputs", "norms.csv"), header=[0, 1, 2], index_col=0)
data = data.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=1)
data = data.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=2)

lines = [
    ax.loglog(data["runtime"], data[("l2norm", "u", "2.0")].ewm(span=10).mean(), label=r"$t=2$", alpha=0.8)[0],
    ax.loglog(data["runtime"], data[("l2norm", "u", "8.0")].ewm(span=10).mean(), label=r"$t=8$", alpha=0.8)[0],
    ax.loglog(data["runtime"], data[("l2norm", "u", "32.0")].ewm(span=10).mean(), label=r"$t=32$", alpha=0.8)[0]
]

legend1 = ax.legend(handles=lines, loc=2, bbox_to_anchor=(0.16, 1), ncol=1, fontsize=9, title="PINN")

# petibm performance data
data = pandas.read_csv(petibmdir.joinpath("perfs.csv"), index_col=0)
data.columns = pandas.MultiIndex.from_product([data.columns, [""], [""]])

# load PetIBM data of error v.s. time in simulation; load the error at T_{sim} = 32.0
for job in [f"{res}x{res}" for res in [16, 32, 64, 128, 256, 512, 1024]]:
    errors = pandas.read_csv(petibmdir.joinpath(job, "errors.csv"), header=[0, 1], index_col=0)
    for time in [2, 8, 32]:
        data.loc[job, ("l2norm", "u", time)] = errors.loc[time, ("l2norm", "u")]

lines = [
    ax.loglog(data["performance"], data[("l2norm", "u", 2)], label=r"$t=2$", alpha=0.8, ls="--")[0],
    ax.loglog(data["performance"], data[("l2norm", "u", 8)], label=r"$t=8$", alpha=0.8, ls="--")[0],
    ax.loglog(data["performance"], data[("l2norm", "u", 32)], label=r"$t=32$", alpha=0.8, ls="--")[0]
]

legend2 = ax.legend(handles=lines, loc=2, bbox_to_anchor=(0, 1), ncol=1, fontsize=9, title="PetIBM")

ax.set_xlabel("Wall time (seconds)")
ax.set_ylabel(r"$L_2$-norm")
ax.add_artist(legend1)
ax.add_artist(legend2)
fig.suptitle(r"$L_2$ error in $u$ v.s. wall time")
fig.savefig(rootdir.joinpath("figures", "tgv-run-time-errors.png"), bbox_inches="tight", dpi=166)

# ================================================================
# wall time per 1000 iterations v.s. number of GPUs (weak scaling)
# ================================================================
fig, ax = pyplot.subplots(1, 1, figsize=(4, 2), dpi=166)

perf = []
for ngpus in [1, 2, 4, 8]:
    data = pandas.read_csv(modulusdir.joinpath(f"a100_{ngpus}", "outputs", "norms.csv"), header=[0, 1, 2], index_col=0)
    data = data.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=1)
    data = data.rename(columns=lambda inp: "" if "Unnamed" in inp else inp, level=2)
    perf.append(data["runtime"].iloc[-1] / data.index[-1] * 1000)

ax.bar(range(4), perf, label="Time per 1k iterations")
ax.set_xticks(range(4), [1, 2, 4, 8])
ax.set_xlabel("Numbers of GPUs (NVIDIA A100-80GB)")
ax.set_ylabel("Seconds per 1000 iterations")
ax.legend(loc=3, bbox_to_anchor=(0, 0))

eff = [perf[0] * 100 / val for val in perf]

axright = ax.twinx()
axright.plot(range(4), eff, "k.-", lw=2, markersize=15, label="Efficiency")
axright.set_ylabel(r"Parallel efficiency ($\%$)")
axright.set_ylim(0, 110)
axright.legend(loc=3, bbox_to_anchor=(0, 0.2))

fig.suptitle("Weak scaling performance, PINN")
fig.savefig(rootdir.joinpath("figures", "tgv-weak-scaling.png"), bbox_inches="tight", dpi=166)

# print the table
print(perf)
print(eff)

# =====================================================================
# PDE residuals v.s. number of iterations
# =====================================================================
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)
for ngpus in [1, 2, 4, 8]:
    data = read_tensorboard_data(modulusdir.joinpath(f"a100_{ngpus}", "outputs"))
    ax.semilogy(data["step"], data["momem_x_res"].ewm(span=10).mean(), label=f"{ngpus} A100")
ax.set_xlabel("Iteration")
ax.set_ylabel("Total loss")
ax.legend(loc=0)
fig.suptitle("Convergence history: total loss v.s. iteration")
fig.savefig(rootdir.joinpath("figures", "tgv-pinn-monitor-residuals.png"), bbox_inches="tight", dpi=166)

# =====================================================================
# kinetic energy v.s. simulation time
# =====================================================================
fig, ax = pyplot.subplots(1, 1, figsize=(6, 3), dpi=166)

# data = numpy.linspace(0., 100., 101)
# ax.semilogy(data, numpy.exp(-4.*0.01*data), label="Analytical")

data = pandas.read_csv(modulusdir.joinpath("a100_8", "outputs", "sim-time-kinetic-energy.csv"))
ans = numpy.pi**2 * numpy.exp(-4*0.01*data["time"])
err = abs((data["energy"] - ans) / ans)
ax.semilogy(data["time"], err, label="PINN, 8 A100")

data = pandas.read_csv(petibmdir.joinpath("32x32", "energy.csv"))
ans = numpy.pi**2 * numpy.exp(-4*0.01*data["time"])
err = abs((data["energy"] - ans) / ans)
ax.semilogy(data["time"], err, label="PetIBM, 32x32")

data = pandas.read_csv(petibmdir.joinpath("128x128", "energy.csv"))
ans = numpy.pi**2 * numpy.exp(-4*0.01*data["time"])
err = abs((data["energy"] - ans) / ans)
ax.semilogy(data["time"], err, label="PetIBM, 128x128")

data = pandas.read_csv(petibmdir.joinpath("1024x1024", "energy.csv"))
ans = numpy.pi**2 * numpy.exp(-4*0.01*data["time"])
err = abs((data["energy"] - ans) / ans)
ax.semilogy(data["time"], err, label="PetIBM, 1024x1024")

ax.set_xlabel("Time in simulation")
ax.set_ylabel("Normalized kinetic energy")
ax.legend(loc=0)
fig.suptitle("Total kinetic energy v.s. time")
fig.savefig(rootdir.joinpath("figures", "tgv-kinetic-energy.png"), bbox_inches="tight", dpi=166)
