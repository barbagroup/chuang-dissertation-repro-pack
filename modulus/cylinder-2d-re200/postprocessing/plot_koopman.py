#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Koopman analysis (dynamic mode decomposition).
"""

# %% headers
import pathlib
import numpy
import h5py
from matplotlib import pyplot
from matplotlib import colors
from matplotlib import cm
from matplotlib import ticker

# find helpers and locate workdir
for _projdir in pathlib.Path(__file__).resolve().parents:
    if _projdir.joinpath("modulus").is_dir():
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


class CustomCBarFormat(ticker.ScalarFormatter):
    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        super().__init__(useOffset, useMathText, useLocale)
    
    def _set_format(self):
        # docstring inherited
        self.format = "%1.1f"


# %%
def plot_eigval_cmplx_dist(eigvals, energys, figdir):
    """Create plots.
    """
    solvers = ["PetIBM", "Data-driven PINN"]

    # sort from low to high in energy
    for i in range(2):
        idx = numpy.argsort(energys[i])
        energys[i] = energys[i][idx]
        eigvals[i] = eigvals[i][idx]

    # create an uniform color mapping
    norm = colors.Normalize(
        vmin=min(energys[0][0], energys[1][0]),
        vmax=max(energys[0][-2], energys[1][-2])
    )  # excluding the steady mode
    cmap = pyplot.get_cmap("inferno_r")

    fig = pyplot.figure(figsize=(3.75, 2.7))
    gs = fig.add_gridspec(2, 2, height_ratios=(10, 1))

    for i in range(2):
        cs = cmap(norm(energys[i][:-1]))
        ax = fig.add_subplot(gs[0, i])
        ax.add_artist(pyplot.Circle((0., 0.), 1, fc="gainsboro", ec="k", lw=0.3))
        ax.scatter(  # only plot non-time-averaged mode
            eigvals[i][:-1].real, eigvals[i][:-1].imag, s=10, c=cs,
            edgecolors="k", linewidth=0.2, alpha=0.9,
        )
        ax.scatter(
            eigvals[i][-1].real, eigvals[i][-1].imag, s=20, marker="*",
            c="crimson", alpha=0.9, ec="k", lw=0.2,
        )
        ax.annotate(
            "Steady mode\n" + rf"(strength: ${energys[i][-1]:.2f}$)",
            (eigvals[i][-1].real, eigvals[i][-1].imag), (0, 0.2),
            ha="center", va="center", fontsize=8,
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc3", "lw": 0.7}
        )
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(r"$\mathfrak{Re}(\lambda_j)$")

        if i == 0:
            ax.set_ylabel(r"$\mathfrak{Im}(\lambda_j)$")
        else:
            ax.get_yaxis().set_visible(False)
        ax.set_title(solvers[i])

    cax = fig.add_subplot(gs[1, :])
    fig.colorbar(
        cm.ScalarMappable(norm, cmap), cax=cax, label="Normalized mode strength",
        orientation="horizontal")

    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)

    fig.savefig(figdir.joinpath("koopman_eigenvalues_complex.png"), bbox_inches="tight")
    pyplot.close(fig)


# %%
def plot_mode_strengths(freqs, energys, dt, figdir):
    """Plot bar plot of energy v.s. frequencies (excluding time-averaged mode).

    Notes
    -----
    Assuming 1) freqs are non-negative and 2) freqs and energy are sorted in an decreasing order.
    """

    solvers = ["PetIBM", "Data-driven PINN"]
    fastest = 1. / (2 * dt)  # the possible max frequency

    for i in range(2):
        # retain only positive freqs and those <= 2
        idx = numpy.where((freqs[i] >= -1e-12) & (freqs[i] <= fastest))[0]
        freqs[i] = freqs[i][idx]
        energys[i] = energys[i][idx]

        # sort with decreasing energy
        idx = numpy.argsort(energys[i])[::-1]
        freqs[i] = freqs[i][idx]
        energys[i] = energys[i][idx]

    # create an uniform color mapping
    norm = colors.LogNorm(
        vmin=min(energys[0].min(), energys[1].min()),
        vmax=max(energys[0].max(), energys[1].max())
    )  # excluding the steady mode
    cmap = pyplot.get_cmap("YlOrRd")

    fig = pyplot.figure(figsize=(3.75, 1.875))
    gs = fig.add_gridspec(1, 2, width_ratios=(10, 10))

    for i in range(2):
        cs = cmap(norm(energys[i]))
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(solvers[i])
        ax.bar(freqs[i], energys[i], 0.05, color=cs, edgecolor="k", lw=0.2)
        ax.set_yscale("log")
        ax.set_xlim(-0.1, fastest + 0.1)
        ax.set_ylim(1e-4, 1.1)

        ax.set_xlabel(r"Strouhal number ($St$)")
        if i == 0:
            ax.set_ylabel("Normalized mode strength")
        else:
            ax.get_yaxis().set_ticklabels([])

    fig.savefig(figdir.joinpath("koopman_mode_strength.png"))
    pyplot.close(fig)


# %%
def plot_modes(h5grp, solver, figdir, npics=5):
    """Plot contours for modes.
    """

    names = {
        "u": r"$u$ velocity", "v": r"$v$ velocity", "p": "Pressure",
        "petibm": "PetIBM", "pinn": "Data-driven PINN"
    }

    lims = h5grp.file.attrs["lims"]
    energys = h5grp["energys"][...]
    freqs = h5grp["freqs"][...]
    eigvals = h5grp["eigvals"][...]

    # argument sort of energy from high to low
    indices = numpy.argsort(energys)[::-1]

    k = 0
    for j in range(len(freqs)):
        idx = indices[j]

        # ignore negative frequencies
        if freqs[idx] <= -1e-12:
            continue
        # > -1e-12 but < 0 -> just set it to zero
        elif freqs[idx] < 0:
            freqs[idx] = 0

        fig = pyplot.figure(figsize=(7.5, 4.5))
        gs = fig.add_gridspec(2, 3)

        for i, field in enumerate(["u", "v", "p"]):

            x, y = h5grp[f"{field}/x"][...], h5grp[f"{field}/y"][...]
            mask = h5grp[f"{field}/mask"][...]

            data = numpy.zeros_like(mask, dtype=h5grp[f"{field}/modes"].dtype)
            data[~mask] = h5grp[f"{field}/modes"][:, idx]
            data = numpy.ma.array(data, mask=mask)

            # real part
            ax = [fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])]
            ax[0].set_title(rf"{names[field]}, $\mathfrak{{Re}}$")
            ax[1].set_title(rf"{names[field]}, $\mathfrak{{Im}}$")

            cs = ax[0].contourf(x, y, data.real, 128, cmap="turbo")
            cbar = fig.colorbar(cs, ax=ax[0], format=CustomCBarFormat(True), orientation="horizontal")
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position("left")
            cbar.update_ticks()

            cs = ax[1].contourf(x, y, data.imag, 128, cmap="turbo")
            cbar = fig.colorbar(cs, ax=ax[1], format=CustomCBarFormat(True), orientation="horizontal")
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position("left")
            cbar.update_ticks()

            for m in range(2):
                ax[m].add_artist(pyplot.Circle((0., 0.), 0.5, color="w"))
                ax[m].set_xlim(*lims[:2])
                ax[m].set_ylim(*lims[2:])
                ax[m].set_aspect("equal", "box")

                if i == 0:
                    ax[m].set_ylabel(r"$y$")
                else:
                    ax[m].yaxis.set_visible(False)

            ax[0].xaxis.set_visible(False)
            ax[1].set_xlabel(r"$x$")

        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)

        fig.savefig(
            figdir.joinpath(f"koopman_{solver}_{k:03d}_st{freqs[idx]:.3f}.png"),
            bbox_inches="tight")
        pyplot.close()
        print(f"Done plotting the contour of mode St={freqs[idx]:.3f}")

        # update counter
        k += 1
        if k >= npics:
            break


# %% main function
if __name__ == "__main__":

    # directories and paths
    _outfile = pathlib.Path(__file__).resolve().parents[1].joinpath("outputs", "koopman.h5")
    _figdir = _projdir.joinpath("modulus", "cylinder-2d-re200", "figures", "koopman")
    _figdir.mkdir(exist_ok=True)

    # unified figure style
    pyplot.style.use(_projdir.joinpath("resources", "figstyle"))

    # plot eigenvalues and energys on an complex plane
    with h5py.File(_outfile, "r") as h5file:
        plot_eigval_cmplx_dist(
            [h5file[solver]["eigvals"][...] for solver in ["petibm", "pinn"]],
            [h5file[solver]["energys"][...] for solver in ["petibm", "pinn"]],
            _figdir
        )

    # %% plot strength distribution
    with h5py.File(_outfile, "r") as h5file:
        plot_mode_strengths(
            [h5file[solver]["freqs"][...] for solver in ["petibm", "pinn"]],
            [h5file[solver]["energys"][...] for solver in ["petibm", "pinn"]],
            h5file.attrs["dt"] * h5file.attrs["nsteps"],
            _figdir
        )

    # %% plot modes' contours
    with h5py.File(_outfile, "r") as h5file:
        plot_modes(h5file["petibm"], "petibm", _figdir, 5)
        plot_modes(h5file["pinn"], "pinn", _figdir, 5)
