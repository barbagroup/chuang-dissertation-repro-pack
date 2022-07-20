#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D TGV Re100 analytical solution.
"""
import pathlib
import numpy
from matplotlib import pyplot


# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
    "figure.dpi": 166,
})


def analytical_solution(x, y, t, nu, field, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    if field == "u":
        return V0 * numpy.cos(x/L) * numpy.sin(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "v":
        return - V0 * numpy.sin(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2)
    elif field == "p":
        return - rho * V0**2 * numpy.exp(-4.*nu*t/L**2) * (numpy.cos(2.*x/L) + numpy.cos(2.*y/L)) / 4.
    elif field == "wz":
        return - 2. * V0 * numpy.cos(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2) / L
    elif field == "KE":  # kinetic energy
        return numpy.pi**2 * L**2 * V0**2 * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "KEDR":  # kinetic energy dissipation rate
        return 4. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    elif field == "enstrophy":  # enstrophy
        return 2. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    else:
        raise ValueError


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    figdir = root.joinpath("figures")
    figdir.mkdir(exist_ok=True)

    x = numpy.linspace(-numpy.pi, numpy.pi, 101)
    y = numpy.linspace(-numpy.pi, numpy.pi, 101)
    x, y = numpy.meshgrid(x, y)

    u40 = analytical_solution(x, y, numpy.full_like(x, 40.), 0.01, "u")
    v40 = analytical_solution(x, y, numpy.full_like(x, 40.), 0.01, "v")
    p40 = analytical_solution(x, y, numpy.full_like(x, 40.), 0.01, "p")
    wz40 = analytical_solution(x, y, numpy.full_like(x, 40.), 0.01, "wz")
    u80 = analytical_solution(x, y, numpy.full_like(x, 80.), 0.01, "u")
    v80 = analytical_solution(x, y, numpy.full_like(x, 80.), 0.01, "v")
    p80 = analytical_solution(x, y, numpy.full_like(x, 80.), 0.01, "p")
    wz80 = analytical_solution(x, y, numpy.full_like(x, 80.), 0.01, "wz")

    fig, axs = pyplot.subplots(2, 4, sharex=True, sharey=True, figsize=(6, 5))
    fig.suptitle(r"Analytical solutions for 2D Taylor-Green Vortex (TGV), $Re=100$")

    axs[0, 0].set_title(r"$t=40$, $u$", fontsize=9)
    cs = axs[0, 0].contourf(x, y, u40, 64)
    cbar = fig.colorbar(cs, ax=axs[0, 0], orientation="horizontal")
    pyplot.setp(axs[0, 0].get_xticklabels(), visible=False)
    axs[0, 0].set_ylabel(r"$y$", fontsize=8)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[0, 1].set_title(r"$t=40$, $v$", fontsize=9)
    cs = axs[0, 1].contourf(x, y, v40, 64)
    cbar = fig.colorbar(cs, ax=axs[0, 1], orientation="horizontal")
    pyplot.setp(axs[0, 1].get_xticklabels(), visible=False)
    pyplot.setp(axs[0, 1].get_yticklabels(), visible=False)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[0, 2].set_title(r"$t=40$, $p$", fontsize=9)
    cs = axs[0, 2].contourf(x, y, p40, 64)
    cbar = fig.colorbar(cs, ax=axs[0, 2], orientation="horizontal")
    pyplot.setp(axs[0, 2].get_xticklabels(), visible=False)
    pyplot.setp(axs[0, 2].get_yticklabels(), visible=False)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[0, 3].set_title(r"$t=40$, $\omega_z$", fontsize=9)
    cs = axs[0, 3].contourf(x, y, wz40, 64)
    cbar = fig.colorbar(cs, ax=axs[0, 3], orientation="horizontal")
    pyplot.setp(axs[0, 3].get_xticklabels(), visible=False)
    pyplot.setp(axs[0, 3].get_yticklabels(), visible=False)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[1, 0].set_title(r"$t=80$, $u$", fontsize=9)
    cs = axs[1, 0].contourf(x, y, u80, 64)
    cbar = fig.colorbar(cs, ax=axs[1, 0], orientation="horizontal")
    axs[1, 0].set_xlabel(r"$x$", fontsize=8)
    axs[1, 0].set_ylabel(r"$y$", fontsize=8)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[1, 1].set_title(r"$t=80$, $v$", fontsize=9)
    cs = axs[1, 1].contourf(x, y, v80, 64)
    cbar = fig.colorbar(cs, ax=axs[1, 1], orientation="horizontal")
    axs[1, 1].set_xlabel(r"$x$", fontsize=8)
    pyplot.setp(axs[1, 1].get_yticklabels(), visible=False)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[1, 2].set_title(r"$t=80$, $p$", fontsize=9)
    cs = axs[1, 2].contourf(x, y, p80, 64)
    cbar = fig.colorbar(cs, ax=axs[1, 2], orientation="horizontal")
    axs[1, 2].set_xlabel(r"$x$", fontsize=8)
    pyplot.setp(axs[1, 2].get_yticklabels(), visible=False)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    axs[1, 3].set_title(r"$t=80$, $\omega_z$", fontsize=9)
    cs = axs[1, 3].contourf(x, y, wz80, 64)
    cbar = fig.colorbar(cs, ax=axs[1, 3], orientation="horizontal")
    axs[1, 3].set_xlabel(r"$x$", fontsize=8)
    pyplot.setp(axs[1, 3].get_yticklabels(), visible=False)
    cbar.ax.tick_params(axis='x', labelsize=7, labelrotation=-90)
    cbar.ax.tick_params(axis='y', labelsize=7, labelrotation=-90)

    for i in range(2):
        for j in range(4):
            axs[i, j].tick_params(axis='x', labelsize=8)
            axs[i, j].tick_params(axis='y', labelsize=8)

    fig.savefig(figdir.joinpath("tgv-re100-analytical-demo.png"), bbox_inches="tight")
    pyplot.close(fig)
