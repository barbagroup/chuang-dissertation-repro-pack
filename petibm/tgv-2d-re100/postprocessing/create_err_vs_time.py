#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create error vs walltime for PetIBM.
"""
import re
import pathlib
import numpy
import h5py
from matplotlib import pyplot

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def analytical_solution(x, y, t, nu, field, V0=1., L=1., rho=1.):
    """Get analytical solution of 2D TGV.
    """
    if field == "u":
        return V0 * numpy.cos(x/L) * numpy.sin(y/L) * numpy.exp(-2.*nu*t/L**2)
    if field == "v":
        return - V0 * numpy.sin(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2)
    if field == "p":
        return - rho * V0**2 * numpy.exp(-4.*nu*t/L**2) * \
            (numpy.cos(2.*x/L) + numpy.cos(2.*y/L)) / 4.
    if field in ["wz", "vorticity_z"]:
        return - 2. * V0 * numpy.cos(x/L) * numpy.cos(y/L) * numpy.exp(-2.*nu*t/L**2) / L
    if field == "KE":  # kinetic energy
        return numpy.pi**2 * L**2 * V0**2 * rho * numpy.exp(-4.*nu*t/L**2)
    if field == "KEDR":  # kinetic energy dissipation rate
        return 4. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)
    if field == "enstrophy":  # enstrophy
        return 2. * numpy.pi**2 * V0**2 * nu * rho * numpy.exp(-4.*nu*t/L**2)

    raise ValueError(f"Unknown field: {field}")


def get_data(workdir, casename):
    """Get data of a case.
    """
    outdir = workdir.joinpath(casename, "output")
    gridfile = outdir.joinpath("grid.h5")
    solns = list(outdir.glob("*.h5"))
    solns.remove(gridfile)
    solns.remove(outdir.joinpath("0000000.h5"))

    # gridlines
    with h5py.File(gridfile, "r") as h5file:
        x = h5file["u/x"][...]
        y = h5file["u/y"][...]

    # grid
    x, y = numpy.meshgrid(x, y)

    err = 0.
    for soln in solns:
        with h5py.File(soln, "r") as h5file:
            t = h5file["p"].attrs["time"]
            err += (numpy.abs(h5file["u"][...]-analytical_solution(x, y, t, 0.01, "u"))**2).sum()

    err /= (len(solns) * x.size)
    err = numpy.sqrt(err)

    # get run time
    logfile = max(outdir.joinpath("logs").glob("*.log"), key=lambda inp: int(inp.stem))

    with open(logfile, "r") as fobj:
        for line in fobj:
            result = re.search(r"^Time\s+\(sec\):\s+(\d\.\d+e\+\d+)\s", line)
            if result is not None:
                time = float(result.group(1))
                break
        else:
            raise RuntimeError(f"Couldn't find run time in {logfile}")

    # infer the number of time steps
    nt = int(logfile.stem)

    # return err, time, x.size * nt
    return err, time, (x.size * nt)**(1./3.), x.size


def create_plot(workdir, cases, figdir):
    """Plot.
    """

    errs = []
    walltimes = []
    ncells = []
    for case in cases:
        err, walltime, ncell, _ = get_data(workdir, case)
        errs.append(err)
        walltimes.append(walltime)
        ncells.append(ncell)

    fig = pyplot.figure(figsize=(6.5, 3))
    fig.suptitle(r"PetIBM: spatial-temporal error of $u$")
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Error v.s. run time")
    ax1.set_xlabel("Run time (hours)")
    ax1.set_ylabel(r"Spatial-temporal $L_2$ error of $u$")
    ax1.grid()
    ax1.loglog(walltimes, errs)

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.set_title("Order of error convergence")
    ax2.set_xlabel(r"$\sqrt[3]{\mathrm{Number\ of\ spatial-temporal\ cells}}$")
    ax2.grid()
    ax2.loglog(ncells, errs, label="Simulation error")
    ax2.loglog([ncells[2], ncells[2]*4], [errs[2]*0.6, errs[2]*0.6/16], label="2nd-order ref.", ls="--")
    ax2.legend(loc=0)

    # save
    fig.savefig(figdir.joinpath("tgv-2d-re100-err-u.png"))


if __name__ == "__main__":
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _figdir = pathlib.Path(__file__).resolve().parents[2].joinpath("figures")
    _figdir.mkdir(exist_ok=True)

    _cases = [f"{2**i}x{2**i}" for i in range(4, 11)]
    create_plot(_workdir, _cases, _figdir)
