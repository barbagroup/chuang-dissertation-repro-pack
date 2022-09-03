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
    x = {}
    y = {}
    with h5py.File(gridfile, "r") as h5file:
        for field in ["u", "v", "p"]:
            x[field] = h5file[f"{field}/x"][...]
            y[field] = h5file[f"{field}/y"][...]

    # grid
    for field in ["u", "v", "p"]:
        x[field], y[field] = numpy.meshgrid(x[field], y[field])

    err = {"u": 0.0, "v": 0.0, "p": 0.0}
    for soln in solns:
        with h5py.File(soln, "r") as h5file:
            t = h5file["p"].attrs["time"]
            for field in ["u", "v", "p"]:
                val = h5file[field][...]
                ans = analytical_solution(x[field], y[field], t, 0.01, field)
                val = val - val.mean() + ans.mean()
                err[field] += (numpy.abs(val-ans)**2).sum()

    err = {field: numpy.sqrt(val/(len(solns)*x[field].size)) for field, val in err.items()}
    print(x["p"].size, x["p"].shape)

    # get run time
    logfile = max(outdir.joinpath("logs").glob("*.log"), key=lambda inp: int(inp.stem))

    # infer the number of time steps
    nt = int(logfile.stem)

    # return err, time, x.size * nt
    return (err["u"], err["v"], err["p"]), \
        ((x["u"].size * nt)**(1./3.), (x["v"].size * nt)**(1./3.), (x["p"].size * nt)**(1./3.))


def create_plot(workdir, cases, figdir):
    """Plot.
    """

    errs = {"u": [], "v": [], "p": []}
    ncells = {"u": [], "v": [], "p": []}
    for case in cases:
        print(case)
        err, ncell = get_data(workdir, case)
        errs["u"].append(err[0])
        errs["v"].append(err[1])
        errs["p"].append(err[2])
        ncells["u"].append(ncell[0])
        ncells["v"].append(ncell[1])
        ncells["p"].append(ncell[2])

    fig = pyplot.figure(figsize=(6.5, 3.5))
    fig.suptitle(r"PetIBM, TGV 2D $Re=100$: grid convergence of spatial-temporal errors")
    gs = fig.add_gridspec(1, 3)

    for i, field in enumerate(["u", "v", "p"]):
        print(i, field)
        ax = fig.add_subplot(gs[i])
        ax.set_title(rf"Field: ${field}$")
        ax.set_xlabel(r"$\sqrt[3]{\mathrm{Total\ sp.-temp.\ pts}}$")

        if i == 0:
            ax.set_ylabel(r"Spatial-temporal $L_2$ error")

        ax.grid()
        ax.loglog(ncells[field], errs[field], marker=".", label="Simulation error")
        ax.loglog(
            [ncells[field][0], ncells[field][-1]],
            [errs[field][0]*0.6, errs[field][0]*0.6/(ncells[field][-1]/ncells[field][0])**2],
            label="2nd-order ref.", ls="--"
        )
        ax.legend(loc=0)

    # save
    fig.savefig(figdir.joinpath("petibm-tgv-2d-re100-convergence.png"))


if __name__ == "__main__":
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _figdir = pathlib.Path(__file__).resolve().parents[2].joinpath("figures")
    _figdir.mkdir(exist_ok=True)

    _cases = [f"{2**i}x{2**i}" for i in range(4, 11)]
    create_plot(_workdir, _cases, _figdir)
