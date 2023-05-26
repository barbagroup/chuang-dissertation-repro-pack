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
import pandas
import scipy.interpolate
from h5py import File as h5open
from matplotlib import pyplot

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def read_validation_data(petibmdir):
    """Read validation data (surface pressure).
    """
    sen2009 = pandas.read_csv(
        petibmdir.joinpath("validation", "sen_et_al_2009.csv"),
        header=0, names=["degrees", "values"]
    )
    sen2009["degrees"] = 180. - sen2009["degrees"]  # they defined 0 from left

    park1998 = pandas.read_csv(
        petibmdir.joinpath("validation", "park_et_al_1998.csv"),
        header=None, names=["degrees", "values"]
    )
    park1998["degrees"] = 180. - park1998["degrees"]  # they defined 0 from left

    grove1964 = pandas.read_csv(
        petibmdir.joinpath("validation", "grove_et_al_1964.csv"),
        header=0, names=["degrees", "values"]
    )
    grove1964["degrees"] = 180. - grove1964["degrees"]  # they defined 0 from left

    # use the mean pressure as the reference pressure
    with h5open(petibmdir.joinpath("output", "0002000.h5"), "r") as h5file:
        pref = h5file["p"][...].mean()

    # get probe data
    with h5open(petibmdir.joinpath("output", "probe-p.h5"), "r") as h5file:
        x = h5file["mesh/x"][...]
        y = h5file["mesh/y"][...]
        ids = numpy.argsort(h5file["mesh/IS"][...].flatten())
        p = h5file[f"p/{max(h5file['p'].keys(), key=float)}"][...].flatten()[ids].reshape(y.size, x.size)
        p -= pref   # correct the pressure

    # initialize final dataframe
    petibm = pandas.DataFrame(data={
        "degrees": numpy.linspace(0., numpy.pi, 361),
        "values": numpy.zeros(361)
    })

    # create an interpolater for surface pressure from PetIBM
    probe_interp = scipy.interpolate.RectBivariateSpline(x, y, p.T)

    # note for PetIBM's diffused immersed boundary, we use r+3dx as the cylinder surface
    surfx = (0.5 + 4 * (x[1] - x[0])) * numpy.cos(petibm.degrees.array)
    surfy = (0.5 + 4 * (x[1] - x[0])) * numpy.sin(petibm.degrees.array)

    # interpolation
    petibm["values"] = probe_interp(surfx, surfy, grid=False) * 2  # Cp = (p-p_ref) * 2

    # convert from radius to degrees
    petibm["degrees"] = petibm["degrees"] * 180 / numpy.pi

    return {"sen2009": sen2009, "park1998": park1998, "grove1964": grove1964, "petibm": petibm}


def plot_surface_pressure(workdir, figdir, refs):
    """Plot surface pressure coefficients.
    """

    cases = {
        "nl6-nn512-npts25600-large-cycle-steady": "Steady PINN solver",
        "nl6-nn512-npts25600-large-cycle-unsteady": "Unsteady PINN solver",
    }

    fig = pyplot.figure(figsize=(3.75, 2.4))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    # reference: petibm
    ax.plot(
        refs["petibm"]["degrees"], refs["petibm"]["values"],
        c="k", ls="-", lw=1, zorder=1,
        label="PetIBM"
    )

    # other references
    ax.plot(
        refs["grove1964"]["degrees"], refs["grove1964"]["values"],
        ls="none", marker="s", ms=4, alpha=0.8, zorder=2,
        label="Grove et al., 1964",
    )

    ax.plot(
        refs["sen2009"]["degrees"], refs["sen2009"]["values"],
        ls="none", marker="o", mfc="none", ms=4, alpha=0.8, zorder=2,
        label="Sen et al., 2009"
    )

    ax.plot(
        refs["park1998"]["degrees"], refs["park1998"]["values"],
        ls="none", marker="^", mfc="none", ms=4, alpha=0.8, zorder=2,
        label="Park et al., 1998"
    )

    kwargs = {"lw": 1, "zorder": 3}
    lss = ["-.", "--", ":"]
    for i, (case, label) in enumerate(cases.items()):
        with h5open(workdir.joinpath("outputs", f"{case}-raw.h5"), "r") as h5file:
            thetas = h5file["surfp/degrees"][...]
            values = h5file["surfp/cp"][...]

        ax.plot(thetas, values, label=label, ls=lss[i], **kwargs)

    ax.set_xlabel(r"Degree from $+x$ axis")
    ax.set_xlim(0, 180.)

    ax.set_ylabel(r"Pressure coefficient, $C_p$")
    ax.set_ylim(-1.2, 1.5)

    ax.legend(loc=0)

    # save
    fig.savefig(figdir.joinpath("surface-pressure.png"))


if __name__ == "__main__":

    # point workdir to the correct folder
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _petibmdir = pathlib.Path(__file__).resolve().parents[3].joinpath("petibm", "cylinder-2d-re40")
    _figdir = _workdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    _refs = read_validation_data(_petibmdir)
    plot_surface_pressure(_workdir, _figdir, _refs)
