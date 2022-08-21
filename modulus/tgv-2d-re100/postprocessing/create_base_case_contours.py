#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
"""
import pathlib
import numpy
from h5py import File as h5open
from matplotlib import pyplot
from matplotlib import colors
from matplotlib import ticker

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


def create_base_case_contours(outdir, figdir, arch, time):
    """Plot contours.
    """

    nl, nn, nbs = arch
    with h5open(outdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
        coords = (h5file["field/x"][...], h5file["field/y"][...])

        vals = {
            r"$u$": h5file[f"field/{time}/u"][...],
            r"$v$": h5file[f"field/{time}/v"][...],
            r"$p$": h5file[f"field/{time}/p"][...],
            r"$\omega_z$": h5file[f"field/{time}/vorticity_z"][...],
        }

        errs = {
            r"$u$": h5file[f"field/{time}/err-u"][...],
            r"$v$": h5file[f"field/{time}/err-v"][...],
            r"$p$": h5file[f"field/{time}/err-p"][...],
            r"$\omega_z$": h5file[f"field/{time}/err-vorticity_z"][...],
        }

    # use analytical solutions for levels
    lvls = {  # calculate solutions firstj
        r"$u$": numpy.cos(coords[0]) * numpy.sin(coords[1]) * numpy.exp(-2.*0.01*float(time)),
        r"$v$": - numpy.sin(coords[0]) * numpy.cos(coords[1]) * numpy.exp(-2.*0.01*float(time)),
        r"$p$": - numpy.exp(-4.*0.01*float(time)) * (numpy.cos(2.*coords[0]) + numpy.cos(2.*coords[1])) / 4.,
        r"$\omega_z$": - 2. * numpy.cos(coords[0]) * numpy.cos(coords[1]) * numpy.exp(-2.*0.01*float(time)),
    }

    # actually convert them to levels
    lvls = {field: numpy.linspace(val.min(), val.max(), 17) for field, val in lvls.items()}

    # normalizer for error contourf
    normerr = colors.LogNorm

    # re-cal. the pressure w/ the mean from analytical soln. as it is assumed to have a constant shift
    ptrue = - numpy.exp(-4.*0.01*float(time)) * (numpy.cos(2.*coords[0]) + numpy.cos(2.*coords[1])) / 4.
    vals[r"$p$"] = vals[r"$p$"] - vals[r"$p$"].mean()
    errs[r"$p$"] = abs(vals[r"$p$"] - ptrue)

    fig = pyplot.figure(figsize=(6.5, 9))
    fig.suptitle(rf"TGV 2D@$t={float(time)}$, $Re=100$, $(N_l, N_n, N_{{bs}})=({nl}, {nn}, {nbs})$")
    gs = fig.add_gridspec(4, 2)

    for i, ((field, err), val, lvl) in enumerate(zip(errs.items(), vals.values(), lvls.values())):
        # field values
        axf = fig.add_subplot(gs[i, 0])
        ct = axf.contourf(*coords, val, lvl, extend="both")
        axf.set_aspect("equal")
        axf.set_title(field)
        axf.set_ylabel(r"$y$")
        fig.colorbar(ct, ax=axf, extend="both")

        # errors
        axerr = fig.add_subplot(gs[i, 1])
        ct = axerr.contourf(*coords, err, 17, extend="both", norm=normerr(err.min(), err.max()))
        axerr.set_aspect("equal")
        axerr.set_title(f"Absolute error, {field}")
        fig.colorbar(ct, ax=axerr, format=ticker.LogFormatter(), extend="both")

        if i == 3:
            axf.set_xlabel(r"$x$")
            axerr.set_xlabel(r"$x$")

    figdir.joinpath("contours").mkdir(parents=True, exist_ok=True)
    pyplot.savefig(figdir.joinpath("contours", f"nl{nl}-nn{nn}-npts{nbs}-t{time}.png"))


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).parents[1]
    _outdir = _projdir.joinpath("outputs", "base-cases")
    _figdir = _projdir.joinpath("figures")
    _figdir.mkdir(parents=True, exist_ok=True)

    create_base_case_contours(_outdir, _figdir, (1, 32, 16384), "40.0")
    create_base_case_contours(_outdir, _figdir, (2, 32, 65536), "40.0")
    create_base_case_contours(_outdir, _figdir, (3, 256, 4096), "40.0")
