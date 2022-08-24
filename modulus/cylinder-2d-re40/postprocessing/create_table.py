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
from h5py import File as h5open
from matplotlib import pyplot

# unified figure style
pyplot.style.use(pathlib.Path(__file__).resolve().parents[3].joinpath("resources", "figstyle"))


template = \
"""\\begin{{table}}[H]
    \\singlespacing
    \\begin{{threeparttable}}[b]
        \\begin{{tabular}}{{lccc}}
            \\toprule
            & $C_D$ & $C_{{D_p}}$ & $C_{{D_f}}$ \\
            \\midrule
            $(6, 512, 6400)$, steady & {:.2f} & {:.2f} & {:.2f} \\
            $(6, 512, 6400)$, unsteady & {:.2f} & {:.2f} & {:.2f} \\
            $(6, 512, 6400)$, large cycle, steady & {:.2f} & {:.2f} & {:.2f} \\
            $(6, 512, 6400)$, large cycle, unsteady & {:.2f} & {:.2f} & {:.2f} \\
            $(6, 512, 25600)$, large cycle, steady & {:.2f} & {:.2f} & {:.2f} \\
            $(6, 512, 25600)$, large cycle, unsteady & {:.2f} & {:.2f} & {:.2f} \\
            PetIBM & 1.63 & 1.02 & 0.61 \\
            Rosetti et al., 2012\\cite{{rosetti_urans_2012}}\\tnote{{1}} & 1.74\\pm 0.09 & n/a & n/a \\\\
            Rosetti et al., 2012\\cite{{rosetti_urans_2012}}\\tnote{{2}} & 1.61 & n/a & n/a \\\\
            Sen et al., 2009\\cite{{sen_steady_2009}}\\tnote{{2}} & 1.51 & n/a & n/a \\\\
            Park et al., 1988\\cite{{park_numerical_1998}}\\tnote{{2}} & 1.51 & 0.99 & 0.53 \\\\
            Tritton, 1959\\cite{{tritton_experiments_1959}}\\tnote{{1}} & 1.48--1.65 & n/a & n/a \\\\
            Grove et al., 1964\\cite{{grove_experimental_1964}}\\tnote{{1}} & n/a & 0.94 & n/a \\\\
            \\bottomrule
        \\end{{tabular}}%
        \\begin{{tablenotes}}
            \\footnotesize
            \\item [1] Experimental result
            \\item [2] Simulation result
        \\end{{tablenotes}}
        \\caption[Validation of drag coefficients]{{%
            Validation of drag coefficients.%
            $C_D$, $C_{{D_p}}$, and $C_{{D_f}}$ denote the coefficients of total drag, pressure drag, %
            and friction drag, respectively.%
        }}%
        \\label{{table:cylinder-2d-re40-comparison-cd}}
    \\end{{threeparttable}}
\\end{{table}}%\n"""


def create_table(workdir, tabledir):
    """Plot a field of all cases at a single time.
    """

    cases = [
        "nl6-nn512-npts6400-steady",
        "nl6-nn512-npts6400-unsteady",
        "nl6-nn512-npts6400-large-cycle-steady",
        "nl6-nn512-npts6400-large-cycle-unsteady",
        "nl6-nn512-npts25600-large-cycle-steady",
        "nl6-nn512-npts25600-large-cycle-unsteady",
    ]

    values = []

    # add lines from each case
    for case in cases:
        with h5open(workdir.joinpath("outputs", f"{case}-raw.h5"), "r") as h5file:
            if h5file.attrs["unsteady"]:
                cd = h5file["coeffs/cd"][...]
                cd = (cd[-1] + cd[-2]) / 2.
                cdp = h5file["coeffs/cdp"][...]
                cdp = (cdp[-1] + cdp[-2]) / 2.
                cdv = h5file["coeffs/cdv"][...]
                cdv = (cdv[-1] + cdv[-2]) / 2.
            else:
                cd = float(h5file["coeffs/cd"][...])
                cdp = float(h5file["coeffs/cdp"][...])
                cdv = float(h5file["coeffs/cdv"][...])

        values.extend([cd, cdp, cdv])

    table = template.format(*values)

    with open(tabledir.joinpath("drag-lift-coeff.tex"), "w") as fobj:
        fobj.write(table)


if __name__ == "__main__":

    # point workdir to the correct folder
    _workdir = pathlib.Path(__file__).resolve().parents[1]
    _tabledir = _workdir.joinpath("tables")
    _tabledir.mkdir(parents=True, exist_ok=True)
    create_table(_workdir, _tabledir)
