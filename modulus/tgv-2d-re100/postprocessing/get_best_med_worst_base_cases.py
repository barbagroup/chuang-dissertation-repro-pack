#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
"""
import itertools
import pathlib
import numpy
from h5py import File as h5open


def get_best_med_worst_base_cases(outdir, nls, nns, nbss):
    """Get the cases with the best, median, and the worst errors.
    """
    bestl2norm = float("inf")
    best = None
    errs = []
    cases = []
    for nl, nn, nbs in itertools.product(nls, nns, nbss):
        with h5open(outdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}-raw.h5"), "r") as h5file:
            errs.append(float(h5file[f"sterrs/u/l2norm"][...]))
        cases.append((nl, nn, nbs))
    
    errs = numpy.array(errs)
    cases = numpy.array(cases, dtype=object)

    besterr = numpy.min(errs)
    bestconf = cases[numpy.argmin(errs)].tolist()
    worsterr = numpy.max(errs)
    worstconf = cases[numpy.argmax(errs)].tolist()
    mederr = numpy.median(errs)
    medconf = cases[numpy.where(errs == mederr)[0][0]].tolist()
    
    return (besterr, bestconf), (mederr, medconf), (worsterr, worstconf)


if __name__ == "__main__":
    _outdir = pathlib.Path(__file__).resolve().parents[1].joinpath("outputs", "base-cases")
    _nls = [1, 2, 3]
    _nns = [16, 32, 64, 128, 256]
    _nbss = [2**i for i in range(10, 17)]
    print(get_best_med_worst_base_cases(_outdir, _nls, _nns, _nbss))