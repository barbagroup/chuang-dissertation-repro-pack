#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Convert PetIBM original data files to a single HDF5 file.
"""
import pathlib
import numpy
import h5py


def get_petibm_data(workdir, destdir, radius):
    """Get an interior pointwise constraint for initial conditions (i.e., t=0)
    """

    out = {}
    for field in ["u", "v", "p"]:

        # gridlines are only related to fields and have nothing to do with times
        with h5py.File(datadir.joinpath("grid.h5"), "r") as dset:
            coords = numpy.meshgrid(dset[field]["x"], dset[field]["y"], indexing="xy")

        # number of points
        npts = coords[0].size

        # concatenate and expand to 4 columns for x, y, t, and field values
        coords = [coord.reshape(-1, 1) for coord in coords]
        data = numpy.concatenate(coords + [numpy.zeros_like(coords[0]), numpy.zeros_like(coords[0])], axis=1)

        # repeat x and y coords (140-125+1) times
        data = numpy.tile(data, (140-125+1, 1))

        # more PetIBM data from t=126 to t=140
        for i, time in enumerate(range(125, 141)):
            print(f"{field}-{time}")

            # time
            data[i*npts:(i+1)*npts, 2] = float(time)

            # field values
            with h5py.File(datadir.joinpath(f"{time*2*100:07d}.h5"), "r") as dset:
                data[i*npts:(i+1)*npts, 3] = dset[field][...].flatten()

        # only retain points outside of the cylinder and reshape to column vectors
        locs = (data[:, 0]**2 + data[:, 1]**2) >= radius**2
        data = data[locs]

        # shuffle dataset; this function change orders in-place
        numpy.random.shuffle(data)

        # write to a single HDF5 file
        print(f"saving to a HDF5 file: {destdir.joinpath('petibm.h5')}")
        fmode = "w" if field == "u" else "a"
        with h5py.File(destdir.joinpath("petibm.h5"), fmode) as h5file:
            h5file.create_dataset(f"{field}", data=data, compression="gzip")

        out[field] = data

    return out


if __name__ == "__main__":
    curdir = pathlib.Path(__file__).resolve().parent
    datadir = curdir.parents[2].joinpath("petibm", "cylinder-2d-re200", "output")
    results = get_petibm_data(datadir, curdir, 0.5)
