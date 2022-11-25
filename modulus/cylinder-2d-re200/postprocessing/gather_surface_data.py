#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Gather data on the cylinder surface.
"""
# %% preprocessing
import sys
import pathlib
import numpy
import torch
import h5py
from modulus.key import Key
from modulus.graph import Graph
from omegaconf import OmegaConf

# find helpers and locate top project folder
for _projdir in pathlib.Path(__file__).resolve().parents:
    if _projdir.joinpath("helpers").is_dir():  # _projdir is currently the `modulus` folder
        sys.path.insert(0, str(_projdir))
        from helpers.utils import get_graph_from_checkpoint
        _projdir = _projdir.parent
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


# %% 1D delta function
def dirac(r, dr):
    """Descretized Dirac function.
    """

    r = numpy.abs(r)
    out = numpy.zeros_like(r)

    idx = (r <= dr / 2.)
    out[idx] = (1 + numpy.sqrt(1 - 3 * (r[idx] / dr)**2)) / (3 * dr)

    idx = ((r >= dr / 2.) & (r <= dr * 3. / 2.))
    out[idx] = (5 - 3 * r[idx] / dr - numpy.sqrt(1 - 3 * (1 - r[idx] / dr)**2)) / (6 * dr)

    return out


# %% get_petibm_data
def get_petibm_data(casedir, fields, time, surfx, surfy, cache=None):
    """Get surface data from PetIBM.
    """

    # get config file
    cfg = OmegaConf.load(casedir.joinpath("config.yaml"))

    # convert time to filename index
    fno = int(time/cfg.parameters.dt+0.5)

    print(f"Predicting PetIBM surface data at t={time}, fno={fno}")
    preds = {}

    if cache is None:
        cache = {}

    # get field one by one
    for field in fields:

        # resolve naming discrepency
        field = "wz" if field == "vorticity_z" else field

        # get field data
        with h5py.File(casedir.joinpath("output", f"{fno:07d}.h5"), "r") as h5file:
            data = h5file[f"{field}"][...]

        # get coordinates
        with h5py.File(casedir.joinpath("output", "grid.h5"), "r") as h5file:
            x = h5file[f"{field}/x"][...]
            y = h5file[f"{field}/y"][...]

        # restore field name
        field = "vorticity_z" if field == "wz" else field

        if field not in cache:
            cache[field] = {}

            # calculate dx dy
            dx = x[(x <= surfx.max()) & (x >= surfx.min())]
            dx = (dx[1:] - dx[:-1]).mean()
            dy = y[(y <= surfy.max()) & (y >= surfy.min())]
            dy = (dy[1:] - dy[:-1]).mean()

            # create Cartesian grid
            x, y = numpy.meshgrid(x, y)

            # get indices that are inside the support windows
            cache[field]["idx"] = (
                (x <= (surfx.max() + 3 * dx)) &
                (x >= (surfx.min() - 3 * dx)) &
                (y <= (surfy.max() + 3 * dy)) &
                (y >= (surfy.min() - 3 * dy))
            )

            # get grid point around cylinder
            x = x[cache[field]["idx"]].reshape(-1, 1)
            y = y[cache[field]["idx"]].reshape(-1, 1)

            cache[field]["delta"] = dirac(x-surfx, dx) * dirac(y-surfy, dy)
            cache[field]["dxdy"] = dx * dy

        # interpolate to surface using the dirac function
        preds[field] = (
            cache[field]["delta"] * data[cache[field]["idx"]].reshape(-1, 1)
        ).sum(axis=0) * cache[field]["dxdy"]

    return preds


# %% get_pinn_data
def get_pinn_data(model, fields, time, surfx, surfy):
    """Get surface data from PINN.
    """

    # torch version of gridlines; reshape to N by 1
    kwargs = {
        "dtype": next(model.parameters()).dtype,
        "device": "cpu", "requires_grad": True}

    # convert numpy coordinates to torch
    surfx = torch.tensor(surfx.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    surfy = torch.tensor(surfy.reshape(-1, 1), **kwargs)  # pylint: disable=no-member

    # predict and save
    print(f"Predicting PINN surface data at t={time}")
    preds = model({
        "x": surfx, "y": surfy,
        "t": torch.full_like(surfx, time, requires_grad=True)
    })

    preds = {
        field: val.detach().cpu().numpy().flatten()
        for field, val in preds.items()
    }

    return preds


# %% get_pinn_model
def get_pinn_model(casedir, step, fields):
    """Get PINN model.
    """

    # get configuration
    cfg = OmegaConf.load(casedir.joinpath("outputs", ".hydra", "config.yaml"))
    cfg.custom.scaling = False  # no scaling for re200

    # get computational graph and network model
    graph, _ = get_graph_from_checkpoint(
        cfg, casedir.joinpath("outputs", f"flow-net.pth.{step}"),
        dim=2, unsteady=True, mtype="raw", device="cpu"
    )

    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    return model


# %% main
if __name__ == "__main__":

    # folders and paths
    _pinndir = _projdir.joinpath("modulus", "cylinder-2d-re200", "nl6-nn512-npts6400-unsteady-petibm")
    _petibmdir = _projdir.joinpath("petibm", "cylinder-2d-re200")

    # data file
    _pinndir.parent.joinpath("data").mkdir(exist_ok=True)
    _datafile = _pinndir.parent.joinpath("data", "surface_data.h5")

    # %% case configuration
    _fields = ["u", "v", "p", "vorticity_z"]
    _device = "cpu"
    _times = list(float(val) for val in numpy.linspace(135, 145, 21))
    _nths = 720  # number of cells on cylinder surface
    _radius = 0.5
    _thetas = numpy.linspace(0., numpy.pi*2., _nths, endpoint=False)
    _surfx = _radius * numpy.cos(_thetas)
    _surfy = _radius * numpy.sin(_thetas)

    # save input data
    with h5py.File(_datafile, "w") as _dset:
        _dset.create_dataset("x", _surfx.shape, _surfx.dtype, _surfx, compression="gzip")
        _dset.create_dataset("y", _surfy.shape, _surfy.dtype, _surfy, compression="gzip")
        _dset.create_dataset("theta", _thetas.shape, _thetas.dtype, _thetas, compression="gzip")
        _dset.create_dataset("times", (len(_times),), numpy.float64, _times, compression="gzip")

    # %% get PINN model
    _model = get_pinn_model(_pinndir, 1000000, _fields)

    # %% predict and save
    _cache = {}
    for _time in _times:

        # get petibm data
        _data = get_petibm_data(_petibmdir, _fields, _time, _surfx, _surfy, _cache)

        for _field, _val in _data.items():
            with h5py.File(_datafile, "r+") as _dset:
                _dset.create_dataset(
                    f"petibm/{_field}/{_time}", _val.shape, _val.dtype,
                    _val, compression="gzip"
                )

        # get pinn data
        _data = get_pinn_data(_model, _fields, _time, _surfx, _surfy)

        for _field, _val in _data.items():
            with h5py.File(_datafile, "r+") as _dset:
                _dset.create_dataset(
                    f"pinn/{_field}/{_time}", _val.shape, _val.dtype,
                    _val, compression="gzip"
                )
