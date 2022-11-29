#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Gather data inside a refined region.
"""
# %%
import sys
import pathlib
import numpy
import torch
import h5py
from modulus.key import Key
from modulus.graph import Graph
from omegaconf import OmegaConf

# find helpers and locate workdir
for _projdir in pathlib.Path(__file__).resolve().parents:
    if _projdir.joinpath("helpers").is_dir():
        sys.path.insert(0, str(_projdir))
        from helpers.utils import get_graph_from_checkpoint
        _projdir = _projdir.parent
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


# %% get_petibm_data
def get_petibm_data(casedir, fields, time, xlim, ylim, cache=None):
    """Get data from PetIBM.
    """

    # get config file
    cfg = OmegaConf.load(casedir.joinpath("config.yaml"))

    # convert time to filename index
    fno = int(time/cfg.parameters.dt+0.5)

    print(f"Predicting PetIBM vicinity data at t={time}, fno={fno}")
    preds = {}

    if cache is None:
        cache = {}

    # get field one by one
    for field in fields:

        # resolve naming discrepency
        field = "wz" if field == "vorticity_z" else field

        if field not in cache:

            # get coordinates
            with h5py.File(casedir.joinpath("output", "grid.h5"), "r") as h5file:
                x = h5file[f"{field}/x"][...]
                y = h5file[f"{field}/y"][...]

            xslc = numpy.arange(x.size)[(x >= xlim[0]) & (x <= xlim[1])]
            yslc = numpy.arange(y.size)[(y >= ylim[0]) & (y <= ylim[1])]
            xslc = slice(xslc.min(), xslc.max()+1)
            yslc = slice(yslc.min(), yslc.max()+1)

            cache[field] = (yslc, xslc)

        # get field data
        with h5py.File(casedir.joinpath("output", f"{fno:07d}.h5"), "r") as h5file:
            data = h5file[f"{field}"][cache[field]]

        # restore field name; stupid approach
        field = "vorticity_z" if field == "wz" else field
        preds[field] = data

    return preds


# %% get_pinn_data
def get_pinn_data(model, fields, time, x, y, bsize=8192):
    """Get vicinity data from PINN.
    """

    # torch version of gridlines; reshape to N by 1
    kwargs = {
        "dtype": next(model.parameters()).dtype,
        "device": "cpu", "requires_grad": True}

    # dataholder
    preds = {field: numpy.zeros(x.size) for field in fields}  # x is supposed to be a numpy array at this point

    # backup shape
    shape = x.shape

    # convert numpy coordinates to torch
    x = torch.tensor(x.reshape(-1, 1), **kwargs)  # pylint: disable=no-member
    y = torch.tensor(y.reshape(-1, 1), **kwargs)  # pylint: disable=no-member

    # block
    nblocks = x.numel() // bsize + int(bool(x.numel() % bsize))  # x is torch array, so use numel()

    for block in range(nblocks):
        # predict and save
        print(f"Predicting PINN data at t={time}, batch={block+1}/{nblocks}")
        loc = (slice(block * bsize, (block + 1) * bsize), slice(None))
        data = model({
            "x": x[loc], "y": y[loc],
            "t": torch.full_like(x[loc], time, requires_grad=True)
        })

        for field in fields:
            preds[field][loc[0]] = data[field].detach().cpu().numpy().flatten()

    # restore shape
    preds = {field: val.reshape(shape) for field, val in preds.items()}

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
    _pinndir.parent.joinpath("outputs").mkdir(exist_ok=True)
    _datafile = _pinndir.parent.joinpath("outputs", "refined_region.h5")

    # %% case configuration
    _fields = ["u", "v", "p", "vorticity_z", "qcriterion", "continuity", "momentum_x", "momentum_y"]
    _petibmfields = ["u", "v", "vorticity_z", "p"]
    _device = "cpu"
    _times = list(float(val) for val in numpy.linspace(135, 145, 21))
    _xlim = [-1.0, 3.0]
    _ylim = [-2.0, 2.0]
    _nx = 125
    _ny = 125
    _cache = {}

    # gridlines (vertices)
    _x = numpy.linspace(*_xlim, _nx+1)
    _y = numpy.linspace(*_ylim, _ny+1)

    # gridlines (cell centers)
    _x = (_x[1:] + _x[:-1]) / 2
    _y = (_y[1:] + _y[:-1]) / 2

    # mesh
    _x, _y = numpy.meshgrid(_x, _y)

    # save input data
    with h5py.File(_datafile, "w") as _dset:
        _dset.create_dataset("times", (len(_times),), numpy.float64, _times, compression="gzip")
        _dset.create_dataset("pinn/x", _x.shape, _x.dtype, _x, compression="gzip")
        _dset.create_dataset("pinn/y", _y.shape, _y.dtype, _y, compression="gzip")

    # %% petibm grid
    for _field in _petibmfields:

        _field = "wz" if _field == "vorticity_z" else _field

        with h5py.File(_petibmdir.joinpath("output", "grid.h5"), "r") as _dset:
            _petx = _dset[_field]["x"][...]
            _pety = _dset[_field]["y"][...]

        _field = "vorticity_z" if _field == "wz" else _field

        _xslc = numpy.arange(_petx.size)[(_petx >= _xlim[0]) & (_petx <= _xlim[1])]
        _yslc = numpy.arange(_pety.size)[(_pety >= _ylim[0]) & (_pety <= _ylim[1])]

        _cache[_field] = (slice(_yslc.min(), _yslc.max()+1), slice(_xslc.min(), _xslc.max()+1))

        _petx, _pety = numpy.meshgrid(_petx, _pety)
        _petx = _petx[_cache[_field]]
        _pety = _pety[_cache[_field]]

        with h5py.File(_datafile, "r+") as _dset:
            _dset.create_dataset(f"petibm/{_field}/x", _petx.shape, _petx.dtype, _petx, compression="gzip")
            _dset.create_dataset(f"petibm/{_field}/y", _pety.shape, _pety.dtype, _pety, compression="gzip")

    # %% get PINN model
    _model = get_pinn_model(_pinndir, 1000000, _fields)

    # %% predict and save
    for _time in _times:

        # get petibm data
        _data = get_petibm_data(_petibmdir, _petibmfields, _time, _xlim, _ylim, _cache)

        for _field, _val in _data.items():
            with h5py.File(_datafile, "r+") as _dset:
                _dset.create_dataset(
                    f"petibm/{_field}/{_time}", _val.shape, _val.dtype,
                    _val, compression="gzip"
                )

        # get pinn data
        _data = get_pinn_data(_model, _fields, _time, _x, _y)

        for _field, _val in _data.items():
            with h5py.File(_datafile, "r+") as _dset:
                _dset.create_dataset(
                    f"pinn/{_field}/{_time}", _val.shape, _val.dtype,
                    _val, compression="gzip"
                )
