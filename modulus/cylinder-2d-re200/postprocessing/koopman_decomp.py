#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Koopman analysis (dynamic mode decomposition).
"""

# %% headers
import sys
import pathlib
import numpy
import scipy.signal
import scipy.fftpack
import scipy.linalg
import scipy.sparse.linalg
import h5py
import torch
from omegaconf import OmegaConf
from modulus.key import Key
from modulus.graph import Graph

# find helpers and locate workdir
for _projdir in pathlib.Path(__file__).resolve().parents:
    if _projdir.joinpath("modulus").is_dir():
        sys.path.insert(0, str(_projdir.joinpath("modulus")))
        from helpers.utils import get_graph_from_checkpoint
        from helpers.utils import log_parser
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


# %%
def get_pinn_graph(datadir):
    """Get immediately usable PINN computational graph.
    """

    # use log to determine should we use the last checkpoint or the checkpoint with best loss
    logs = log_parser(datadir.parent)

    # find the step with the minumum loss
    steps = [int(f.name.replace("flow-net.pth.", "")) for f in datadir.glob("flow-net.pth.*")]
    bestloss = logs.loc[steps, "loss"].min()
    beststep = logs.loc[steps, "loss"].idxmin()

    # case configuration
    cfg = OmegaConf.load(datadir.parent.joinpath("config.yaml"))
    cfg.device = "cpu"
    cfg.custom.scaling = False

    modelfile = datadir.joinpath(f"flow-net.pth.{beststep}")

    graph, _ = get_graph_from_checkpoint(cfg, modelfile, dim=2, unsteady=True, mtype="raw", device="cpu")
    print(f"Done reading model from {modelfile.name} w/ best loss of {bestloss}")

    return graph


# %%
def get_pinn_predictions(graph, coords, times, chosen):
    """Use the computational graph to get predictions.
    """
    K = {
        field: numpy.zeros((numpy.count_nonzero(idxs), len(times)-1))
        for field, idxs in chosen.items()
    }  # snapshot matrix

    xm = {
        field: numpy.zeros(numpy.count_nonzero(idxs))
        for field, idxs in chosen.items()
    }  # the last snapshot

    for field in chosen.keys():
        model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list([field]))
        kwargs = {"dtype": next(model.parameters()).dtype, "device": "cpu", "requires_grad": False}
        x = torch.tensor(coords[field][0][chosen[field]].reshape(-1, 1), **kwargs)
        y = torch.tensor(coords[field][1][chosen[field]].reshape(-1, 1), **kwargs)

        bsize = 8192 * 64
        nblocks = x.numel() // bsize + int(bool(x.numel() % bsize))

        for i, time in enumerate(times[:-1]):
            for block in range(nblocks):
                print(f"Predicting field = {field},  time = {time}, batch {block}/{nblocks}")
                seg = slice(block*bsize, (block+1)*bsize)
                preds = model({
                    "x": x[seg, :], "y": y[seg, :], "t": torch.full_like(x[seg, :], time)
                })
                K[field][seg, i] = preds[field].detach().cpu().numpy().flatten()

        # the last time snapshot
        for block in range(nblocks):
            print(f"Predicting field = {field}, time = {times[-1]}, batch {block}/{nblocks}")
            xm[field][seg] = model({
                "x": x[seg, :], "y": y[seg, :], "t": torch.full_like(x[seg, :], times[-1])
            })[field].detach().cpu().numpy().flatten()

    # correct for reference pressure
    K["p"] -= K["p"].mean(axis=0)  # shape: (N1, N2) = (N1, N2) - (N2,) = (N1, N2) - (1, N2)
    xm["p"] -= xm["p"].mean()

    return K, xm


# %%
def get_pinn_snapshots(h5grp, datadir):
    """Read snapshots.
    """

    # times must exist already at the root group/file level
    times = h5grp.file.attrs["times"]
    fields = ["u", "v", "p"]  # for convenience

    if "data_done" in h5grp.attrs:
        if h5grp.attrs["data_done"] == len(times):
            return  # we have everything; no need to proceed further
    else:
        h5grp.attrs["data_done"] = 0  # initialize the counter

    # all fields share the same coordinates, so we use use u's for everything
    mask = h5grp["u/mask"][...]
    shape = (numpy.count_nonzero(~mask), len(times)-1)

    # initialize data holder directly in the HDF5
    for field in fields:
        h5grp[field].require_dataset("K", shape, h5grp["u/x"].dtype, True)
        h5grp[field].require_dataset("xm", shape[0], h5grp["u/x"].dtype, True)

    # get the model
    graph = get_pinn_graph(datadir)
    model = Graph(graph, Key.convert_list(["x", "y", "t"]), Key.convert_list(fields))

    # setup the inputs
    kwargs = {"dtype": next(model.parameters()).dtype, "device": "cpu", "requires_grad": False}
    x = torch.tensor(h5grp["u/x"][~mask].reshape(-1, 1), **kwargs)
    y = torch.tensor(h5grp["u/y"][~mask].reshape(-1, 1), **kwargs)

    # setup batches
    bsize = 8192 * 64
    nblocks = shape[0] // bsize + int(bool(shape[0] % bsize))

    # predict
    for i, time in enumerate(times[:-1]):
        # this time step already exists in K
        if h5grp.attrs["data_done"] > i:
            continue

        for block in range(nblocks):
            print(f"Predicting PINN t = {time}, batch {block+1}/{nblocks}")
            seg = slice(block*bsize, (block+1)*bsize)
            preds = model({
                "x": x[seg, :], "y": y[seg, :], "t": torch.full_like(x[seg, :], time)
            })
            for field in fields:
                h5grp[f"{field}/K"][seg, i] = preds[field].detach().cpu().numpy().flatten()

        # update the flag
        h5grp.attrs["data_done"] = i + 1

    # no need to do the last time step
    if h5grp.attrs["data_done"] == len(times):
        return

    # the last time step
    for block in range(nblocks):
        print(f"Predicting PINN t = {times[-1]}, batch {block+1}/{nblocks}")
        seg = slice(block*bsize, (block+1)*bsize)
        preds = model({
            "x": x[seg, :], "y": y[seg, :], "t": torch.full_like(x[seg, :], times[-1])
        })
        for field in fields:
            h5grp[f"{field}/xm"][seg] = preds[field].detach().cpu().numpy().flatten()

    # update the flag
    h5grp.attrs["data_done"] = len(times)


# %%
def get_pinn_coords(h5grp, lims, res, radius):
    """Get coordinates for PINN.
    """
    # in PINN, all fields shared the same coordinates
    fields = ["u", "v", "p"]
    data = {"x": None, "y": None, "mask": None}

    # pre-create groups if not existing
    for field in fields:
        h5grp.require_group(field)

    # determine if we need to proceed further
    if all(set(data.keys()).issubset(set(h5grp[field].keys())) for field in fields):
        return

    data["x"], data["y"] = numpy.meshgrid(
        numpy.linspace(*lims[:2], res[1]),
        numpy.linspace(*lims[2:], res[0]),
    )
    data["mask"] = ((data["x"]**2 + data["y"]**2) <= radius**2)

    # only save the actual data to u field
    for key, val in data.items():
        h5grp.require_dataset(f"u/{key}", val.shape, val.dtype, True)[...] = val

    # for other fields, we make hard links
    for field in ["v", "p"]:
        for key in data.keys():
            h5grp[f"{field}/{key}"] = h5grp[f"u/{key}"]


# %%
def get_petibm_coords(h5grp, datadir, lims, radius):
    """Get petibm coordinates.
    """

    # coordinates of the nodes
    for field in ["u", "v", "p"]:
        # for convenience, we'll use dict
        data = {"x": None, "y": None, "mask": None}
        idxrange = numpy.zeros(4, dtype=int)

        # check if everything is already on record
        h5grp.require_group(field)  # make sure it exists
        if set(data.keys()).issubset(set(h5grp[field].keys())):
            continue  # we have everything, jump to the next field

        # otherwise, read in coords and re-calculate masks and block range
        with h5py.File(datadir.joinpath("grid.h5"), "r") as dset:
            data["x"], data["y"] = dset[field]["x"][...], dset[field]["y"][...]

        # block slices corresponding to the `lims`
        indices = numpy.where((data["x"] >= lims[0]) & (data["x"] <= lims[1]))[0]
        idxrange[:2] = [indices.min(), indices.max()+1]

        indices = numpy.where((data["y"] >= lims[2]) & (data["y"] <= lims[3]))[0]
        idxrange[2:] = [indices.min(), indices.max()+1]

        # only generate 2D grid within `lims`
        data["x"], data["y"] = numpy.meshgrid(
            data["x"][idxrange[0]:idxrange[1]],
            data["y"][idxrange[2]:idxrange[3]]
        )

        # a mask to mask out the cylinder
        data["mask"] = ((data["x"]**2 + data["y"]**2) <= radius**2)

        # write data
        for key, val in data.items():
            h5grp[field].require_dataset(key, val.shape, val.dtype, True)[...] = val

        h5grp[field].attrs["idxrange"] = idxrange


# %%
def get_petibm_snapshots(h5grp, datadir, fids):
    """Read snapshots.
    """

    # times must exist already at the root group/file level
    times = h5grp.file.attrs["times"]

    for field in ["u", "v", "p"]:
        # determine if we already have requested data in the cachefile
        if {"K", "xm"}.issubset(set(h5grp[field].keys())):
            continue  # we have everything, jump to the next field

        mask = h5grp[f"{field}/mask"][...]
        slcs = h5grp[field].attrs["idxrange"]

        K = numpy.zeros((numpy.count_nonzero(~mask), len(fids)-1))  # snapshot matrix
        xm = numpy.zeros(numpy.count_nonzero(~mask))

        # store data to K
        for i, fid in enumerate(fids[:-1]):
            with h5py.File(datadir.joinpath(f"{fid:07d}.h5"), "r") as dset:
                assert abs(times[i]-dset["p"].attrs["time"]) < 1e-8, \
                    f"{times[i]} v.s. {dset['p'].attrs['time']}"
                K[:, i] = dset[field][slcs[2]:slcs[3], slcs[0]:slcs[1]][~mask].flatten()
            print(f"Done reading PetIBM {field} at t={times[i]} from {fid:07d}.h5")

        # the last snapshot
        with h5py.File(datadir.joinpath(f"{fids[-1]:07d}.h5"), "r") as dset:
            assert abs(times[-1]-dset["p"].attrs["time"]) < 1e-8, \
                f"{times[-1]} v.s. {dset['p'].attrs['time']}"
            xm[...] = dset[field][slcs[2]:slcs[3], slcs[0]:slcs[1]][~mask].flatten()
        print(f"Done reading PetIBM {field} at t={times[-1]} from {fid:07d}.h5")

        # correct for reference pressure
        if field == "p":
            K -= K.mean(axis=0)
            xm -= xm.mean()

        # save
        for key, val in zip(("K", "xm"), (K, xm)):
            dset = h5grp[field].require_dataset(key, val.shape, val.dtype, True)
            dset[...] = val


# %%
def get_companion_matrix(cachefile, K, xm):
    """Get the companion matrix.
    """

    # try to read economic-QR data from cache
    try:
        with h5py.File(cachefile, "r") as dset:
            Q = dset["Q"][...]
            R = dset["R"][...]
        print(f"Done reading cached economic-QR data from {cachefile.name}")
    except KeyError:
        Q, R = scipy.linalg.qr(K, mode="economic")
        print("Done re-calculating economic-QR.")

        # save
        with h5py.File(cachefile, "r+") as dset:
            dset.create_dataset("Q", shape=Q.shape, dtype=Q.dtype, data=Q)
            dset.create_dataset("R", shape=R.shape, dtype=R.dtype, data=R)

    # try to read economic-QR data from cache
    try:
        with h5py.File(cachefile, "r") as dset:
            c = dset["c"][...]
        print(f"Done reading cached c vector from {cachefile.name}")
    except KeyError:
        c = scipy.linalg.solve(R, Q.T.dot(xm))
        print("Done re-calculating c vector")

        # save
        with h5py.File(cachefile, "r+") as dset:
            dset.create_dataset("c", shape=c.shape, dtype=c.dtype, data=c)

    # residual check
    assert numpy.allclose(K.T.dot(xm-K.dot(c)), 0, 1e-8, 1e-8), numpy.linalg.norm(K.T.dot(xm-K.dot(c)))

    # construct C matrix
    C = numpy.zeros((K.shape[1], K.shape[1]))
    C[range(1, K.shape[1]), range(0, K.shape[1]-1)] = 1.0
    C[:, -1] = c
    print("Done constructing the companion matrix C")

    return C


# %%
def get_modes_through_companion_matrix(cachefile, K, xm):
    """Get Koopman eigenvalues and modes through diagonalizaion of a companion matrix.
    """

    # get the companion matrix
    C = get_companion_matrix(cachefile, K, xm)

    print("Calculating Koopman eigenvalues and modes.")
    lambdas, T = scipy.linalg.eig(C)
    V = K.dot(T)

    # save
    with h5py.File(cachefile, "r+") as dset:
        dset.create_dataset("lambda", shape=lambdas.shape, dtype=lambdas.dtype, data=lambdas)
        dset.create_dataset("V", shape=V.shape, dtype=V.dtype, data=V)

    return lambdas, V


# %%
def get_modes_through_svd(h5grp):
    """Get Koopman eigenvalues and modes through the method of snapshots and SVD.
    """

    # if modes and eigenvalues already exist, nothing to do
    if {"eigvals", "energys"}.issubset(set(h5grp.keys())):
        if all("modes" in h5grp[field] for field in ["u", "v", "p"]):
            return

    # intermediate results will be saved to here
    chkgrp = h5grp.require_group("checkpt")

    # get K and xm
    K = numpy.concatenate(tuple(h5grp[f"{field}/K"][...] for field in ["u", "v", "p"]), axis=0)
    xm = numpy.concatenate(tuple(h5grp[f"{field}/xm"][...] for field in ["u", "v", "p"]), axis=0)

    # calculate normalizer
    h5grp.attrs["normalizer"] = numpy.sqrt(((K**2).sum()+(xm**2).sum())/(K.shape[1]+1))

    # K^* \ cdot K = W \cdot \diag(sigma**2) \cdot W^*
    # symmetric real matrix; eigvals all real
    singvals, eigvecs = scipy.linalg.eigh(K.T.dot(K))  # `sigvals` holds eigen vals

    # save the square of singular values because they are real numbers
    for key, val in zip(("SIGMA2", "W"), (singvals, eigvecs)):
        chkgrp.require_dataset(key, val.shape, val.dtype, True)[...] = val

    # singvals now denotes K's singular values (singular vals = (eigen vals)**0.5)
    singvals = numpy.sqrt(singvals.astype(numpy.complex128))

    print("Calculating Koopman eigenvalues and modes")

    # reuse `eigvecs` to hold W \dot diag(1/sigmas)
    eigvecs = eigvecs.dot(numpy.diag(numpy.reciprocal(singvals)))

    # U = K \cdot W \cdot diag(1/sigmas)
    U = K.dot(eigvecs)

    # `eigvecs` now denotes K_* \cdot W \cdot \diag(1/sigma)
    eigvecs = numpy.concatenate((K[:, 1:], xm[:, None]), axis=1).dot(eigvecs)
    x0 = K[:, 0].copy()  # needed later
    del K, xm  # to save memory

    # `eigvecs` denotes K \cdot W \cdot diag(1/sigmas) \cdot K_* \cdot W \cdot \diag(1/sigma)
    eigvecs = U.conj().T.dot(eigvecs)

    # diagonalization: Y \cdot \Lambda Y^{-1}; eigvecs now denotes Y
    eigvals, eigvecs = scipy.linalg.eig(eigvecs)

    # save
    chkgrp.require_dataset("Y", eigvecs.shape, eigvecs.dtype, True)[...] = eigvecs
    h5grp.require_dataset("eigvals", eigvals.shape, eigvals.dtype, True)[...] = eigvals

    # V = K \cdot W \codt diag(1/sigma) \ cdot Y = U \cdot Y
    eigvecs = U.dot(eigvecs)

    for key, val in zip(("unscaled_modes", "U"), (eigvecs, U)):
        chkgrp.require_dataset(key, val.shape, val.dtype, True)[...] = val

    # scale modes so that K = V \cdot T; T -- the Vandermonda matrix; reuse symbols
    phi0, U = numpy.linalg.eigh(eigvecs.conj().T.dot(eigvecs))  # reuse symbols
    phi0 = U.dot(numpy.diag(numpy.reciprocal(phi0)).dot(U.conj().T).dot(eigvecs.conj().T.dot(x0)))
    eigvecs = eigvecs.dot(numpy.diag(phi0))

    # save
    chkgrp.require_dataset("phi0", phi0.shape, phi0.dtype, True)[...] = phi0

    # energy norm (2-norm)
    h5grp.require_dataset("energys", eigvecs.shape[1], float, True)[...] = \
        numpy.linalg.norm(numpy.absolute(eigvecs), axis=0) / h5grp.attrs["normalizer"]

    # split modes into u, v, and p; reuse symbols
    phi0 = [h5grp[f"{field}/K"].shape[0] for field in ["u", "v"]]
    phi0[1] += phi0[0]
    eigvecs = numpy.split(eigvecs, phi0, axis=0)

    # save
    for i, (field, val) in enumerate(zip(["u", "v", "p"], eigvecs)):
        h5grp.require_dataset(f"{field}/modes", val.shape, val.dtype, True)[...] = val


# %%
def get_petibm_modes(datadir, outfile, lims, radius, fids):
    """Get Koopman modes and eigenvalues.
    """

    with h5py.File(outfile, "r+") as h5file:
        # make sure this group exists
        h5grp = h5file.require_group("petibm")

        # coordinates of the nodes (write to outfile)
        get_petibm_coords(h5grp, datadir, lims, radius)

        # get data matrix (write to outfile)
        get_petibm_snapshots(h5grp, datadir, fids)

        # get modes and eigenvalues
        get_modes_through_svd(h5grp)

        # post-processing eigenvalues
        h5grp.require_dataset("growths", h5grp["eigvals"].shape, float, True)[...] = \
            numpy.log(h5grp["eigvals"]).real
        h5grp.require_dataset("angles", h5grp["eigvals"].shape, float, True)[...] = \
            numpy.log(h5grp["eigvals"]).imag
        h5grp.require_dataset("freqs", h5grp["eigvals"].shape, float, True)[...] = \
            h5grp["angles"] / (2. * numpy.pi * h5file.attrs["dt"] * h5file.attrs["nsteps"])


# %%
def get_pinn_modes(datadir, outfile, lims, res, radius):
    """Get Koopman modes and eigenvalues.
    """

    with h5py.File(outfile, "r+") as h5file:
        # make sure this group exists
        h5grp = h5file.require_group("pinn")

        # coordinates of the nodes (write to outfile)
        get_pinn_coords(h5grp, lims, res, radius)

        # get data matrix (write to outfile)
        get_pinn_snapshots(h5grp, datadir)

        # get modes and eigenvalues
        get_modes_through_svd(h5grp)

        # post-processing eigenvalues
        h5grp.require_dataset("growths", h5grp["eigvals"].shape, float, True)[...] = \
            numpy.log(h5grp["eigvals"]).real
        h5grp.require_dataset("angles", h5grp["eigvals"].shape, float, True)[...] = \
            numpy.log(h5grp["eigvals"]).imag
        h5grp.require_dataset("freqs", h5grp["eigvals"].shape, float, True)[...] = \
            h5grp["angles"] / (2. * numpy.pi * h5file.attrs["dt"] * h5file.attrs["nsteps"])


# %% main function
if __name__ == "__main__":

    _outfile = pathlib.Path(__file__).resolve().parents[1].joinpath("outputs", "koopman.h5")
    _outfile.parent.mkdir(parents=True, exist_ok=True)

    # some parameters
    _dt = 0.005
    _nsteps = 40
    _fids = numpy.arange(25000, 28001, _nsteps, dtype=int)
    _times = _fids * _dt
    _radius = 0.5
    _lims = (-3, 7, -3., 3.)

    with h5py.File(_outfile, "a") as _h5file:
        _h5file.attrs["dt"] = _dt
        _h5file.attrs["times"] = _times
        _h5file.attrs["fids"] = _fids
        _h5file.attrs["nsteps"] = _nsteps
        _h5file.attrs["radius"] = _radius
        _h5file.attrs["lims"] = _lims

    # %% get petibm modes
    _datadir = _projdir.joinpath("petibm", "cylinder-2d-re200", "output")
    get_petibm_modes(_datadir, _outfile, _lims, _radius, _fids)

    # %% set the resolution of pinn (use PetIBM's p grid)
    with h5py.File(_outfile, "r+") as _h5file:
        _res = _h5file["petibm/p/mask"].shape

    _datadir = _projdir.joinpath(
        "modulus", "cylinder-2d-re200", "nl6-nn512-npts6400-unsteady-petibm",
        "outputs")

    get_pinn_modes(_datadir, _outfile, _lims, _res, _radius)
