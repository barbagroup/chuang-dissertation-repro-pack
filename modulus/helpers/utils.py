#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Utilities for Modulus.
"""
import io
import pathlib
import pandas
import lzma
from sympy import sympify as _sympify
from tensorboard.backend.event_processing import event_accumulator
from torch import load
from torch import jit
from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.key import Key
from .pdes import IncompNavierStokes
from .pdes import Vorticity
from .pdes import QCriterion
from .pdes import VelocityGradients


def read_tensorboard_data(workdir: pathlib.Path):
    """Read tensorboard events.
    """

    # get a list of all event files
    filenames = workdir.glob("**/events.out.tfevents.*")

    # data holder
    data = []

    # read events, one dataframe per event file
    for filename in filenames:
        reader = event_accumulator.EventAccumulator(
            path=str(filename),
            size_guidance={event_accumulator.TENSORS: 0}
        )
        reader.Reload()

        keymap = {
            "Train/loss_aggregated": "loss",
            "Monitors/pde_residual/continuity_res": "cont_res",
            "Monitors/pde_residual/momentum_x_res": "momem_x_res",
            "Monitors/pde_residual/momentum_y_res": "momem_y_res",
            "Monitors/pde-residual/continuity_res": "cont_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/momentum_x_res": "momem_x_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/momentum_y_res": "momem_y_res",  # underscore was replaced by hyphen at some point
        }

        frame = pandas.DataFrame()

        for key, name in keymap.items():
            try:
                temp = pandas.DataFrame(reader.Tensors(key)).set_index("step")
            except KeyError as err:
                if "not found in Reservoir" in str(err):
                    continue
                raise

            temp[name] = temp["tensor_proto"].apply(lambda inp: inp.float_val[0])

            if "wall_time" in frame.columns:
                temp = temp.drop(columns=["tensor_proto", "wall_time"])
            else:
                temp = temp.drop(columns=["tensor_proto"])

            frame = frame.join(temp, how="outer")

        # add to collection
        data.append(frame)

    # concatenate (and sort) all partial individual dataframes
    data = pandas.concat(data).sort_index()

    return data.reset_index(drop=False)


def process_domain(inp):
    """Use SymPy to parse and process domain bounds.

    Arguments
    ---------
    inp : a list of strs of floats

    Returns
    -------
    inp : a tuple of floats
    """
    return tuple(float(_sympify(key).evalf()) for key in inp)


def create_graph(cfg, dim=2, params=None):
    """Create a computational graph that has every thing needed for postporcessing (with random model parameters).

    Notes
    -----
    The Hydra config file must have a node called `custom` and have the following structure:
        ```
        custom:
          x: [<lower bound>, <upper bound>]
          y: [<lower bound>, <upper bound>]
          t: [<lower bound>, <upper bound>]
          nu: 0.01
          rho: 1.0
        ```
    Two extra sub-nodes are allowed:

    1. If dealing a 3D problem: `z: [<lower bound>, <upper bound>]`
    2. If having periodic domain, for example: `periodic: ["x", "y"]`
    """

    # names of inputs into and outputs from the network model
    if dim == 2:
        inkeys = ["x", "y", "t"]
        outkeys = ["u", "v", "p"]
    elif dim == 3:
        inkeys = ["x", "y", "z", "t"]
        outkeys = ["u", "v", "w", "p"]
    else:
        raise ValueError(f"Incorrect value for dim: {dim}")

    # set up scales
    scales = dict({key: process_domain(cfg.custom[key]) for key in inkeys})

    # set up periodicity
    periodicity = {key: process_domain(cfg.custom[key]) for key in cfg.custom.periodic}

    # convert lists of modulus.keys.Key instances
    inkeys = [Key(key, scale=scales[key]) for key in inkeys]
    outkeys = [Key(key) for key in outkeys]

    # network
    net = FullyConnectedArch(
        input_keys=inkeys, output_keys=outkeys, periodicity=periodicity,
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    # update parameters
    if params is not None:
        net.load_state_dict(params)

    # navier-stokes equation
    nseq = IncompNavierStokes(cfg.custom.nu, cfg.custom.rho, dim, True)

    # vorticity
    vorticity = Vorticity(dim=dim)

    # q-criterion
    qcriterion = QCriterion(dim=dim)

    # velocity gradients
    velgradient = VelocityGradients(dim=dim)

    nodes = \
        nseq.make_nodes() + vorticity.make_nodes() + qcriterion.make_nodes() + velgradient.make_nodes() + \
        [net.make_node(name="flow-net", jit=cfg.jit)]

    dtype = next(net.parameters()).dtype

    return nodes, dtype


def get_rawmodel_from_file(cfg, filename, dim=2):
    """Return a computational with model parameters read from a snapshot file generated by SaveModelInferencer.
    """

    # load the snatshot data
    with lzma.open(filename, "rb") as obj:  # open the file
        snapshot = load(obj, map_location=cfg.device)

    # load the model parameters form the snapshot data
    with io.BytesIO(snapshot["model"]) as obj:  # load the model
        params = jit.load(obj, map_location=cfg.device).state_dict()

    # get a computational graph
    graph, dtype = create_graph(cfg, dim, params)

    # timestamp and training iteration
    timestamp = snapshot["time"]
    step = snapshot["step"]

    return step, timestamp, graph, dtype


def get_swamodel_from_file(cfg, rawmodel, filename):
    """Return the swa-model of a given raw model.
    """
    with lzma.open(filename, "rb") as obj:
        snapshot = load(obj, map_location=cfg.device)

    # a TorchScript-compiled ModuleList from Domain.create_global_optimizer_model
    with io.BytesIO(snapshot["model"]) as obj:
        modellist = jit.load(obj, map_location=cfg.device)

    # this assumes the first sub-model is the N-S net
    for name, model in modellist.named_children():
        if name == "0":
            break

    return model
