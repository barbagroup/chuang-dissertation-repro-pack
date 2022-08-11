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
from modulus.architecture.layers import Activation
from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.key import Key
from omegaconf.errors import ConfigAttributeError
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
            "Train/time_elapsed": "elapsed",  # originally in milliseconds
            "Monitors/pde_residual/continuity_res": "cont_res",
            "Monitors/pde_residual/momentum_x_res": "momem_x_res",
            "Monitors/pde_residual/momentum_y_res": "momem_y_res",
            "Monitors/pde-residual/continuity_res": "cont_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/momentum_x_res": "momem_x_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/momentum_y_res": "momem_y_res",  # underscore was replaced by hyphen at some point
            "Monitors/pde-residual/continuity": "cont_res",  # _res was removed in some cases
            "Monitors/pde-residual/momentum_x": "momem_x_res",  # _res was removed in some cases
            "Monitors/pde-residual/momentum_y": "momem_y_res",  # _res was removed in some cases
            'Monitors/noslip-residual/cylinder_u': "cylinder_u",
            'Monitors/noslip-residual/cylinder_v': "cylinder_v",
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
    data = pandas.concat(data).reset_index(drop=False)
    data = data.drop_duplicates("step", keep="first").set_index("step").sort_index()

    # convert elapsed time to run time
    data["run_time"] = data["elapsed"].cumsum() / 1000  # also convert from milliseconds to seconds

    return data


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


def get_activation_fn(name: str):
    """Convert string name to modulus.architecture.Activation.
    """
    mapping = {
        "elu": Activation.ELU,
        "leaky_relu": Activation.LEAKY_RELU,
        "mish": Activation.MISH,
        "poly": Activation.POLY,
        "relu": Activation.RELU,
        "gelu": Activation.GELU,
        "selu": Activation.SELU,
        "prelu": Activation.PRELU,
        "sigmoid": Activation.SIGMOID,
        "silu": Activation.SILU,
        "sin": Activation.SIN,
        "tanh": Activation.TANH,
        "identity": Activation.IDENTITY,
    }
    return mapping[name]


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
    try:
        periodicity = {key: process_domain(cfg.custom[key]) for key in cfg.custom.periodic}
    except ConfigAttributeError:
        periodicity = None

    # convert lists of modulus.keys.Key instances
    inkeys = [Key(key, scale=scales[key]) for key in inkeys]
    outkeys = [Key(key) for key in outkeys]

    # network
    net = FullyConnectedArch(
        input_keys=inkeys, output_keys=outkeys, periodicity=periodicity,
        activation_fn=get_activation_fn(cfg.custom.activation),
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


def get_model_from_file(cfg, filename, dim=2, device="cpu"):
    """Return a computational with model parameters read from a snapshot file generated by SaveModelInferencer.
    """

    # load the snatshot data
    with lzma.open(filename, "rb") as obj:  # open the file
        snapshot = load(obj, map_location=device)

    # load the model parameters form the snapshot data
    with io.BytesIO(snapshot["model"]) as obj:  # load the model
        if cfg.jit:
            model = jit.load(obj, map_location=device)
        else:
            model = load(obj, map_location=device)

    # `model` currently may be a sub-model collection (e.g., flow model, density model); grab the flow model
    if hasattr(model, "name"):  # both jit and non-jit FullyConnectedArch have `name` attribute
        if model.name != "flow-net":
            raise RuntimeError("Couldn't find a networkd called flow-net.")
    else:  # otherwise, assume it's a ModuleList
        if cfg.jit:  # only jit RecursiveScriptModule has `original_name`
            mdname = model.original_name
        else:
            mdname = model.__class__.__name__

        if mdname == "ModuleList":
            for name, model in model.named_children():
                if model.name == "flow-net":
                    break
            else:
                raise RuntimeError("Couldn't find a networkd called flow-net.")
        else:
            raise RuntimeError(f"Unrecognized module type: {mdname}")

    # double check the flow-net it's a FullyConnectedArch
    if cfg.jit:
        mdname = model.original_name
    else:
        mdname = model.__class__.__name__

    if mdname != "FullyConnectedArch":
        raise NotImplementedError

    # get a computational graph
    graph, dtype = create_graph(cfg, dim, model.state_dict())

    # timestamp and training iteration
    timestamp = snapshot["time"]
    step = snapshot["step"]

    return step, timestamp, graph, dtype
