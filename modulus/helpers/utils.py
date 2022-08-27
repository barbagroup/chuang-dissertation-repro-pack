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
import lzma
import re
import pandas
from sympy import sympify as _sympify
from tensorboard.backend.event_processing import event_accumulator
from torch import load
from torch import jit
from modulus.architecture.layers import Activation
from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.key import Key
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from .pdes import IncompNavierStokes
from .pdes import Vorticity
from .pdes import QCriterion
from .pdes import VelocityGradients


def read_tensorboard_data(workdir: pathlib.Path, excludes=None):
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
    data = data.drop_duplicates("step", keep="last").set_index("step").sort_index()

    # apply exclusions for steps that show up in history due to restarting
    if excludes is not None:
        excludes = set(excludes).intersection(set(data.index))
        data = data.drop(index=excludes)

    # convert elapsed time to run time
    try:
        data["run_time"] = data["elapsed"].cumsum() / 1000  # also convert from milliseconds to seconds
    except KeyError:  # not every case has the key `elapsed`
        pass

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


def create_graph(cfg, dim=2, unsteady=True, params=None):
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
        inkeys = ["x", "y"]
        outkeys = ["u", "v", "p"]
    elif dim == 3:
        inkeys = ["x", "y", "z"]
        outkeys = ["u", "v", "w", "p"]
    else:
        raise ValueError(f"Incorrect value for dim: {dim}")

    # temporal variable
    if unsteady:
        inkeys.append("t")

    # set up scales
    if "scaling" in cfg.custom and not cfg.custom.scaling:
        assert isinstance(cfg.custom.scaling, bool)
        scales = dict({key: (0., 1.) for key in inkeys})
    else:
        scales = dict({key: process_domain(cfg.custom[key]) for key in inkeys})
        scales = {key: (val[0], val[1]-val[0]) for key, val in scales.items()}

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
    nseq = IncompNavierStokes(cfg.custom.nu, cfg.custom.rho, dim, unsteady)

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


def get_graph_from_file(cfg, filename, dim=2, unsteady=True, device="cpu"):
    """Return a computational graph with model parameters from a snapshot file generated by SaveModelInferencer.
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
    graph, dtype = create_graph(cfg, dim, unsteady, model.state_dict())

    # timestamp and training iteration
    timestamp = snapshot["time"]
    step = snapshot["step"]

    return step, timestamp, graph, dtype


def get_graph_from_checkpoint(cfg, filename, dim=2, unsteady=True, mtype="raw", device="cpu"):
    """Return a computational with model parameters read from a checkpoint file.
    """

    if mtype == "raw":
        model = load(filename, map_location=device)
    elif mtype == "swa":
        data = load(filename, map_location=device)
        model = {}
        for key, val in data.items():
            result = re.search(r"module.0.([\w\.]+)", key)
            if result is None:
                continue
            model[result.group(1)] = val

    # get a computational graph
    graph, dtype = create_graph(cfg, dim, unsteady, model)

    return graph, dtype


def get_graph_modulus_cylinder40_example(outdir, device="cpu"):
    """Read model checkpoint from official example for Cylinder Re40.
    """

    if not isinstance(outdir, pathlib.Path):
        outdir = pathlib.Path(outdir).resolve()

    cfg = OmegaConf.load(outdir.joinpath(".hydra", "config.yaml"))
    cfg.custom = {}
    cfg.custom.x = [-10., 30.]
    cfg.custom.y = [-10., 10.]
    cfg.custom.nu = 0.025
    cfg.custom.rho = 1.0
    cfg.custom.activation = "silu"

    graph, dtype = get_graph_from_checkpoint(cfg, outdir.joinpath("flow_network.pth"), 2, False, device)

    return graph, dtype


def update_graph_with_file(cfg, filename, graph, device="cpu"):
    """Return a computational graph with model parameters from a snapshot file generated by SaveModelInferencer.
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

    # identify which node represents the actual neural network for the flow field
    for node in graph:
        if node.name == "flow-net":
            node.evaluate.load_state_dict(model.state_dict())
            break

    # determine dtype
    dtype = next(model.parameters()).dtype

    # timestamp and training iteration
    timestamp = snapshot["time"]
    step = snapshot["step"]

    return step, timestamp, graph, dtype


def log_parser(casedir):
    """Get iteration and elapsed wall time info from log files.
    """
    logs = list(casedir.glob("logs/run-*.log"))

    oldpat = r"\[step:\s+(\d+)\]\s+loss=\s*(\d\.\d+e[+-]\d+).*?time/iter=\s*(\d\.\d+e\+\d+)$"
    newpat = r"\[step:\s+(\d+)\]\s+loss=\s*(\d\.\d+e[+-]\d+).*?time elapsed=\s*(\d\.\d+e\+\d+)ms$"

    out = []
    for logfile in logs:
        # try old pattern first
        with open(logfile, "r") as fobj:
            results = re.findall(oldpat, fobj.read(), re.MULTILINE)

        if len(results) == 0:  # try new pattern if results is empty
            with open(logfile, "r") as fobj:
                results = re.findall(newpat, fobj.read(), re.MULTILINE)

            # update data type to int and floats
            for i, result in enumerate(results):
                results[i] = (int(result[0]), float(result[1]), float(result[2]))

        else:  # this file contains old pattern
            if results[0][0] == "0":
                results[0] = (int(0), float(results[0][1]), 0.0)
            else:
                results[0] = (int(results[0][0]),float(results[0][1]), float(results[0][2]))

            # old pattern records time per iteration; change to elapsed time
            for i, result in zip(range(1, len(results)), results[1:]):
                acctime = float(result[2]) * (int(result[0]) - int(results[i-1][0]))
                results[i] = (int(result[0]), float(result[1]), acctime)

        out.extend(results)

    out = pandas.DataFrame(out, columns=["step", "loss", "time elapsed"])
    out = out.drop_duplicates(subset="step", keep="first")
    out = out.sort_values(by="step", axis=0)
    out = out.set_index("step")
    out["time elapsed"] = out["time elapsed"].cumsum() / 1000 / 3600  # ms -> s -> hr
    # out = out["time elapsed"].to_dict()

    return out
