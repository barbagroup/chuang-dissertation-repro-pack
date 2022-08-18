#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D Cylinder flow Re=200.
"""
# pylint: disable=invalid-name, relative-beyond-top-level
import sys
import pathlib
import numpy
import sympy
from h5py import File as _h5File
from modulus.key import Key as _Key
from modulus.hydra.config import ModulusConfig as _ModulusConfig
from modulus.geometry.csg.csg_2d import Rectangle as _Rectangle
from modulus.geometry.csg.csg_2d import Circle as _Circle
from modulus.architecture.fully_connected import FullyConnectedArch as _FullyConnectedArch
from modulus.continuous.domain.domain import Domain as _Domain
from omegaconf.errors import ConfigAttributeError as _ConfigAttributeError

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.aggregators import register_loss_configs  # pylint: disable=import-error
        from helpers.schedulers import register_scheduler_configs  # pylint: disable=import-error
        from helpers.solvers import AdamNCGSWA  # pylint: disable=import-error
        from helpers.solvers import register_optimizer_configs  # pylint: disable=import-error
        from helpers.utils import process_domain  # pylint: disable=import-error
        from helpers.utils import get_activation_fn  # pylint: disable=import-error
        from helpers.constraints import StepAwarePointwiseConstraint  # pylint: disable=import-error
        register_loss_configs()
        register_scheduler_configs()
        register_optimizer_configs()
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_computational_graph(cfg: _ModulusConfig):
    """Returns the computational graph as a list of nodes.
    """

    # set up scales
    scales = dict({key: process_domain(cfg.custom[key]) for key in ["x", "y", "t"]})

    # set up periodicity
    try:
        periodicity = {key: process_domain(cfg.custom[key]) for key in cfg.custom.periodic}
    except _ConfigAttributeError:
        periodicity = None

    net = _FullyConnectedArch(
        input_keys=[_Key(key, scale=scales[key]) for key in ["x", "y", "t"]],
        output_keys=[_Key("u"), _Key("v"), _Key("p")],
        periodicity=periodicity,
        activation_fn=get_activation_fn(cfg.custom.activation),
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    nodes = [net.make_node(name="flow-net", jit=cfg.jit)]

    return nodes, net


def get_computational_domain(cfg: _ModulusConfig):
    """Get a geometry object representing the computational domain.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    radius = sympy.sympify(cfg.custom.radius).evalf()
    geo = _Rectangle((xbg, ybg), (xed, yed)) - _Circle((0, 0), radius)

    return geo


def get_petibm_constraints(nodes, geo, cfg):
    """Get an interior pointwise constraint for initial conditions (i.e., t=0)
    """

    datadir = pathlib.Path(cfg.orgdir).resolve().joinpath("data")

    constraints = {}
    for time in range(125, 141):  # PetIBM data from t=125 to t=140
        for field in ["u", "v", "p"]:
            invars = {}
            outvars = {}

            with _h5File(datadir.joinpath("grid.h5"), "r") as dset:
                invars["x"], invars["y"] = numpy.meshgrid(dset[field]["x"], dset[field]["y"], indexing="xy")
                invars["x"] = invars["x"].reshape(-1, 1)
                invars["y"] = invars["y"].reshape(-1, 1)
                invars["t"] = numpy.full_like(invars["x"], float(time))

            with _h5File(datadir.joinpath(f"{time*2*100:07d}.h5"), "r") as dset:
                outvars[field] = dset[field][...].reshape(-1, 1)

            constraints[f"{field}{time}"] = StepAwarePointwiseConstraint.from_numpy(
                nodes=nodes,
                invar=invars,
                outvar=outvars,
                batch_size=cfg.batch_size.nic,
            )

    return constraints


def get_solver_domains(nodes, geo, cfg):
    """Get a modules' Domain for the first time window.

    No BC constraints needed because they are periodic BCs and are handled by the neural network.
    """

    domain = _Domain()

    # petibm data
    for name, val in get_petibm_constraints(nodes, geo, cfg).items():
        domain.add_constraint(val, name=name)

    return domain


def main(cfg: _ModulusConfig):
    """The main function.
    """
    nodes, _ = get_computational_graph(cfg)
    geo = get_computational_domain(cfg)
    domain = get_solver_domains(nodes, geo, cfg)
    solver = AdamNCGSWA(cfg=cfg, domain=domain)
    solver.solve()
    return 0


if __name__ == "__main__":
    import modulus
    root = pathlib.Path(__file__).resolve().parent

    # re-make the cmd arguments with the program name and desired workdir
    sys.argv = sys.argv[:1] + [r"hydra.run.dir=outputs", rf"+orgdir={str(root)}"]

    # modulus.main is a function wrapper to help loading configs
    sys.exit(modulus.main(str(root), "config")(main)())
