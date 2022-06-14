#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D Taylor-Green vortex.
"""
# pylint: disable=invalid-name, relative-beyond-top-level
import sys
import pathlib
import sympy
from torch import sum as _torchsum  # pylint: disable=no-name-in-module
from torch import abs as _torchabs  # pylint: disable=no-name-in-module
from modulus.key import Key as _Key
from modulus.hydra.config import ModulusConfig as _ModulusConfig
from modulus.geometry.csg.csg_2d import Rectangle as _Rectangle
from modulus.architecture.fully_connected import FullyConnectedArch as _FullyConnectedArch
from modulus.continuous.domain.domain import Domain as _Domain
from modulus.continuous.monitor.monitor import PointwiseMonitor as _PointwiseMonitor

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.pdes import IncompNavierStokes  # pylint: disable=import-error
        from helpers.inferencers import SaveModelInferencer  # pylint: disable=import-error
        from helpers.aggregators import register_loss_configs  # pylint: disable=import-error
        from helpers.schedulers import register_scheduler_configs  # pylint: disable=import-error
        from helpers.solvers import LBFGSSolver  # pylint: disable=import-error
        from helpers.solvers import register_optimizer_configs  # pylint: disable=import-error
        from helpers.utils import process_domain  # pylint: disable=import-error
        from helpers.constraints import StepAwarePointwiseInteriorConstraint as _StepAwarePointwiseInteriorConstraint
        register_loss_configs()
        register_scheduler_configs()
        register_optimizer_configs()
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_symbolic_ic(L, V0, x="x", y="y", rho="rho"):
    """Sympy symbolic initial confition for 3D Taylor-Green vortex.

        u = sin(x) * cos(y) * cos(z)
        v = - cos(x) * sin(y) * cos(z)
        z = 0
        p = (cos(2*x) + cos(2*y)) * (cos(2*z) + 2) / 16

    """

    _L = sympy.Number(L)  # don't want to make L a parameter
    _V0 = sympy.Number(V0)  # don't want to make V0 a parameter

    _x, _y, _rho = sympy.symbols("x, y, rho")

    u =   _V0 * sympy.cos(_x/_L) * sympy.sin(_y/_L)  # noqa: E222
    v = - _V0 * sympy.sin(_x/_L) * sympy.cos(_y/_L)
    p = - _rho * _V0**2 * (sympy.cos(2*_x/_L) + sympy.cos(2*_y/_L)) / 4

    u = u.subs([(_x, x), (_y, y), (_rho, rho)]).evalf()
    v = v.subs([(_x, x), (_y, y), (_rho, rho)]).evalf()
    p = p.subs([(_x, x), (_y, y), (_rho, rho)]).evalf()

    return u, v, p


def get_computational_graph(cfg: _ModulusConfig):
    """Returns the computational graph as a list of nodes.
    """

    # set up scales
    scales = dict({key: process_domain(cfg.custom[key]) for key in ["x", "y", "t"]})

    # set up periodicity
    periodicity = {key: process_domain(cfg.custom[key]) for key in cfg.custom.periodic}

    pde = IncompNavierStokes(cfg.custom.nu, cfg.custom.rho, 2, True)

    net = _FullyConnectedArch(
        input_keys=[_Key(key, scale=scales[key]) for key in ["x", "y", "t"]],
        output_keys=[_Key("u"), _Key("v"), _Key("p")],
        periodicity=periodicity,
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    nodes = pde.make_nodes() + [net.make_node(name="flow-net", jit=cfg.jit)]

    return nodes, net


def get_computational_domain(cfg: _ModulusConfig):
    """Get a geometry object representing the computational domain.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.x)
    geo = _Rectangle((xbg, ybg), (xed, yed))

    return geo


def get_initial_constraint(nodes, geo, cfg):
    """Get an interior pointwise constraint for initial conditions (i.e., t=0)
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.x)
    tbg, _ = process_domain(cfg.custom.t)

    x, y, t = sympy.symbols("x, y, t")
    u0, v0, p0 = get_symbolic_ic(1.0, 1.0, rho=cfg.custom.rho)

    constraint = _StepAwarePointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": u0, "v": v0, "p": p0},
        batch_size=cfg.batch_size.npts,
        batch_per_epoch=cfg.batch_size.nbatches,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: tbg}
    )

    return constraint


def get_pde_constraint(nodes, geo, cfg):
    """Get an interior constraint for PDE residuals.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.x)
    tbg, ted = process_domain(cfg.custom.t)

    x, y, t = sympy.symbols("x, y, t")

    constraint = _StepAwarePointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0},
        batch_size=cfg.batch_size.npts,
        batch_per_epoch=cfg.batch_size.nbatches,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: (tbg, ted)}
    )

    return constraint


def get_residual_monitor(nodes, geo, cfg):
    """Get a monitor to record PDE residuals.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.x)
    tbg, ted = process_domain(cfg.custom.t)

    x, y, t = sympy.symbols("x, y, t")

    data = geo.sample_interior(
        nr_points=cfg.batch_size.npts,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: (tbg, ted)}
    )

    monitor = _PointwiseMonitor(
        invar=data,
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "continuity_res": lambda var: _torchsum(var["area"] * _torchabs(var["continuity"])),
            "momentum_x_res": lambda var: _torchsum(var["area"] * _torchabs(var["momentum_x"])),
            "momentum_y_res": lambda var: _torchsum(var["area"] * _torchabs(var["momentum_y"])),
        },
        nodes=nodes,
    )

    return monitor


def get_ic_monitor(nodes, geo, cfg):
    """Get monitor against initial solutions.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.x)

    x, y, t = sympy.symbols("x, y, t")

    indata = geo.sample_interior(
        nr_points=cfg.batch_size.npts,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: 0.0}
    )

    uic, vic, pic = get_symbolic_ic(1.0, 1.0, x="x", y="y", rho=1.0)

    uic = sympy.lambdify([x, y], uic)(indata["x"], indata["y"])
    vic = sympy.lambdify([x, y], vic)(indata["x"], indata["y"])
    pic = sympy.lambdify([x, y], pic)(indata["x"], indata["y"])

    monitor = _PointwiseMonitor(
        invar=indata,
        output_names=["u", "v", "p"],
        metrics={
            "u_res": lambda var: _torchsum(var["area"].data.cpu() * _torchabs(var["u"].data.cpu()-uic)),
            "v_res": lambda var: _torchsum(var["area"].data.cpu() * _torchabs(var["v"].data.cpu()-vic)),
            "p_res": lambda var: _torchsum(var["area"].data.cpu() * _torchabs(var["p"].data.cpu()-pic)),
        },
        nodes=nodes,
    )

    return monitor


def get_solver_domains(nodes, geo, cfg):
    """Get a modules' Domain for the first time window.

    No BC constraints needed because they are periodic BCs and are handled by the neural network.
    """

    domain = _Domain()

    # true IC, i.e., at t = 0 sec, for the first window
    domain.add_constraint(get_initial_constraint(nodes, geo, cfg), name="ic")

    # interior PDE residuals (same for both)
    domain.add_constraint(get_pde_constraint(nodes, geo, cfg), name="residual")

    # inferencers (save the model params and timestamps)
    domain.add_inferencer(inferencer=SaveModelInferencer(nodes, "flow-net"))

    # add validator against initial solution
    domain.add_monitor(get_ic_monitor(nodes, geo, cfg), name="ic-residual")

    # add monitored quanties to tensorboard
    domain.add_monitor(get_residual_monitor(nodes, geo, cfg), name="pde-residual")

    return domain


def main(cfg: _ModulusConfig):
    """The main function.
    """
    nodes, _ = get_computational_graph(cfg)
    geo = get_computational_domain(cfg)
    domain = get_solver_domains(nodes, geo, cfg)
    solver = LBFGSSolver(cfg=cfg, domain=domain)
    solver.solve()
    return 0


if __name__ == "__main__":
    import modulus
    root = pathlib.Path(__file__).resolve().parent

    # re-make the cmd arguments with the program name and desired workdir
    sys.argv = sys.argv[:1] + [r"hydra.run.dir=outputs"]

    # modulus.main is a function wrapper to help loading configs
    sys.exit(modulus.main(str(root), "config")(main)())
