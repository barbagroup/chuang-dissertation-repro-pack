#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""2D Cylinder flow Re=200, unsteady solver.
"""
# pylint: disable=invalid-name, relative-beyond-top-level, import-error
import sys
import pathlib
import sympy
from modulus.key import Key as _Key
from modulus.hydra.config import ModulusConfig as _ModulusConfig
from modulus.geometry.csg.csg_2d import Rectangle as _Rectangle
from modulus.geometry.csg.csg_2d import Circle as _Circle
from modulus.architecture.fully_connected import FullyConnectedArch as _FullyConnectedArch
from modulus.continuous.domain.domain import Domain as _Domain
from modulus.PDES.navier_stokes import NavierStokes
from omegaconf.errors import ConfigAttributeError as _ConfigAttributeError

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.aggregators import register_loss_configs
        from helpers.schedulers import register_scheduler_configs
        from helpers.solvers import AdamNCGSWA
        from helpers.solvers import register_optimizer_configs
        from helpers.utils import process_domain
        from helpers.utils import get_activation_fn
        from helpers.constraints import StepAwarePointwiseInteriorConstraint
        from helpers.constraints import StepAwarePointwiseBoundaryConstraint
        register_loss_configs()
        register_scheduler_configs()
        register_optimizer_configs()
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")


def get_computational_graph(cfg: _ModulusConfig):
    """Returns the computational graph as a list of nodes.
    """

    # set up periodicity
    try:
        periodicity = {key: process_domain(cfg.custom[key]) for key in cfg.custom.periodic}
    except _ConfigAttributeError:
        periodicity = None

    pde = NavierStokes(cfg.custom.nu, cfg.custom.rho, 2, True)

    net = _FullyConnectedArch(
        input_keys=[_Key(key) for key in ["x", "y", "t"]],
        output_keys=[_Key("u"), _Key("v"), _Key("p")],
        periodicity=periodicity,
        activation_fn=get_activation_fn(cfg.custom.activation),
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    nodes = pde.make_nodes() + [net.make_node(name="flow-net", jit=cfg.jit)]

    return nodes, net


def get_computational_domain(cfg: _ModulusConfig):
    """Get a geometry object representing the computational domain.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    radius = sympy.sympify(cfg.custom.radius).evalf()
    geo = _Rectangle((xbg, ybg), (xed, yed)) - _Circle((0, 0), radius)

    return geo


def get_initial_constraint(nodes, geo, cfg):
    """Get an interior pointwise constraint for initial conditions (i.e., t=0)
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    tbg, _ = process_domain(cfg.custom.t)

    x, y, t = sympy.symbols("x, y, t")

    constraint = StepAwarePointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": float(cfg.custom.uic), "v": 0.0, "p": 0.0},
        batch_size=cfg.batch_size.nic,
        batch_per_epoch=cfg.batch_size.nbatches,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: tbg},
        quasirandom=True,
    )

    return constraint


def get_boundary_constraints(nodes, geo, cfg):
    """Get BCs.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    tbg, ted = process_domain(cfg.custom.t)
    radius = sympy.sympify(cfg.custom.radius).evalf()  # 0.5

    x, y, t = sympy.symbols("x, y, t")

    # inlet
    inlet = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.nbcy,
        batch_per_epoch=cfg.batch_size.nbatches,
        criteria=sympy.Eq(x, xbg),
        param_ranges={t: (tbg, ted)},
        quasirandom=True,
    )

    # outlet: zero pressure
    outlet = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p": 0.0},
        batch_size=cfg.batch_size.nbcy,
        batch_per_epoch=cfg.batch_size.nbatches,
        criteria=sympy.Eq(x, xed),
        param_ranges={t: (tbg, ted)},
        quasirandom=True,
    )

    # bottom
    bottom = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.nbcx,
        batch_per_epoch=cfg.batch_size.nbatches,
        criteria=sympy.Eq(y, ybg),
        param_ranges={t: (tbg, ted)},
        quasirandom=True,
    )

    # top
    top = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.nbcx,
        batch_per_epoch=cfg.batch_size.nbatches,
        criteria=sympy.Eq(y, yed),
        param_ranges={t: (tbg, ted)},
        quasirandom=True,
    )

    # noslip
    noslip = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.ncylinder,
        batch_per_epoch=cfg.batch_size.nbatches,
        criteria=sympy.And(
            sympy.Ge(x, -radius), sympy.Le(x, radius),
            sympy.Ge(y, -radius), sympy.Le(y, radius)
        ),
        param_ranges={t: (tbg, ted)},
        quasirandom=True,
    )

    return {"inlet": inlet, "outlet": outlet, "top": top, "bottom": bottom, "noslip": noslip}


def get_pde_constraint(nodes, geo, cfg):
    """Get an interior constraint for PDE residuals.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    tbg, ted = process_domain(cfg.custom.t)

    x, y, t = sympy.symbols("x, y, t")

    constraint = StepAwarePointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0.0, "momentum_x": 0.0, "momentum_y": 0.0},
        batch_size=cfg.batch_size.npts,
        batch_per_epoch=cfg.batch_size.nbatches,
        bounds={x: (xbg, xed), y: (ybg, yed)},
        param_ranges={t: (tbg, ted)},
        quasirandom=True,
    )

    return constraint


def get_solver_domains(nodes, geo, cfg):
    """Get a modules' Domain for the first time window.

    No BC constraints needed because they are periodic BCs and are handled by the neural network.
    """

    domain = _Domain()

    # IC, i.e., at t = 0 sec
    domain.add_constraint(get_initial_constraint(nodes, geo, cfg), name="ic")

    # boundary conditions
    for name, bc in get_boundary_constraints(nodes, geo, cfg).items():  # pylint: disable=invalid-name
        domain.add_constraint(bc, name=name)

    # interior PDE residuals (same for both)
    domain.add_constraint(get_pde_constraint(nodes, geo, cfg), name="residual")

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
    sys.argv = sys.argv[:1] + [r"hydra.run.dir=outputs"]

    # modulus.main is a function wrapper to help loading configs
    sys.exit(modulus.main(str(root), "config")(main)())
