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
import sympy
from torch import sum as _torchsum  # pylint: disable=no-name-in-module
from torch import abs as _torchabs  # pylint: disable=no-name-in-module
from torch import zeros_like as _torchzeroslike  # pylint: disable=no-name-in-module
from torch import tensor as _torchtensor  # pylint: disable=no-name-in-module
from torch import arange as _torcharange  # pylint: disable=no-name-in-module
from torch import meshgrid as _torchmeshgrid  # pylint: disable=no-name-in-module
from modulus.key import Key as _Key
from modulus.hydra.config import ModulusConfig as _ModulusConfig
from modulus.geometry.csg.csg_2d import Rectangle as _Rectangle
from modulus.geometry.csg.csg_2d import Circle as _Circle
from modulus.architecture.fully_connected import FullyConnectedArch as _FullyConnectedArch
from modulus.continuous.domain.domain import Domain as _Domain
from modulus.continuous.monitor.monitor import PointwiseMonitor as _PointwiseMonitor
from modulus.PDES.navier_stokes import NavierStokes
from omegaconf.errors import ConfigAttributeError as _ConfigAttributeError

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.pdes import ConvectiveBC  # pylint: disable=import-error
        from helpers.inferencers import SaveModelInferencer  # pylint: disable=import-error
        from helpers.aggregators import register_loss_configs  # pylint: disable=import-error
        from helpers.schedulers import register_scheduler_configs  # pylint: disable=import-error
        from helpers.solvers import AdamNCGSWA  # pylint: disable=import-error
        from helpers.solvers import register_optimizer_configs  # pylint: disable=import-error
        from helpers.utils import process_domain  # pylint: disable=import-error
        from helpers.utils import get_activation_fn  # pylint: disable=import-error
        from helpers.constraints import StepAwarePointwiseConstraint  # pylint: disable=import-error
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

    # set up scales
    scales = dict({key: process_domain(cfg.custom[key]) for key in ["x", "y", "t"]})

    # set up periodicity
    try:
        periodicity = {key: process_domain(cfg.custom[key]) for key in cfg.custom.periodic}
    except _ConfigAttributeError:
        periodicity = None

    pde = NavierStokes(cfg.custom.nu, cfg.custom.rho, 2, True)

    net = _FullyConnectedArch(
        input_keys=[_Key(key, scale=scales[key]) for key in ["x", "y", "t"]],
        output_keys=[_Key("u"), _Key("v"), _Key("p")],
        periodicity=periodicity,
        activation_fn=get_activation_fn(cfg.custom.activation),
        **{k: v for k, v in cfg.arch.fully_connected.items() if k != "_target_"}
    )

    u_outlet = ConvectiveBC("u", 1.0, "convective_u", 2, True)
    v_outlet = ConvectiveBC("v", 1.0, "convective_v", 2, True)

    nodes = \
        pde.make_nodes() + u_outlet.make_nodes() + v_outlet.make_nodes() + \
        [net.make_node(name="flow-net", jit=cfg.jit)]

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
        lambda_weighting={"u": geo.sdf, "v": geo.sdf, "p": geo.sdf},
        param_ranges={t: tbg}
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
        criteria=sympy.Eq(x, xbg),
        param_ranges={t: (tbg, ted)}
    )

    # outlet: convective BC
    outlet = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"convective_u": 0.0, "convective_v": 0.0},
        batch_size=cfg.batch_size.nbcy,
        criteria=sympy.Eq(x, xed),
        param_ranges={t: (tbg, ted)}
    )

    # bottom
    bottom = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.nbcx,
        criteria=sympy.Eq(y, ybg),
        param_ranges={t: (tbg, ted)}
    )

    # top
    top = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1.0, "v": 0.0},
        batch_size=cfg.batch_size.nbcx,
        criteria=sympy.Eq(y, yed),
        param_ranges={t: (tbg, ted)}
    )

    # noslip
    noslip = StepAwarePointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.ncylinder,
        criteria=sympy.And(
            sympy.Ge(x, -radius), sympy.Le(x, radius),
            sympy.Ge(y, -radius), sympy.Le(y, radius)
        ),
        param_ranges={t: (tbg, ted)}
    )

    # reference pressure
    refx = _torchtensor([xed])
    refy = _torchtensor([0.0])
    reft = _torcharange(float(tbg), float(ted+1))
    refx, refy, reft = _torchmeshgrid(refx, refy, reft, indexing="xy")
    refp = _torchzeroslike(refx)

    refx = refx.reshape(-1, 1).numpy()
    refy = refy.reshape(-1, 1).numpy()
    reft = reft.reshape(-1, 1).numpy()
    refp = refp.reshape(-1, 1).numpy()

    refp = StepAwarePointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": refx, "y": refy, "t": reft},
        outvar={"p": refp},
        batch_size=refx.size,
    )

    return {
        "inlet": inlet, "outlet": outlet, "top": top, "bottom": bottom, "noslip": noslip,
        "refp": refp,
    }


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
        lambda_weighting={"continuity": geo.sdf, "momentum_x": geo.sdf, "momentum_y": geo.sdf},
        param_ranges={t: (tbg, ted)}
    )

    return constraint


def get_residual_monitors(nodes, geo, cfg):
    """Get a monitor to record PDE residuals.
    """

    xbg, xed = process_domain(cfg.custom.x)
    ybg, yed = process_domain(cfg.custom.y)
    tbg, ted = process_domain(cfg.custom.t)
    radius = sympy.sympify(cfg.custom.radius).evalf()

    x, y, t = sympy.symbols("x, y, t")

    pde = _PointwiseMonitor(
        invar=geo.sample_interior(
            nr_points=cfg.batch_size.npts,
            bounds={x: (xbg, xed), y: (ybg, yed)},
            param_ranges={t: (tbg, ted)}
        ),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "continuity": lambda var: _torchsum(var["area"] * _torchabs(var["continuity"])),
            "momentum_x": lambda var: _torchsum(var["area"] * _torchabs(var["momentum_x"])),
            "momentum_y": lambda var: _torchsum(var["area"] * _torchabs(var["momentum_y"])),
        },
        nodes=nodes,
    )

    ic = _PointwiseMonitor(
        invar=geo.sample_interior(
            nr_points=cfg.batch_size.nic,
            bounds={x: (xbg, xed), y: (ybg, yed)},
            param_ranges={t: 0.0}
        ),
        output_names=["u", "v", "p"],
        metrics={
            "ic_u": lambda var: _torchsum(var["area"] * _torchabs(var["u"]-cfg.custom.uic)),
            "ic_v": lambda var: _torchsum(var["area"] * _torchabs(var["v"])),
            "ic_p": lambda var: _torchsum(var["area"] * _torchabs(var["p"])),
        },
        nodes=nodes,
    )

    freestream = _PointwiseMonitor(
        invar=geo.sample_boundary(
            nr_points=cfg.batch_size.nbcy+cfg.batch_size.nbcx+cfg.batch_size.nbcx,
            criteria=sympy.Or(sympy.Eq(x, xbg), sympy.Eq(y, ybg), sympy.Eq(y, yed)),
            param_ranges={t: (tbg, ted)},
        ),
        output_names=["u", "v"],
        metrics={
            "freestream_u": lambda var: _torchsum(var["area"] * _torchabs(var["u"]-1.0)),
            "freestream_v": lambda var: _torchsum(var["area"] * _torchabs(var["v"])),
        },
        nodes=nodes,
    )

    noslip = _PointwiseMonitor(
        invar=geo.sample_boundary(
            nr_points=cfg.batch_size.ncylinder,
            criteria=sympy.And(
                sympy.Ge(x, -radius), sympy.Le(x, radius),
                sympy.Ge(y, -radius), sympy.Le(y, radius)
            ),
            param_ranges={t: (tbg, ted)}
        ),
        output_names=["u", "v"],
        metrics={
            "cylinder_u": lambda var: _torchsum(var["area"] * _torchabs(var["u"])),
            "cylinder_v": lambda var: _torchsum(var["area"] * _torchabs(var["v"])),
        },
        nodes=nodes,
    )

    outlet = _PointwiseMonitor(
        invar=geo.sample_boundary(
            nr_points=cfg.batch_size.nbcy,
            criteria=sympy.Eq(x, xed),
            param_ranges={t: (tbg, ted)},
        ),
        output_names=["convective_u", "convective_v"],
        metrics={
            "outlet_conv_u": lambda var: _torchsum(var["area"] * _torchabs(var["convective_u"])),
            "outlet_conv_v": lambda var: _torchsum(var["area"] * _torchabs(var["convective_v"])),
        },
        nodes=nodes,
    )

    return {
        "pde-residual": pde,
        "ic-residual": ic,
        "freestreem-residual": freestream,
        "noslip-residual": noslip,
        "outlet-residual": outlet
    }


def get_solver_domains(nodes, geo, cfg):
    """Get a modules' Domain for the first time window.

    No BC constraints needed because they are periodic BCs and are handled by the neural network.
    """

    domain = _Domain()

    # true IC, i.e., at t = 0 sec, for the first window
    domain.add_constraint(get_initial_constraint(nodes, geo, cfg), name="ic")

    # boundary conditions
    for name, bc in get_boundary_constraints(nodes, geo, cfg).items():  # pylint: disable=invalid-name
        domain.add_constraint(bc, name=name)

    # interior PDE residuals (same for both)
    domain.add_constraint(get_pde_constraint(nodes, geo, cfg), name="residual")

    # inferencers (save the model params and timestamps)
    domain.add_inferencer(inferencer=SaveModelInferencer(nodes, "flow-net"))

    # add monitored quanties to tensorboard
    for name, monitor in get_residual_monitors(nodes, geo, cfg).items():
        domain.add_monitor(monitor, name=name)

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