#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Custom constraints that only use new batches data when requested.
"""
from sympy import Basic as _spbasic
from typing import Tuple as _Tuple
from typing import List as _List
from typing import Dict as _Dict
from typing import Union as _Union
from typing import Callable as _Callable
from modulus.continuous.constraints.constraint import PointwiseConstraint as _PointwiseConstraint
from modulus.continuous.constraints.constraint import PointwiseBoundaryConstraint as _PointwiseBoundaryConstraint
from modulus.continuous.constraints.constraint import PointwiseInteriorConstraint as _PointwiseInteriorConstraint
from modulus.node import Node as _Node
from modulus.loss import Loss as _Loss
from modulus.loss import PointwiseLossNorm as _PointwiseLossNorm
from modulus.constraint import Constraint as _Constraint
from modulus.geometry.geometry import Geometry as _Geometry


class StepAwarePointwiseConstraint(_PointwiseConstraint):
    """A derived PointwiseConstraint that only uses a new batch of data when provided step differs.
    """

    def __init__(self, *args, **kwargs):
        # meaning this instance is not used as a plugin, so we need to initialize the instance first
        if not hasattr(self, "model"):
            super().__init__(*args, **kwargs)

        # plug in step-aware attributes
        self.step = -1
        self.cur_invar = None
        self.true_outvar = None
        self.lambda_weighting = None

    def loss(self, step: int):

        # get train points from dataloader only when the porvided step value is different
        if step != self.step:
            self.invar, self.true_outvar, self.lambda_weighting = next(self.dataloader)
            self.invar = _Constraint._set_device(self.invar, requires_grad=True)
            self.true_outvar = _Constraint._set_device(self.true_outvar)
            self.lambda_weighting = _Constraint._set_device(self.lambda_weighting)
            self.step = step

        # compute pred outvar
        pred_outvar = self.model(self.invar)

        # compute loss
        losses = self._loss(self.invar, pred_outvar, self.true_outvar, self.lambda_weighting, step)

        return losses


class StepAwarePointwiseBoundaryConstraint(_PointwiseBoundaryConstraint, StepAwarePointwiseConstraint):
    """A derived PointwiseBoundaryConstraint that only uses a new batch of data when provided step differs.

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.

    geometry : Geometry
        Modulus `Geometry` to apply the constraint with.

    outvar : Dict[str, Union[int, float, sp.Basic]]
        To describe the constraints with respect to different output variables from the computational graph.

    batch_size : int
        Batch size used in training.

    criteria : Union[sp.basic, True]
        SymPy criteria function for limiting the ranges of input variables that this constraint instance applies.

    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The spatial pointwise weighting of the constraint.

    param_ranges : Dict[sp.Basic, Tuple[float, float]] = {}
        This allows adding parameterization or additional inputs.

    fixed_dataset : bool = True
        If True then the points sampled for this constraint are done right when initialized and fixed.

    importance_measure : Union[Callable, None] = None
        A callable function that computes a scalar importance measure.

    batch_per_epoch : int = 1000
        If True, the total number of points generated is `total_nr_points=batch_per_epoch*batch_size`.

    quasirandom : bool = False
        If true then sample the points using the Halton sequence.

    num_workers : int
        Number of worker used in fetching data.

    loss : Loss
        Modulus `Loss` module that defines the loss type, (e.g. L2, L1, ...).
    """

    def __init__(
        self,
        nodes: _List[_Node],
        geometry: _Geometry,
        outvar: _Dict[str, _Union[int, float, _spbasic]],
        batch_size: int,
        criteria: _spbasic = True,
        lambda_weighting: _Dict[str, _Union[int, float, _spbasic]] = None,
        param_ranges: _Dict[_spbasic, _Tuple[float, float]] = {},
        fixed_dataset: bool = True,
        importance_measure: _Union[_Callable, None] = None,
        batch_per_epoch: int = 1000,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: _Loss = _PointwiseLossNorm(),
    ):
        # the base classes from Modulus were not well designed, so we cannot use MRO to auto construct
        _PointwiseBoundaryConstraint.__init__(
            self, nodes, geometry, outvar, batch_size, criteria, lambda_weighting, param_ranges,
            fixed_dataset, importance_measure, batch_per_epoch, quasirandom, num_workers, loss,
        )

        # the base classes from Modulus were not well designed, so we cannot use MRO to auto construct
        StepAwarePointwiseConstraint.__init__(self)


class StepAwarePointwiseInteriorConstraint(_PointwiseInteriorConstraint, StepAwarePointwiseConstraint):
    """A derived PointwiseInteriorConstraint that only uses a new batch of data when provided step differs.

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.

    geometry : Geometry
        Modulus `Geometry` to apply the constraint with.

    outvar : Dict[str, Union[int, float, sp.Basic]]
        To describe the constraints with respect to different output variables from the computational graph.

    batch_size : int
        Batch size used in training.

    bounds : Dict[sp.Basic, Tuple[float, float]] = None
        Bounds of the given geometry, (e.g. `bounds={sympy.Symbol('x'): (0, 1), sympy.Symbol('y'): (0, 1)}).

    criteria : Union[sp.basic, True]
        SymPy criteria function for limiting the ranges of input variables that this constraint instance applies.

    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The spatial pointwise weighting of the constraint.

    param_ranges : Dict[sp.Basic, Tuple[float, float]] = {}
        This allows adding parameterization or additional inputs.

    fixed_dataset : bool = True
        If True then the points sampled for this constraint are done right when initialized and fixed.

    importance_measure : Union[Callable, None] = None
        A callable function that computes a scalar importance measure.

    batch_per_epoch : int = 1000
        If True, the total number of points generated is `total_nr_points=batch_per_epoch*batch_size`.

    quasirandom : bool = False
        If true then sample the points using the Halton sequence.

    num_workers : int
        Number of worker used in fetching data.

    loss : Loss
        Modulus `Loss` module that defines the loss type, (e.g. L2, L1, ...).
    """

    def __init__(
        self,
        nodes: _List[_Node],
        geometry: _Geometry,
        outvar: _Dict[str, _Union[int, float, _spbasic]],
        batch_size: int,
        bounds: _Dict[_spbasic, _Tuple[float, float]] = None,
        criteria: _spbasic = True,
        lambda_weighting: _Dict[str, _Union[int, float, _spbasic]] = None,
        param_ranges: _Dict[_spbasic, _Tuple[float, float]] = {},
        fixed_dataset: bool = True,
        importance_measure: _Union[_Callable, None] = None,
        batch_per_epoch: int = 1000,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: _Loss = _PointwiseLossNorm(),
    ):
        # the base classes from Modulus were not well designed, so we cannot use MRO to auto construct
        _PointwiseInteriorConstraint.__init__(
            self, nodes, geometry, outvar, batch_size, bounds, criteria, lambda_weighting, param_ranges,
            fixed_dataset, importance_measure, batch_per_epoch, quasirandom, num_workers, loss,
        )

        # the base classes from Modulus were not well designed, so we cannot use MRO to auto construct
        StepAwarePointwiseConstraint.__init__(self)
