#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Custom aggregators.
"""
import torch
from typing import List as _List
from typing import Dict as _Dict
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from modulus.hydra.loss import LossConf as _LossConf
from modulus.aggregator import Aggregator as _Aggregator
from modulus.derivatives import gradient as _gradient


class NSAnnealingLossAggregator(_Aggregator):
    """Step-awared annealing loss aggragation algorithm for multidimension N-S equations.
    """

    def __init__(self, params, num_losses, update_freq=1, alpha=0.01, eps=1e-8, weights=None):
        super().__init__(params, num_losses, weights)
        self.update_freq: int = update_freq
        self.alpha: float = alpha
        self.eps: float = eps
        self.register_buffer("lambda_ema", torch.ones(self.num_losses, device=self.device))
        self.step: int = -1
        self.dtype = self.params[0].dtype

    def forward(self, losses: _Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """Weights and aggregates the losses using the learning rate annealing algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # initialize the final loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # current step needs to update the weights of loss terms and we haven't done that
        if step % self.update_freq == 0 and step != self.step:

            # record the current step
            self.step = step

            # empty numerator holder
            numerator: torch.Tensor = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            # pde residual loss
            for key in ["continuity", "momentum_x", "momentum_y", "momentum_z"]:

                # only losses other than PDE residuals are weighted
                if key in losses.keys():  # annoying.. but torch.script hates `if not ...: continue` statement

                    # gradients w.r.t. model parameters; each element denotes gradients of a neuron layer
                    grads: _List[torch.Tensor] = _gradient(losses[key], self.params)

                    # initialize a list to hold flattened gradients from all layers
                    flattened: _List[torch.Tensor] = []

                    with torch.no_grad():
                        # flatten and concatnate all parameters' gradients
                        for layer_grad in grads:
                            if layer_grad is not None:
                                # grads are detached from the graph because they will not be involbed in later gradients
                                flattened.append(torch.abs(torch.flatten(layer_grad.detach().data)))

                        # update the numerator
                        numerator += torch.mean(torch.cat(flattened))

                    # update total loss
                    loss += losses[key]

            # compute the mean of each loss gradients
            for i, key in enumerate(losses.keys()):

                # only losses other than PDE residuals are weighted
                if key in ["continuity", "momentum_x", "momentum_y", "momentum_z"]:
                    continue

                # current loss term's derivatives w.r.t. model parameters as a list; each list element denotes a layer
                grads: _List[torch.Tensor] = _gradient(losses[key], self.params)

                # initialize a list to hold flattened gradients from all layers
                flattened: _List[torch.Tensor] = []

                with torch.no_grad():
                    # parameters in each layer may be a matrix; flatten them and combine them into one long 1D vector
                    for layer_grad in grads:
                        if layer_grad is not None:
                            # grads are detached from the graph because they will not be involbed in later gradients
                            flattened.append(torch.abs(torch.flatten(layer_grad.detach().data)))

                    # update the corresponding weight
                    denomenator: torch.Tensor = torch.mean(torch.cat(flattened))  # mean gradients
                    self.lambda_ema[i] *= (1.0 - self.alpha)
                    self.lambda_ema[i] += (self.alpha * numerator / (denomenator + self.eps))

                # update the total loss with this term weighted
                # using .data cauz' lambda_ema won't be involved in later gradient calculations
                loss += (self.lambda_ema[i].detach().data * losses[key])

        # otherwise, add up all losses using previous weights
        else:
            for i, key in enumerate(losses.keys()):
                # using .data cauz' lambda_ema won't be involved in later gradient calculations
                loss += (self.lambda_ema[i].detach().data * losses[key])

        return loss


@dataclass
class NSAnnealingLossAggregatorConf(_LossConf):
    """The Hydra configuration data model.
    """
    _target_: str = "helpers.aggregators.NSAnnealingLossAggregator"
    update_freq: int = 1
    alpha: float = 0.01
    eps: float = 1e-8


def register_loss_configs() -> None:
    """ Register the aggregator classes in this module
    """
    ConfigStore.instance().store(
        group="loss",
        name="NSAnnealingLossAggregator",
        node=NSAnnealingLossAggregatorConf,
    )
