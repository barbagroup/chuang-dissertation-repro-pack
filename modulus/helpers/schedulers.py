#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Custom learning rate schedulers.
"""
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from modulus.hydra.scheduler import SchedulerConf as _SchedulerConf


@dataclass
class ReduceLROnPlateauConf(_SchedulerConf):
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.
    eps: float = 1e-8
    verbose: bool = False


def register_scheduler_configs() -> None:
    """Register for custom learning rate scheduler.
    """
    ConfigStore.instance().store(
        group="scheduler",
        name="ReduceLROnPlateau",
        node=ReduceLROnPlateauConf,
    )
