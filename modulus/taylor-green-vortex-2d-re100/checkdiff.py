#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Check the difference in config.yaml across different folders.
"""
import pathlib
import deepdiff
from omegaconf import OmegaConf


root = pathlib.Path(__file__).resolve().parent


baseconf = OmegaConf.load(root.joinpath("nl1-nn16-npts1024", "config.yaml"))

for nl in [1, 2, 3]:
    for nn in [16, 32, 64, 128, 256]:
        for npts in [2**i for i in range(10, 17)]:
            target = OmegaConf.load(root.joinpath(f"nl{nl}-nn{nn}-npts{npts}", "config.yaml"))
            result = deepdiff.DeepDiff(baseconf, target)

            anslen = 3

            if (nl, nn, npts) == (1, 16, 1024):
                continue

            if nl == 1:
                anslen -= 1

            if nn == 16:
                anslen -= 1

            if npts == 1024:
                anslen -= 1

            assert len(result["values_changed"]) == anslen, f"(nl, nn, npts) = ({nl}, {nn}, {npts})\n{result}"

            if nl != 1:
                assert result["values_changed"]["root['arch']['fully_connected']['nr_layers']"]["new_value"] == nl

            if nn != 16:
                assert result["values_changed"]["root['arch']['fully_connected']['layer_size']"]["new_value"] == nn

            if npts != 1024:
                assert result["values_changed"]["root['batch_size']['npts']"]["new_value"] == npts
