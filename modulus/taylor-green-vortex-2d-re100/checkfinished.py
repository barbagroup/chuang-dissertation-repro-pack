#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Check if each case has finished.
"""
import pathlib
from termcolor import colored
root = pathlib.Path(__file__).resolve().parent

for nl in [1, 2, 3]:
    for nn in [16, 32, 64, 128, 256]:
        for npts in [2**i for i in range(10, 17)]:
            print(f"nl{nl}-nn{nn}-npts{npts}: ", end="")
            finished = False
            workdir = root.joinpath(f"nl{nl}-nn{nn}-npts{npts}", "logs")
            logs = workdir.glob("run-*.log")
            for log in logs:
                with open(log, "r") as fobj:
                    data = fobj.readlines()

                for line in data[-10:]:
                    if "finished training" in line:
                        finished = True
                        break

                if finished:
                    break

            print(colored(str(finished), "blue" if finished else "red"))
