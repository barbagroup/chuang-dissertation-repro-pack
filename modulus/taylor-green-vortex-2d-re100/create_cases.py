#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create cases
"""
import pathlib
import itertools


def main():
    """main
    """
    curdir = pathlib.Path(__file__).resolve().parent

    with open(curdir.joinpath("templates", "config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(curdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(curdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    nns = [128]
    nls = [1, 2]
    nptss = [2**i for i in range(10, 18)]

    for nl, nn, npts in itertools.product(nls, nns, nptss):
        path = curdir.joinpath(f"nl{nl}-nn{nn}-npts{npts}")
        path.mkdir(exist_ok=True)

        with open(path.joinpath("config.yaml"), "w") as fobj:
            fobj.write(config_yaml.format(nr_layers=nl, layer_size=nn, npts=npts))

        with open(path.joinpath("main.py"), "w") as fobj:
            fobj.write(main_py)

        with open(path.joinpath("job.sh"), "w") as fobj:
            fobj.write(job_sh.format(ngpus=1, ncpus=6, partition="dgx2", njobs=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
