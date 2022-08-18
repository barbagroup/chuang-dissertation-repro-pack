#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create cases
"""
import pathlib
import itertools


def _copy(filepath, content, force, kwargs):
    """copy
    """

    if not filepath.is_file() or force:
        with open(filepath, "w") as fobj:
            if kwargs is None:
                fobj.write(content)
            else:
                fobj.write(content.format(**kwargs))
    else:
        print(f"Skipped {filepath}")


def create_base_cases(workdir, force=False):
    """Create base-cases.
    """

    with open(workdir.joinpath("templates", "exp-sum-config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(workdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(workdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    with open(workdir.joinpath("templates", "no-singularity.sh.temp"), "r") as fobj:
        noslurm_sh = fobj.read()

    nns = [16, 32, 64, 128, 256]
    nls = [1, 2, 3, 4]
    nptss = [2**i for i in range(10, 17)]

    for nl, nn, npts in itertools.product(nls, nns, nptss):
        path = workdir.joinpath("base-cases", f"nl{nl}-nn{nn}-npts{npts}")

        if not path.is_dir() or force:
            path.mkdir(parents=True, exist_ok=True)

        _copy(
            path.joinpath("config.yaml"), config_yaml, force,
            {"nr_layers": nl, "layer_size": nn, "npts": npts}
        )

        _copy(path.joinpath("main.py"), main_py, force, None)

        _copy(
            path.joinpath("job.sh"), job_sh, force,
            {"ngpus": 1, "ncpus": 32, "partition": "dgxa100_80g_2tb", "njobs": 6}
        )

        _copy(path.joinpath("no-singularity.sh"), noslurm_sh, force, None)


def create_exp_sum_scaling(workdir, force=False):
    """Create base-cases.
    """

    with open(workdir.joinpath("templates", "exp-sum-config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(workdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(workdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    with open(workdir.joinpath("templates", "no-singularity.sh.temp"), "r") as fobj:
        noslurm_sh = fobj.read()

    for ngpus in [2, 4, 8]:
        path = workdir.joinpath("exp-sum-scaling", f"nl3-nn128-npts8192-ngpus{ngpus}")

        if not path.is_dir() or force:
            path.mkdir(parents=True, exist_ok=True)

        _copy(
            path.joinpath("config.yaml"), config_yaml, force,
            {"nr_layers": 3, "layer_size": 128, "npts": 8192}
        )

        _copy(path.joinpath("main.py"), main_py, force, None)

        _copy(
            path.joinpath("job.sh"), job_sh, force,
            {"ngpus": ngpus, "ncpus": 32, "partition": "dgxa100_80g_2tb", "njobs": 6}
        )

        _copy(path.joinpath("no-singularity.sh"), noslurm_sh, force, None)


def create_exp_annealing_scaling(workdir, force=False):
    """Create base-cases.
    """

    with open(workdir.joinpath("templates", "exp-annealing-config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(workdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(workdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    with open(workdir.joinpath("templates", "no-singularity.sh.temp"), "r") as fobj:
        noslurm_sh = fobj.read()

    for ngpus in [1, 2, 4, 8]:
        path = workdir.joinpath("exp-annealing-scaling", f"nl3-nn128-npts8192-ngpus{ngpus}")

        if not path.is_dir() or force:
            path.mkdir(parents=True, exist_ok=True)

        _copy(
            path.joinpath("config.yaml"), config_yaml, force,
            {"nr_layers": 3, "layer_size": 128, "npts": 8192}
        )

        _copy(path.joinpath("main.py"), main_py, force, None)

        _copy(
            path.joinpath("job.sh"), job_sh, force,
            {"ngpus": ngpus, "ncpus": 32, "partition": "dgxa100_80g_2tb", "njobs": 6}
        )

        _copy(path.joinpath("no-singularity.sh"), noslurm_sh, force, None)


def create_cyclic_sum_scaling(workdir, force=False):
    """Create base-cases.
    """

    with open(workdir.joinpath("templates", "cyclic-sum-config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(workdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(workdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    with open(workdir.joinpath("templates", "no-singularity.sh.temp"), "r") as fobj:
        noslurm_sh = fobj.read()

    for ngpus in [1, 2, 4, 8]:
        path = workdir.joinpath("cyclic-sum-scaling", f"nl3-nn128-npts8192-ngpus{ngpus}")

        if not path.is_dir() or force:
            path.mkdir(parents=True, exist_ok=True)

        _copy(
            path.joinpath("config.yaml"), config_yaml, force,
            {"nr_layers": 3, "layer_size": 128, "npts": 8192}
        )

        _copy(path.joinpath("main.py"), main_py, force, None)

        _copy(
            path.joinpath("job.sh"), job_sh, force,
            {"ngpus": ngpus, "ncpus": 32, "partition": "dgxa100_80g_2tb", "njobs": 6}
        )

        _copy(path.joinpath("no-singularity.sh"), noslurm_sh, force, None)


def create_cyclic_annealing_scaling(workdir, force=False):
    """Create base-cases.
    """

    with open(workdir.joinpath("templates", "cyclic-annealing-config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(workdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(workdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    with open(workdir.joinpath("templates", "no-singularity.sh.temp"), "r") as fobj:
        noslurm_sh = fobj.read()

    for ngpus in [1, 2, 4, 8]:
        path = workdir.joinpath("cyclic-annealing-scaling", f"nl3-nn128-npts8192-ngpus{ngpus}")

        if not path.is_dir() or force:
            path.mkdir(parents=True, exist_ok=True)

        _copy(
            path.joinpath("config.yaml"), config_yaml, force,
            {"nr_layers": 3, "layer_size": 128, "npts": 8192}
        )

        _copy(path.joinpath("main.py"), main_py, force, None)

        _copy(
            path.joinpath("job.sh"), job_sh, force,
            {"ngpus": ngpus, "ncpus": 32, "partition": "dgxa100_80g_2tb", "njobs": 6}
        )

        _copy(path.joinpath("no-singularity.sh"), noslurm_sh, force, None)


def create_ncg_sum_scaling(workdir, force=False):
    """Create base-cases.
    """

    with open(workdir.joinpath("templates", "ncg-sum-config.yaml.temp"), "r") as fobj:
        config_yaml = fobj.read()

    with open(workdir.joinpath("templates", "main.py.temp"), "r") as fobj:
        main_py = fobj.read()

    with open(workdir.joinpath("templates", "job.sh.temp"), "r") as fobj:
        job_sh = fobj.read()

    with open(workdir.joinpath("templates", "no-singularity.sh.temp"), "r") as fobj:
        noslurm_sh = fobj.read()

    for nl, nn in itertools.product([2, 3], [64, 128]):
        path = workdir.joinpath("ncg-sum-scaling", f"nl{nl}-nn{nn}-npts8192")

        if not path.is_dir() or force:
            path.mkdir(parents=True, exist_ok=True)

        _copy(
            path.joinpath("config.yaml"), config_yaml, force,
            {"nr_layers": nl, "layer_size": nn, "npts": 8192}
        )

        _copy(path.joinpath("main.py"), main_py, force, None)

        _copy(
            path.joinpath("job.sh"), job_sh, force,
            {"ngpus": 1, "ncpus": 32, "partition": "dgxa100_80g_2tb", "njobs": 6}
        )

        _copy(path.joinpath("no-singularity.sh"), noslurm_sh, force, None)


def main(force: bool):
    """main
    """
    workdir = pathlib.Path(__file__).resolve().parents[1]
    create_base_cases(workdir, force)
    create_exp_sum_scaling(workdir, force)
    create_exp_annealing_scaling(workdir, force)
    create_cyclic_sum_scaling(workdir, force)
    create_cyclic_annealing_scaling(workdir, force)
    create_ncg_sum_scaling(workdir, force)

    return 0


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.force))
