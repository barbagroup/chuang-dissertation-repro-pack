#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot convergence history.
"""
import itertools
import sys
import pathlib
import pandas
from matplotlib import pyplot

# find helpers
for parent in pathlib.Path(__file__).resolve().parents:
    if parent.joinpath("helpers").is_dir():
        sys.path.insert(0, str(parent))
        from helpers.utils import read_tensorboard_data  # pylint: disable=import-error  # noqa: E402
        break
else:
    raise FileNotFoundError("Couldn't find module `helpers`.")

# matplotlib configuration
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["P052", "Pagella", "Palatino", "Palatino Linotype", "Times New Roman"],
    "figure.constrained_layout.use": True,
    "figure.dpi": 166,
})


def gather_data(workdir, jobs):
    """Put and save all training history into a big pandas dataframe.
    """

    workdir.joinpath("outputs").mkdir(exist_ok=True)

    data = []
    for job in jobs:
        print(f"Reading and parsing {job}")
        data.append(read_tensorboard_data(workdir.joinpath(job, "outputs")))
    data = pandas.concat(data, axis=1, keys=jobs)
    data.to_csv(workdir.joinpath("outputs", "training-hist.csv"))

    return data


def fixed_arch_hist(data, figdir, key, ylabel, nls, nns, nptss):
    """Training history with a fixed architecture.
    """

    # separate the adam and ncg stage
    adam = data[data.index <= 100000]
    ncg = data[data.index >= 100000]

    # plots
    sys.stdout.write(f"plotting fixed-{key}:")
    for nl, nn in itertools.product(nls, nns):
        lines = []
        fig, axs = pyplot.subplots(2, 1, figsize=(6, 5))
        for npts in nptss:
            job = f"nl{nl}-nn{nn}-npts{npts}"
            sys.stdout.write(f"{job} ")
            sys.stdout.flush()

            # adam
            data = adam[(job, key)].dropna()
            lines.append(axs[0].semilogy(data.index, data.ewm(span=10).mean(), alpha=0.6, label=npts)[0])
            axs[0].set_title("Adam stage")
            axs[0].set_xlabel("Iteration")
            axs[0].set_ylabel(ylabel)

            # ncg
            data = ncg[(job, key)].dropna()
            axs[1].semilogy(data.index, data.ewm(span=10).mean(), alpha=0.6, color=lines[-1].get_color())
            axs[1].set_title("CG stage")
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel(ylabel)

        leg = fig.legend(handles=lines, title="Batch size", loc="upper left", bbox_to_anchor=(1.0, 0.75))
        leg.set_in_layout(True)
        fig.suptitle(f"Training history: $N_l={nl}$, $N_n={nn}$")
        fig.savefig(figdir.joinpath(f"fixed-arch-{key}-nl{nl}-nn{nn}.png"), bbox_inches="tight")

        # delete the figure object
        pyplot.close(fig)

    # shift line in stdout
    sys.stdout.write("\n")


def main(workdir, figdir, force):
    """main
    """

    nls = range(1, 4)
    nns = [2**i for i in range(4, 9)]
    nptss = [2**i for i in range(10, 17)]

    if not workdir.joinpath("outputs", "training-hist.csv").is_file() or force:
        data = gather_data(
            workdir,
            [f"nl{nl}-nn{nn}-npts{npts}" for nl, nn, npts in itertools.product(nls, nns, nptss)]
        )
    else:
        data = pandas.read_csv(workdir.joinpath("outputs", "training-hist.csv"), index_col=0, header=[0, 1])

    # training histories, with data from same arch in the same figure
    fixed_arch_hist(data, figdir, "loss", "Aggregated loss", nls, nns, nptss)
    fixed_arch_hist(data, figdir, "cont_res", "Continuty residual", nls, nns, nptss)
    fixed_arch_hist(data, figdir, "momem_x_res", "x-momentum residual", nls, nns, nptss)
    fixed_arch_hist(data, figdir, "momem_y_res", "y-momentum residual", nls, nns, nptss)

    return 0


if __name__ == "__main__":
    import argparse

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("taylor-green-vortex-2d-re100").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `modulus`.")

    root = root.joinpath("taylor-green-vortex-2d-re100")
    figdir = root.joinpath("figures", "training-hist")
    figdir.mkdir(exist_ok=True)

    # cmd arguments
    parser = argparse.ArgumentParser(description="Post-processing Modulus Cylinder 2D Re200")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    # calling the main function
    sys.exit(main(root, figdir, args.force))
