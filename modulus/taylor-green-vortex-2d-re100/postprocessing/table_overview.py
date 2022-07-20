#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Create overview tables.
"""
import itertools
import pathlib
import numpy
import pandas


def timestamps_to_elapsed_times(timestamps, noutliers=10, offset=0.):
    """Converts timestamps to elapsed times and eliminates outliers due to restart.

    Notes
    -----
    Timestamps from different training stages must be processed separately.
    """

    timestamps = numpy.array(timestamps)
    diff = timestamps[1:] - timestamps[:-1]
    truncated = numpy.sort(diff)[noutliers:-noutliers]

    assert len(truncated) != 0, f"{diff}\n{truncated}"
    avg = truncated.mean()
    std = truncated.std(ddof=1)
    diff[numpy.logical_or(diff < avg-2*std, diff > avg+2*std)] = avg
    diff = numpy.concatenate((numpy.full((1,), 0), diff))  # add back the time for the first result

    return numpy.cumsum(diff) + offset  # cumulative times


def create_overview_table(workdir, fname, nls, nns, nbss):
    """Create an overview table.
    """

    def formula(nl, nn):
        ninp = 5  # because of the periodic domain
        return ninp * nn + 2 * nn + (nl - 1) * (nn**2 + 2 * nn) + 3 * nn + 3

    values = []
    for nl, nn, nbs in itertools.product(nls, nns, nbss):
        data = pandas.read_csv(
            workdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}.walltime.csv"), index_col=0, header=[0, 1, 2, 3]
        )

        # earlier data use "raw" rather than "orig"
        data = data.rename(columns=lambda val: "orig" if val == "raw" else val, level=0)

        # clear the sub-column names that contain "Unnamed"
        for lv in [1, 2, 3]:
            data = data.rename(columns=lambda val: "" if "Unnamed" in val else val, level=lv)

        # separate the column of time stamps
        timestamps = data.xs("timestamp", axis=1)
        walltime = timestamps_to_elapsed_times(timestamps[timestamps.index <= 100000].array, 1)
        walltime = timestamps_to_elapsed_times(timestamps[timestamps.index >= 100000].array, 3, walltime[-1])
        walltime = walltime[-1] / 3600.

        result = [nl, nn, formula(nl, nn), nbs, walltime]
        for time, field in itertools.product(["0", "40", "80"], ["u", "v"]):
            result.append(data[("orig", "l2norm", field, time)].iloc[-1])
            result.append(data[("swa", "l2norm", field, time)].iloc[-1])

        values.append(result)

    columns = pandas.MultiIndex.from_tuples(
        [
            ("", "", r"$N_l$"),
            ("", "", r"$N_n$"),
            ("", "", r"DoF"),
            ("", "", r"$N_{bs}$"),
            ("", "Run time", "(hr)"),
            ("t=0", "u", "orig"),
            ("t=0", "u", "swa"),
            ("t=0", "v", "orig"),
            ("t=0", "v", "swa"),
            ("t=40", "u", "orig"),
            ("t=40", "u", "swa"),
            ("t=40", "v", "orig"),
            ("t=40", "v", "swa"),
            ("t=80", "u", "orig"),
            ("t=80", "u", "swa"),
            ("t=80", "v", "orig"),
            ("t=80", "v", "swa"),
        ]
    )

    table = pandas.DataFrame(data=values, columns=columns)
    print(table)

    # adjust output formats
    table = table.style.hide()
    table = table.format("{:7.2e}", ["t=0", "t=40", "t=80"])
    table = table.format("{:.1f}", [("", "Run time", "(hr)")])
    table = table.format(subset=[("", "", r"DoF"), ("", "", r"$N_{bs}$")], thousands=",")

    # output
    table.to_latex(
        buf=fname,
        column_format="c"*len(columns),
        hrules=True,
        label="table:tgv-overview",
        caption=r"Overview, run times, and $l_2$-norms of 2D Taylor-Green vortex benchmarks",
        sparse_index=False,
        multirow_align="c",
        multicol_align="c",
        environment="longtable"
    )


def create_orig_swa_comp_table(workdir, fname, nls, nns, nbss):
    """Create an overview table.
    """

    values = []
    for nl, nn, nbs in itertools.product(nls, nns, nbss):
        data = pandas.read_csv(
            workdir.joinpath(f"nl{nl}-nn{nn}-npts{nbs}.walltime.csv"), index_col=0, header=[0, 1, 2, 3]
        )

        # earlier data use "raw" rather than "orig"
        data = data.rename(columns=lambda val: "orig" if val == "raw" else val, level=0)

        # clear the sub-column names that contain "Unnamed"
        for lv in [1, 2, 3]:
            data = data.rename(columns=lambda val: "" if "Unnamed" in val else val, level=lv)

        result = [nl, nn, nbs]
        for time, field in itertools.product(["0", "40", "80"], ["u", "v"]):
            result.append(data[("orig", "l2norm", field, time)].iloc[-1])

        values.append(result)

    columns = pandas.MultiIndex.from_tuples(
        [
            ("", "", r"$N_l$"),
            ("", "", r"$N_n$"),
            ("", "", r"$N_{bs}$"),
            ("t=0", "u", "orig"),
            ("t=0", "u", "swa"),
            ("t=0", "v", "orig"),
            ("t=40", "u", "orig"),
            ("t=40", "v", "orig"),
            ("t=80", "u", "orig"),
            ("t=80", "v", "orig"),
        ]
    )

    table = pandas.DataFrame(data=values, columns=columns)
    print(table)

    # adjust output formats
    table = table.style.hide()
    table = table.format("{:7.2e}", ["t=0", "t=40", "t=80"])
    table = table.format("{:.1f}", "Run time")
    table = table.format(subset=[("", r"DoF"), ("", r"$N_{bs}$")], thousands=",")

    # output
    table.to_latex(
        buf=fname,
        column_format="c"*len(columns),
        hrules=True,
        label="table:tgv-overview",
        caption=r"Overview, run times, and $l_2$-norms (from original models) of 2D Taylor-Green vortex benchmarks",
        sparse_index=False,
        multirow_align="c",
        multicol_align="c",
        environment="longtable"
    )


def main(rootdir, force: bool = False):
    """Main function.
    """

    # folders
    workdir = rootdir.joinpath("outputs")

    # save the table to figures' folder
    figdir = rootdir.joinpath("figures")
    figdir.mkdir(exist_ok=True)

    nls = [1, 2, 3]
    nns = [16, 32, 64, 128, 256]
    nbss = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    create_overview_table(workdir, figdir.joinpath("overview.tex"), nls, nns, nbss)

    return 0


if __name__ == "__main__":
    import sys
    import argparse

    # find the root of the folder `modulus`
    for root in pathlib.Path(__file__).resolve().parents:
        if root.joinpath("taylor-green-vortex-2d-re100").is_dir():
            break
    else:
        raise FileNotFoundError("Couldn't locate the path to the folder `modulus`.")

    root = root.joinpath("taylor-green-vortex-2d-re100")

    # cmd arguments
    parser = argparse.ArgumentParser(description="Post-processing Modulus TGV 2D Re100")
    parser.add_argument("--force", action="store_true", default=False, help="Force re-write.")
    args = parser.parse_args()

    # calling the main function
    sys.exit(main(root, args.force))
