#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Post processing data of TGV 2D Re100.
"""
import itertools
import pathlib
import re
import sys
import pandas
from h5py import File as h5open

# find helpers and locate workdir
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from helpers.utils import log_parser  # pylint: disable=import-error # noqa: E402


def create_weak_scaling_table(simdir, outdir, tabledir):
    """Create strong scaling table.
    """

    gpus = [1, 2, 4, 8]
    cases = [(2, 32, 8192), (3, 128, 8192)]
    indices = ["Time cost", "Efficiency", "Loss", r"$L_2$ err., $u$", r"$L_2$ err., $v$"]
    columns = pandas.MultiIndex.from_product([cases, gpus])
    columns = columns.set_names("GPUs", level=1)
    data = pandas.DataFrame(data=None, index=indices, columns=columns)

    for (nl, nn, nbs), ngpus in itertools.product(cases, gpus):
        cname = f"nl{nl}-nn{nn}-npts{nbs}-ngpus{ngpus}"

        with h5open(outdir.joinpath(f"{cname}-raw.h5"), "r") as h5file:
            data.loc["Time cost", ((nl, nn, nbs), ngpus)] = h5file["walltime/elapsedtimes"][-1]
            data.loc[r"$L_2$ err., $u$", ((nl, nn, nbs), ngpus)] = float(h5file["sterrs/u/l2norm"][...])
            data.loc[r"$L_2$ err., $v$", ((nl, nn, nbs), ngpus)] = float(h5file["sterrs/v/l2norm"][...])

        data.loc["Loss", ((nl, nn, nbs), ngpus)] = log_parser(simdir.joinpath(cname)).loc[400000, "loss"]

        data.loc["Efficiency", ((nl, nn, nbs), ngpus)] = \
            100 * data.loc["Time cost", ((nl, nn, nbs), 1)].values[0] / \
            data.loc["Time cost", ((nl, nn, nbs), ngpus)].values[0]

    out = data.style
    out = out.format(formatter="{:5.2f}", subset=pandas.IndexSlice["Time cost", :])
    out = out.format(formatter="{:2.0f}", subset=pandas.IndexSlice["Efficiency", :])
    out = out.format(formatter="{:.1e}", subset=pandas.IndexSlice["Loss", :])
    out = out.format(formatter="{:.1e}", subset=pandas.IndexSlice[r"$L_2$ err., $u$", :])
    out = out.format(formatter="{:.1e}", subset=pandas.IndexSlice[r"$L_2$ err., $v$", :])

    out = out.to_latex(
        column_format="lcccccccc",
        position="H",
        position_float="centering",
        hrules=True,
        label="table:weak-scaling-perf",
        caption=(
            "\n    Weak scaling performance for $(N_l, N_n, N_{bs})$ $=$ $(2, 32, 65536)$ and $(3, 128, 65536)$.%\n"
            "    Time costs denote the wall time required to finish 400k training iterations in hours.%\n"
            "    Efficiency here stands for weak scaling efficiency in $\\%$.%\n"
            "    The aggregated losses were those at the last iteration.%\n"
            "    The $L_2$ errors were the overall spatial-temporal errors at the last training iteration.%\n",
            "\n    Weak scaling performance for $(N_l, N_n, N_{bs})=(2, 32, 65536)$ and $(3, 128, 65536)$\n"
        ),
        multicol_align="c",
    )

    patn = r"\\multicolumn(?:\{.*?\}){3}"
    patn = rf"(^\s*?&\s*?{patn}\s*&\s*{patn}.*?$)"
    repl = r"\g<1>\n\\cmidrule(rl){2-5} \\cmidrule(rl){6-9}"
    out = re.sub(patn, repl, out, flags=re.MULTILINE)

    out = re.sub(r"^(\\centering)$", r"\g<1>\n\\singlespacing", out, flags=re.MULTILINE)
    out = re.sub(r"GPUs", "\\\\multicolumn{1}{r}{GPUs}", out)
    out = re.sub(r"(^Time cost.*?)$", r"\g<1>\n\\addlinespace", out, flags=re.MULTILINE)
    out = re.sub(r"(^Efficiency.*?)$", r"\g<1>\n\\addlinespace", out, flags=re.MULTILINE)
    out = re.sub(r"(^Loss.*?)$", r"\g<1>\n\\addlinespace", out, flags=re.MULTILINE)
    out = re.sub(r"(^\$L_2\$ err\., \$u\$.*?)$", r"\g<1>\n\\addlinespace", out, flags=re.MULTILINE)

    tabledir.joinpath("scaling-tests").mkdir(parents=True, exist_ok=True)
    with open(tabledir.joinpath("scaling-tests", "weak-scaling.tex"), "w") as fobj:
        fobj.write(out)

    return out


if __name__ == "__main__":
    _projdir = pathlib.Path(__file__).resolve().parents[1]
    _outdir = _projdir.joinpath("outputs", "exp-sum-scaling")
    _simdir = _projdir.joinpath("exp-sum-scaling")

    _tabledir = _projdir.joinpath("tables")
    _tabledir.mkdir(parents=True, exist_ok=True)

    create_weak_scaling_table(_simdir, _outdir, _tabledir)
