#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Plot bar plots for google scholar search result counts.
"""
import pathlib
from matplotlib import pyplot
from matplotlib import ticker

rootdir = pathlib.Path(__file__).resolve().parents[1]
figdir = rootdir.joinpath("figures")
pyplot.style.use(rootdir.joinpath("resources", "figstyle"))

labels = range(2017, 2023)
values = [21, 45, 169, 692, 1820, 2250]

fig = pyplot.figure(figsize=(3.2, 3.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.2])

ax = fig.add_subplot(gs[0])
ax.bar(labels, values, color="teal")
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([])
ax.xaxis.get_major_formatter().set_scientific(False)
ax.set_xticks(labels, fontsize=8.5)
ax.set_xlabel("Year")

for label, val in zip(labels, values):
    ax.annotate(str(val), (label, val+20), ha="center", fontsize=8.5)

ax = fig.add_subplot(gs[1])
ax.annotate(
    "1. The results were obtained by searching the keyword \"physics\n"
    "    informed neural networks\" on Google Scholar",
    (0.1, 1), xycoords="axes fraction", fontsize=7.5, ha="left", va="top",
)
ax.annotate(
    "2. The result of year 2022 is not complete as the figure was\n"
    "    generated in September, 2022",
    (0.1, 0.3), xycoords="axes fraction", fontsize=7.5, ha="left", va="top",
)
ax.set_xticks([])
ax.set_yticks([])
ax.axis(False)

fig.suptitle(r"Number of PINNs articles per year (2017-2022)$^{1,2}$", fontsize=10)
pyplot.savefig(figdir.joinpath("pinn-search-result-counts.png"))
