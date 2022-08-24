#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Tools to simulate how learning rates change w.r.t. iterations.
"""
import numpy
import ipywidgets
from matplotlib import pyplot


def cyclic_exp_range(step, gamma, maxlr, baselr, halfcycle):
    """CyclicLR exp_range calculator.
    """
    cycle = numpy.floor(1 + step / (2 * halfcycle))
    x = numpy.abs(step / halfcycle - 2 * cycle + 1)
    lr = baselr + (maxlr - baselr) * numpy.maximum(0, (1-x)) * gamma**(step)
    return lr


def exponential(steps, gamma, lr0):
    """ExponentialLR.
    """
    return lr0 * numpy.power(gamma, steps)


def tf_exponential(steps, decay_rate, decay_steps, lr0):
    """ExponentialLR.
    """
    return lr0 * numpy.power(decay_rate, steps/decay_steps)


def plotter(**kwargs):
    """Plot learning rates.
    """

    steps = numpy.arange(0, kwargs["steps"], 100)
    cyclic = cyclic_exp_range(
        steps, kwargs["cyclic_gamma"], kwargs["cyclic_maxlr"],
        kwargs["cyclic_minlr"], kwargs["cyclic_halfcycle"]
    )
    explr = exponential(steps, kwargs["exp_gamma"], kwargs["exp_maxlr"])
    tfexplr = tf_exponential(
        steps, kwargs["tfexp_decay_rate"], kwargs["tfexp_decay_step"],
        kwargs["tfexp_maxlr"]
    )
    pyplot.semilogy(steps, cyclic, label="CyclicLR")
    pyplot.semilogy(steps, explr, label="ExponentialLR")
    pyplot.semilogy(steps, tfexplr, label="TFExponentialLR")
    pyplot.xlabel("Iteration")
    pyplot.xlabel("Learning rate")
    pyplot.legend(loc=0)


def widget():
    """Learning rate simulator widget for Jupyter Notebook.
    """
    # cyclic exp-range
    cyclic = {
        "gamma": ipywidgets.FloatText(value=0.9999915, description="gamma:"),
        "maxlr": ipywidgets.FloatText(value=1e-2, description="max lr:"),
        "minlr": ipywidgets.FloatText(value=1e-6, description="min lr:"),
        "halfcycle": ipywidgets.IntText(value=5000, description="half cycle:"),
    }

    cyclic["panel"] = ipywidgets.VBox([
        ipywidgets.Label("Cyclic exp-range"),
        cyclic["gamma"],
        cyclic["maxlr"],
        cyclic["minlr"],
        cyclic["halfcycle"],
    ])

    # exponential
    explr = {
        "gamma": ipywidgets.FloatText(value=0.9999915, description="gamma"),
        "maxlr": ipywidgets.FloatText(value=1e-2, description="init lr"),
    }

    explr["panel"] = ipywidgets.VBox([
        ipywidgets.Label("Exponential"),
        explr["gamma"],
        explr["maxlr"],
    ])

    # exponential
    tfexplr = {
        "decay_rate": ipywidgets.FloatText(value=0.96, description="decay rate"),
        "decay_step": ipywidgets.IntText(value=5000, description="decay step"),
        "maxlr": ipywidgets.FloatText(value=1e-2, description="max lr"),
    }

    tfexplr["panel"] = ipywidgets.VBox([
        ipywidgets.Label("TF Exponential"),
        tfexplr["decay_rate"],
        tfexplr["decay_step"],
        tfexplr["maxlr"],
    ])

    # total steps
    steps = ipywidgets.IntText(value=1000000, description="Steps")

    # plot canvas
    canvas = ipywidgets.interactive_output(
        plotter,
        {
            "cyclic_gamma": cyclic["gamma"],
            "cyclic_maxlr": cyclic["maxlr"],
            "cyclic_minlr": cyclic["minlr"],
            "cyclic_halfcycle": cyclic["halfcycle"],
            "exp_gamma": explr["gamma"],
            "exp_maxlr": explr["maxlr"],
            "tfexp_decay_rate": tfexplr["decay_rate"],
            "tfexp_decay_step": tfexplr["decay_step"],
            "tfexp_maxlr": tfexplr["maxlr"],
            "steps": steps,
        }
    )

    out = ipywidgets.HBox([
        ipywidgets.VBox([canvas, steps]),
        ipywidgets.VBox([cyclic["panel"], explr["panel"], tfexplr["panel"]]),
    ])
    
    return out


if __name__ == "__main__":
    steps = numpy.arange(0, 400001, 200)
    pyplot.semilogy(steps, cyclic_exp_range(steps, 0.999972, 1e-3, 1e-6, 2000), label="CyclicLR")
    pyplot.semilogy(steps, exponential(steps, 0.99998718, 1e-3), label="ExponentialLR")
    pyplot.semilogy(steps, tf_exponential(steps, 0.95, 1000, 1e-3), label="TFExponentialLR")
    pyplot.xlabel("Iteration")
    pyplot.xlabel("Learning rate")
    pyplot.legend(loc=0)
    pyplot.show()