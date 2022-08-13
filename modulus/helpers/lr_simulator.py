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


if __name__ == "__main__":
    steps = numpy.arange(0, 400001, 200)
    pyplot.semilogy(steps, cyclic_exp_range(steps, 0.999972, 1e-3, 1e-6, 2000), label="CyclicLR")
    pyplot.semilogy(steps, exponential(steps, 0.99998718, 1e-3), label="ExponentialLR")
    pyplot.semilogy(steps, tf_exponential(steps, 0.95, 1000, 1e-3), label="TFExponentialLR")
    pyplot.xlabel("Iteration")
    pyplot.xlabel("Learning rate")
    pyplot.legend(loc=0)
    pyplot.show()
