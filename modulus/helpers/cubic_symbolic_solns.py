#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""Solve 4x4 system symbolically.
"""
import sympy
sympy.init_printing(use_unicode=True)

a, b = sympy.symbols("a, b")
phia, phib = sympy.symbols("\u03d5(a), \u03d5(b)")
dphia, dphib = sympy.symbols("\u03d5'(a), \u03d5'(b)")

A = sympy.Matrix([
    [a**3, a**2, a, 1, phia],
    [b**3, b**2, b, 1, phib],
    [3*a**2, 2*a, 1, 0, dphia],
    [3*b**2, 2*b, 1, 0, dphib],
])


c1, c2, c3, c4 = sympy.symbols("c1, c2, c3, c4")
soln = sympy.solve_linear_system_LU(A, [c1, c2, c3, c4])

print("\nc1:")
nomenator, denomenator = sympy.fraction(soln[c1].simplify())
check = (a - b) * (dphia + dphib) - 2 * (phia - phib)
print(f"\tnomenator: {sympy.factor(nomenator, deep=True)}")
print(f"\tsimplified: {check}")
print(f"\tEquivalent: {nomenator.equals(check)}")
print(f"\tdenomenator: {denomenator}")

print("\nc2:")
nomenator, denomenator = sympy.fraction(soln[c2].simplify())
check = - 2 * (dphia + dphib) * (a + b) * (a - b) + (a - b) * (a * dphia + b * dphib) + 3 * (a + b) * (phia - phib)
print(f"\tnomenator: {nomenator}")
print(f"\tsimplified: {check}")
print(f"\tEquivalent: {nomenator.equals(check)}")
print(f"\tdenomenator: {denomenator}")

print("\nc3:")
nomenator, denomenator = sympy.fraction(soln[c3].simplify())
check = 2 * (a + b) * (a - b) * (b * dphia + a * dphib) - 6 * a * b * (phia - phib) - (a - b) * (a**2 * dphib + b**2 * dphia)
print(f"\tnomenator: {nomenator}")
print(f"\tsimplified: {check}")
print(f"\tEquivalent: {nomenator.equals(check)}")
print(f"\tdenomenator: {denomenator}")

# we don't need c4
print("\nc4:")
nomenator, denomenator = sympy.fraction(soln[c4].simplify())
print(f"\tnomenator: {nomenator}")
print(f"\tdenomenator: {denomenator}")

x = sympy.symbols("x")
dq = 3 * c1 * x**2 + 2 * c2 * x + c3
soln = sympy.solve(sympy.Eq(dq, 0), x)
print()
print(len(soln))
print()
print(sympy.simplify(soln[0]))
print()
print(sympy.simplify(soln[1]))
