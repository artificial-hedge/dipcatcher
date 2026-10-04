"""Wave-324 polyhedral module unit tests."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from quant_fund.models.banerjee_dep import dependence
from quant_fund.models.fourier_motzkin import feasible
from quant_fund.models.omega_test import _mk, satisfiable
from quant_fund.models.pluto_schedule import permutable, schedule
from quant_fund.models.tiling_legality import skew, tile_map
from quant_fund.models.vec_legality import best_loop, vectorizable


def test_fm() -> None:
    F = Fraction
    assert feasible([([-F(1)], F(0)), ([F(1)], F(3))])
    assert not feasible([([F(1)], F(1)), ([-F(1)], -F(2))])


def test_banerjee() -> None:
    assert dependence(([1], 0), ([1], 1), [(0, 10)])
    assert not dependence(([2], 0), ([2], 1), [(0, 20)])


def test_pluto() -> None:
    s = schedule([(1, -1)], 2, max_coeff=3)
    assert s is not None and s[0] >= s[1]
    assert permutable([(1, 0)])


def test_tiling() -> None:
    m = np.array([[1, 0], [1, 1]])
    assert skew([(1, -1)], m) == [(1, 0)]
    assert tile_map((7, 5), (4, 4)) == (1, 1)


def test_omega() -> None:
    assert satisfiable([_mk([-1], -3), _mk([1], 5)])
    assert not satisfiable([_mk([2], 3), _mk([-2], -3)])


def test_vec() -> None:
    assert vectorizable([(1, 0)])
    assert not vectorizable([(1, -1)])
    assert best_loop([(1, -1)], 2) == 0
