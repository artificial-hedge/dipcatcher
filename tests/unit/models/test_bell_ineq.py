import math

import numpy as np

from quant_fund.models.bell_ineq import bench_bell_ineq, chsh_expect, classical_chsh


def _bell_state() -> np.ndarray:
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    return np.outer(bell, bell.conj())


def test_tsirelson_bound():
    # documented convention: S = E(a,b)+E(a,b2)+E(a2,b)-E(a2,b2) hits 2*sqrt(2)
    s = chsh_expect(_bell_state(), 0.0, math.pi / 2, math.pi / 4, -math.pi / 4)
    assert abs(abs(s) - 2 * math.sqrt(2)) < 1e-9


def test_classical_bound():
    assert classical_chsh() == 2.0
    prod = np.kron(np.array([[1, 0], [0, 0]], complex), np.array([[1, 0], [0, 0]], complex))
    s = chsh_expect(prod, 0.0, math.pi / 2, math.pi / 4, -math.pi / 4)
    assert abs(s) <= 2.0 + 1e-9


def test_bench_contract():
    assert bench_bell_ineq() == {"synthetic_bell_ineq": 1.0}
