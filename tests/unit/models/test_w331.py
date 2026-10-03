"""Wave-331 complexity-theory module unit tests."""

from __future__ import annotations

import random

from quant_fund.models.circuit_lb import circuit_size
from quant_fund.models.fpras_dnf import exact_dnf_count, kl_count
from quant_fund.models.np_reduce import brute_sat, sat_to_3sat, three_sat_to_vc
from quant_fund.models.param_fpt import kernel, vc_fpt
from quant_fund.models.pcp_verify import blr_test, decode, hadamard
from quant_fund.models.sumcheck import run_sumcheck, total_sum


def test_reduce() -> None:
    clauses = [(1, 2, 3, 4), (-1, 5)]
    c3, n3 = sat_to_3sat(clauses, 5)
    assert all(len(c) <= 3 for c in c3)
    assert brute_sat(clauses, 5) == brute_sat(c3, n3)
    edges, k, nn = three_sat_to_vc([(1, 2, 3)], 3)
    assert k == 3 + 2


def test_fpras() -> None:
    assert exact_dnf_count([(1,), (2, -3)], 3) == 5
    rng = random.Random(0)
    assert kl_count([], 3, 0.1, rng) == 0.0


def test_sumcheck() -> None:
    def g(xs):
        return xs[0] * xs[1] + xs[2]

    assert total_sum(g, 3) == 6
    rng = random.Random(0)
    assert run_sumcheck(g, 3, 6, rng)
    assert not run_sumcheck(g, 3, 5, rng)


def test_fpt() -> None:
    path = [(0, 1), (1, 2), (2, 3)]
    assert vc_fpt(path, 4, 2) is not None
    assert vc_fpt(path, 4, 1) is None
    rem, cov, k2 = kernel([(0, i) for i in range(1, 6)], 6, 3)
    assert cov == {0} and rem == []


def test_pcp() -> None:
    rng = random.Random(0)
    f = hadamard([1, 0, 1])
    assert blr_test(f, 3, 20, rng) == 0
    assert decode(f, 3, 40, rng) == [1, 0, 1]


def test_circuit() -> None:
    assert circuit_size(0b0110, 2) == 1
    assert circuit_size(0b1001, 2) == 2
