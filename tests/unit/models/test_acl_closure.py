"""Adversarial probes for acl_closure (SYNTHETIC)."""

from __future__ import annotations

import pytest

from quant_fund.models.acl_closure import acl, bench_acl_closure


def test_acl_superset_of_a():
    k4 = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    assert {1, 2} <= acl(4, k4, {1, 2})


def test_acl_complete_graph_symmetric():
    k4 = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    assert acl(4, k4, set()) == set()
    assert acl(4, k4, {0}) == {0}


def test_acl_star_center_vs_leaf():
    star = {(0, i) for i in range(1, 5)}
    assert acl(5, star, {0}) == {0}  # leaves permutable
    assert acl(5, star, {1}) == {0, 1}  # fixed leaf pins center


def test_acl_idempotent():
    star = {(0, i) for i in range(1, 5)}
    a1 = acl(5, star, {1})
    a2 = acl(5, star, a1)
    assert a1 == a2


def test_acl_hostile():
    star = {(0, i) for i in range(1, 4)}
    with pytest.raises(ValueError):
        acl(0, set(), set())  # empty universe
    with pytest.raises(ValueError):
        acl(4, star, {7})  # fixed node out of range
    with pytest.raises(ValueError):
        acl(4, {(0, 9)}, set())  # edge references node 9 >= n
    with pytest.raises(ValueError):
        acl(4, {(-1, 2)}, set())  # negative node


def test_bench_acl_full_pass():
    assert bench_acl_closure()["synthetic_acl_closure"] == 1.0
