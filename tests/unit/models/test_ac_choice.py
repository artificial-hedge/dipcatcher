"""Adversarial probes for ac_choice (SYNTHETIC)."""

from __future__ import annotations

import pytest

from quant_fund.models.ac_choice import (
    bench_ac_choice,
    chains_with_ub,
    choice,
    is_chain,
    upper_bounds,
    zorn_maximal,
)


def test_choice_picks_one_per_member():
    fam = [frozenset({1, 2}), frozenset({"a"}), frozenset({3, 4, 5})]
    c = choice(fam)
    assert len(c) == 3
    for s in fam:
        assert len(c & s) >= 1


def test_choice_empty_member_raises():
    with pytest.raises(ValueError):
        choice([frozenset({1}), frozenset()])


def test_is_chain_total_and_partial():
    # rel[y] = elements <= y (downward closure); here 1,2 <= 0 and 2 <= 1
    rel = {0: {0, 1, 2}, 1: {1, 2}, 2: {2}}
    assert is_chain(rel, {0, 1, 2})
    assert is_chain(rel, {0, 2})
    bad = {0: {0}, 1: {1}, 2: {2}}  # antichain: no comparability
    assert not is_chain(bad, {0, 1})


def test_upper_bounds():
    rel = {0: {0, 1, 2}, 1: {1, 2}, 2: {2}}  # 0 is the top (1,2 <= 0)
    ub = upper_bounds(rel, {0, 1}, {0, 1, 2})
    assert ub == {0}  # only the top bounds {0,1}
    assert upper_bounds(rel, set(), {0, 1, 2}) == {0, 1, 2}  # empty chain: all UB
    assert upper_bounds(rel, {2}, {0, 1, 2}) == {0, 1, 2}  # everything bounds the bottom


def test_zorn_maximal_canonical_pick():
    rel = {0: {0, 1, 2}, 1: {1, 2}, 2: {2}}  # 0 is the unique top
    assert zorn_maximal(rel, {0, 1, 2}) == 0
    # two incomparable maxima {1},{2} (0 below both) — canonical pick = smallest
    rel2 = {0: {0}, 1: {0, 1}, 2: {0, 2}}
    assert zorn_maximal(rel2, {0, 1, 2}) == 1


def test_zorn_maximal_deterministic():
    rel = {b: {a for a in range(4) if (a | b) == b} for b in range(4)}
    for _ in range(10):
        assert zorn_maximal(rel, {0, 1, 2, 3}) == 3


def test_chains_with_ub_poset_and_antichain():
    rel = {b: {a for a in range(4) if (a | b) == b} for b in range(4)}
    assert chains_with_ub(rel, {0, 1, 2, 3})
    # antichain on {0,1}: chain {0,1} isn't a chain (skipped); singles have UBs
    anti = {0: {0}, 1: {1}}
    assert chains_with_ub(anti, {0, 1})


def test_bench_ac_choice_full_pass():
    assert bench_ac_choice()["synthetic_ac_choice"] == 1.0
