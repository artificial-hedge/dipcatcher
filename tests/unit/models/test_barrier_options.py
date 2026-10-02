"""Tests for barrier_options — Reiner-Rubinstein."""

from __future__ import annotations

import pytest

from quant_fund.models.barrier_options import (
    barrier_price,
    bench_barrier_options,
    down_call,
    up_put,
    vanilla_call,
)

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.25


def test_in_plus_out_equals_vanilla():
    for h in (70.0, 80.0, 90.0, 95.0):
        di = down_call(S, K, h, T, R, SIG, knock="in")
        do = down_call(S, K, h, T, R, SIG, knock="out")
        van = vanilla_call(S, K, T, R, SIG)
        assert di + do == pytest.approx(van, rel=1e-6)


def test_ko_decreases_as_barrier_rises():
    lo = down_call(S, K, 70.0, T, R, SIG, knock="out")
    hi = down_call(S, K, 95.0, T, R, SIG, knock="out")
    assert lo > hi


def test_ko_below_vanilla_positive():
    van = vanilla_call(S, K, T, R, SIG)
    ko = down_call(S, K, 90.0, T, R, SIG, knock="out")
    assert 0.0 < ko < van


def test_barrier_zero_gives_vanilla():
    # h -> 0 : knockout ~ vanilla
    ko = down_call(S, K, 1e-6, T, R, SIG, knock="out")
    van = vanilla_call(S, K, T, R, SIG)
    assert ko == pytest.approx(van, rel=1e-3)


def test_put_parity():
    h = 115.0
    ui = up_put(S, K, h, T, R, SIG, knock="in")
    uo = up_put(S, K, h, T, R, SIG, knock="out")
    assert ui + uo > 0
    assert uo >= 0.0


def test_dispatch():
    p = barrier_price(S, K, 80.0, T, R, SIG, kind="doc")
    assert p == pytest.approx(down_call(S, K, 80.0, T, R, SIG, knock="out"))
    with pytest.raises(ValueError):
        barrier_price(S, K, 80.0, T, R, SIG, kind="zzz")


def test_fail_closed_barrier_above_spot():
    with pytest.raises(ValueError):
        down_call(S, K, 110.0, T, R, SIG)


def test_bench():
    out = bench_barrier_options()
    assert out["score"] == 1.0
