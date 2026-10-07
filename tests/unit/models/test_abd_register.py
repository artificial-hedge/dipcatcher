"""Adversarial probes for abd_register (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.abd_register import ABDRegister, bench_abd_register


def test_register_hostile_n():
    with pytest.raises(ValueError):
        ABDRegister(0)
    with pytest.raises(ValueError):
        ABDRegister(-3)


def test_write_then_read_latest():
    rng = np.random.default_rng(0)
    reg = ABDRegister(5)
    t = reg.write(42, rng)
    tr, v = reg.read(rng)
    assert v == 42 and tr == t


def test_reads_monotone_ts():
    rng = np.random.default_rng(1)
    reg = ABDRegister(7)
    reg.write(1, rng)
    reg.write(2, rng)
    reg.write(3, rng)
    t_seen = -1
    for _ in range(8):
        t, v = reg.read(rng)
        assert t >= t_seen
        assert v == 3
        t_seen = t


def test_majority_writeback_propagates():
    rng = np.random.default_rng(2)
    reg = ABDRegister(5)
    reg.write(9, rng)
    # hammer reads: eventually EVERY replica holds latest (write-back spreads)
    for _ in range(40):
        reg.read(rng)
    assert all(r.val == 9 for r in reg.reps)


def test_deterministic_replay():
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    reg1, reg2 = ABDRegister(5), ABDRegister(5)
    reg1.write(7, rng1)
    reg2.write(7, rng2)
    pick1 = [(r.ts, r.val) for r in reg1.reps]
    pick2 = [(r.ts, r.val) for r in reg2.reps]
    assert pick1 == pick2


def test_bench_abd_full_pass():
    assert bench_abd_register()["synthetic_abd_register"] == 1.0
