"""Tests for models/bank_conflict.py — bench must check the exact degree."""

from __future__ import annotations

import math

import numpy as np


def test_degree_is_gcd_exact() -> None:
    from quant_fund.models.bank_conflict import conflicts

    for stride in range(1, 9):
        assert conflicts(np.arange(32) * stride) == math.gcd(stride, 32)


def test_bench_discriminates_wrong_degree(monkeypatch) -> None:
    """The old check ``deg >= 1 and deg <= 32`` was vacuous — any counter
    passed. With an injected wrong degree the bench must report < 1."""
    import quant_fund.models.bank_conflict as bc

    monkeypatch.setattr(bc, "conflicts", lambda addrs, n_banks=32: 5)
    assert bc.bench_bank_conflict()["synthetic_bank_degree"] < 1.0
