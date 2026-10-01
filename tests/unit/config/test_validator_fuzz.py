"""Tests for quant_fund.config.validator_fuzz."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_fund.config.models import NorthsetConfig, PromotionConfig
from quant_fund.config.validator_fuzz import validator_fuzz, validator_fuzz_bench


def test_fuzz_clean_after_fixes():
    r = validator_fuzz()
    assert r["verdict"] == "ok", r["violations"]
    assert r["n_violations"] == 0
    assert r["n_numeric_fields"] > 150


def test_finite_floors_reject_nan():
    for f in (
        "queue_priority_finite_floor",
        "side_notional_finite_floor",
        "tob_size_share_finite_floor",
    ):
        with pytest.raises(ValidationError):
            NorthsetConfig(**{f: float("nan")})
        with pytest.raises(ValidationError):
            NorthsetConfig(**{f: float("inf")})
        NorthsetConfig(**{f: 0.5})  # valid value still accepted


def test_sweep_fraction_rejects_nan():
    with pytest.raises(ValidationError):
        NorthsetConfig(sweep_min_fold_positive_fraction=float("nan"))


def test_promotion_spread_rejects_negative():
    with pytest.raises(ValidationError):
        PromotionConfig(min_cost_adjusted_spread=-0.5)


def test_bench_sealed():
    r = validator_fuzz_bench()
    assert r["schema"] == "validator_fuzz.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r == validator_fuzz_bench()
