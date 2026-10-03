"""Tests for seed_sweep: MCSE audit of sim-vs-real divergences."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.formal.seed_sweep import mcse_audit, sweep_arm, verdicts


def _noisy_builder(scale: float, shift: float = 0.0):
    def build(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        return {"metric": float(rng.normal(shift, scale))}

    return build


def _constant_builder(v: float):
    def build(seed: int) -> dict:
        return {"metric": v}

    return build


def test_sweep_mean_se() -> None:
    swept = sweep_arm(_noisy_builder(1.0, shift=5.0), {"m": lambda d: d["metric"]}, range(30))
    assert swept["m"]["n_valid"] == 30
    assert abs(swept["m"]["mean"] - 5.0) < 0.5
    assert abs(swept["m"]["se"] - 1.0 / np.sqrt(30)) < 0.02


def test_verdict_significant() -> None:
    swept = sweep_arm(_noisy_builder(0.01, shift=5.0), {"m": lambda d: d["metric"]}, range(8))
    v = verdicts(swept, {"m": 1.0}, 8)
    assert v["m"]["verdict"] == "significant_divergence"
    assert v["m"]["z"] is not None and v["m"]["z"] > 3.0


def test_verdict_within_noise() -> None:
    swept = sweep_arm(_noisy_builder(10.0, shift=1.0), {"m": lambda d: d["metric"]}, range(8))
    v = verdicts(swept, {"m": 1.5}, 8)
    assert v["m"]["verdict"] == "within_mc_noise"


def test_verdict_inexpressible() -> None:
    swept = sweep_arm(_constant_builder(1.0), {"m": lambda d: d["metric"]}, range(8))
    v = verdicts(swept, {"m": 0.5}, 8)
    assert v["m"]["verdict"] == "inexpressible"  # SE==0, gap != 0


def test_verdict_matched() -> None:
    swept = sweep_arm(_constant_builder(0.0), {"m": lambda d: d["metric"]}, range(8))
    v = verdicts(swept, {"m": 0.0}, 8)
    assert v["m"]["verdict"] == "matched"


def test_verdict_insufficient() -> None:
    def build(seed: int) -> dict:
        return {"metric": None}

    swept = sweep_arm(build, {"m": lambda d: d["metric"]}, range(8))
    v = verdicts(swept, {"m": 1.0}, 8)
    assert v["m"]["verdict"] == "insufficient_data"


def test_mcse_audit_rollup() -> None:
    out = mcse_audit(
        arms={
            "noisy_far": _noisy_builder(0.01, shift=5.0),
            "noisy_close": _noisy_builder(10.0, shift=1.0),
            "stuck": _constant_builder(1.0),
        },
        extractors={"m": lambda d: d["metric"]},
        real={"m": 0.5},
        seeds=range(8),
    )
    c = out["summary"]["verdict_counts"]
    assert c["significant_divergence"] == 1
    assert c["within_mc_noise"] == 1
    assert c["inexpressible"] == 1
    assert 0 < out["summary"]["significant_share"] <= 1.0


def test_too_few_seeds() -> None:
    with pytest.raises(ValueError):
        mcse_audit(
            {"a": _constant_builder(1.0)}, {"m": lambda d: d["metric"]}, {"m": 1.0}, range(3)
        )
