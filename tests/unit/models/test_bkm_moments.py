"""Unit tests for quant_fund.models.bkm_moments."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bkm_moments import (
    bench_bkm_moments,
    bkm_moments,
    mixture_option_prices,
    synth_bkm,
)


def _flat_surface(n: int = 200) -> dict[str, object]:
    """Flat-vol surface: Gaussian log returns, analytic BS prices."""
    spot, rate, tau, sig = 100.0, 0.02, 0.5, 0.25
    strikes = np.linspace(0.3 * spot, 3.0 * spot, n)
    out = mixture_option_prices(
        strikes,
        spot,
        rate,
        tau,
        np.array([1.0]),
        np.array([np.log(spot) + (rate - 0.5 * sig * sig) * tau]),
        np.array([sig * np.sqrt(tau)]),
    )
    calls = np.asarray(out["call_prices"])
    puts = np.asarray(out["put_prices"])
    is_call = (strikes >= spot).astype(np.float64)
    return {
        "strikes": strikes,
        "prices": np.where(is_call.astype(bool), calls, puts),
        "is_call": is_call,
        "spot": spot,
        "rate": rate,
        "tau": tau,
    }


def test_flat_surface_gaussian_moments() -> None:
    d = _flat_surface()
    est = bkm_moments(
        np.asarray(d["strikes"]),
        np.asarray(d["prices"]),
        np.asarray(d["is_call"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    assert est["var_q"] == pytest.approx(0.25**2, rel=0.1)
    assert abs(est["skew_q"]) < 0.15
    assert est["kurt_q"] == pytest.approx(3.0, abs=0.6)


def test_mixture_moments_track_truth() -> None:
    d = synth_bkm(seed=3)
    est = bkm_moments(
        np.asarray(d["strikes"]),
        np.asarray(d["prices"]),
        np.asarray(d["is_call"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    assert est["var_q"] * float(d["tau"]) == pytest.approx(float(d["var_true"]), rel=0.1)
    assert est["skew_q"] < 0.0
    assert est["skew_q"] == pytest.approx(float(d["skew_true"]), abs=0.12)
    assert est["kurt_q"] == pytest.approx(float(d["kurt_true"]), rel=0.2)


def test_deterministic() -> None:
    d = synth_bkm(seed=4)
    args = (
        np.asarray(d["strikes"]),
        np.asarray(d["prices"]),
        np.asarray(d["is_call"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    assert bkm_moments(*args) == bkm_moments(*args)


def test_forward_consistency() -> None:
    out = mixture_option_prices(
        np.linspace(60.0, 200.0, 20),
        100.0,
        0.05,
        1.0,
        np.array([0.6, 0.4]),
        np.array([4.5, 4.7]),
        np.array([0.2, 0.4]),
    )
    assert float(out["forward_implied"]) == pytest.approx(100.0 * np.exp(0.05), rel=1e-9)


def test_fail_closed_shapes() -> None:
    d = synth_bkm(seed=5)
    with pytest.raises(ValueError):
        bkm_moments(
            np.asarray(d["strikes"])[:3],
            np.asarray(d["prices"])[:3],
            np.asarray(d["is_call"])[:3],
            100.0,
            0.03,
            0.5,
        )
    with pytest.raises(ValueError):
        bkm_moments(
            np.asarray(d["strikes"]),
            np.asarray(d["prices"]),
            np.asarray(d["is_call"]),
            -1.0,
            0.03,
            0.5,
        )
    with pytest.raises(ValueError):
        bkm_moments(
            np.asarray(d["strikes"]),
            np.asarray(d["prices"]),
            np.asarray(d["is_call"]),
            100.0,
            0.03,
            0.0,
        )


def test_fail_closed_one_sided() -> None:
    d = synth_bkm(seed=6)
    with pytest.raises(ValueError):
        bkm_moments(
            np.asarray(d["strikes"]),
            np.asarray(d["prices"]),
            np.ones_like(np.asarray(d["is_call"])),
            100.0,
            0.03,
            0.5,
        )


def test_bench_score() -> None:
    out = bench_bkm_moments()
    assert out["synthetic_score"] == 1.0
    assert set(out) == {
        "synthetic_var_q",
        "synthetic_var_true",
        "synthetic_skew_q",
        "synthetic_skew_true",
        "synthetic_kurt_q",
        "synthetic_kurt_true",
        "synthetic_score",
    }
