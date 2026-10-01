from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.sign_autocorr_real import (
    _fit_exponent,
    lobster_signs,
    sign_autocorr_bench,
    sim_signs,
)


def test_fit_recovers_planted_exponent() -> None:
    # rho(k) = k^-0.6 planted
    lags = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    curve = {f"lag{k}": k**-0.6 for k in lags}
    out = _fit_exponent(curve)
    assert out["exponent"] == pytest.approx(0.6, abs=0.01)


def test_fit_dead_curve_none() -> None:
    curve = {f"lag{k}": float("nan") for k in (1, 2, 4)}
    out = _fit_exponent(curve)
    assert out["exponent"] is None


def test_fit_negative_tail_skipped() -> None:
    # autocorr that goes negative late — only positive pts fit
    curve = {
        "lag1": 0.5,
        "lag2": 0.3,
        "lag4": 0.2,
        "lag8": 0.15,
        "lag16": 0.1,
        "lag32": 0.06,
        "lag64": -0.01,
        "lag128": -0.02,
    }
    out = _fit_exponent(curve)
    assert out["n_positive_lags"] == 6
    assert out["exponent"] is not None and out["exponent"] > 0


def test_lobster_signs_from_csv(tmp_path: Path) -> None:
    p = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        for i in range(300):
            # alternating execs: buy execs are direction -1, sell +1
            w.writerow([34200.0 + i, 4, i + 1, 10, 5000, -1 if i % 2 == 0 else 1])
    out = lobster_signs(tmp_path)
    assert out["n_execs"] == 300
    assert out["curve"]["lag1"] < 0  # perfect alternation


def test_sim_signs_split_persistent() -> None:
    from quant_fund.microstructure.split_flow import SplitFlow

    out = sim_signs(
        flow=SplitFlow(p_start=0.3, size_tail=1.2, k_min=5, k_max=50, intensity_mult=3.0, seed=0),
        horizon=8000,
        seed=3,
    )
    assert out["curve"]["lag1"] > 0.3


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sign_autocorr_bench(tmp_path)
