"""Unit tests for quant_fund.models.gil_pelaez."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gil_pelaez import (
    bench_gil_pelaez,
    gil_pelaez_cdf,
    synth_ou_inversion,
)


def test_ou_inversion_matches_normal() -> None:
    out = synth_ou_inversion(seed=1)
    assert float(out["max_cdf_err"]) < 5e-3
    assert float(out["max_pdf_err"]) < 5e-2


def test_cdf_monotone_and_bounded() -> None:
    out = synth_ou_inversion(seed=2)
    assert float(out["cdf_min"]) >= 0.0
    assert float(out["cdf_max"]) <= 1.0
    assert float(out["cdf_min"]) < 0.05
    assert float(out["cdf_max"]) > 0.95


def test_schema() -> None:
    x = np.linspace(-1.0, 1.0, 21)

    def cf(u: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * u**2)

    out = gil_pelaez_cdf(cf, x)
    assert set(out) == {
        "cdf_min",
        "cdf_max",
        "pdf_sum",
        "max_cdf_err_vs_normal",
        "_cdf",
        "_pdf",
        "_u",
    }


def test_determinism() -> None:
    a = synth_ou_inversion(seed=5)
    b = synth_ou_inversion(seed=5)
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        gil_pelaez_cdf(lambda u: u, np.ones(3))
    with pytest.raises(ValueError):
        gil_pelaez_cdf(lambda u: u, np.linspace(0, 1, 10)[::-1])
    with pytest.raises(ValueError):
        gil_pelaez_cdf(lambda u: np.full(u.shape[0] + 1, 1j), np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        gil_pelaez_cdf("not callable", np.linspace(0, 1, 10))


def test_bench_keys_and_pass() -> None:
    out = bench_gil_pelaez()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_max_cdf_err",
        "synthetic_max_pdf_err",
        "synthetic_cdf_min",
        "synthetic_cdf_max",
        "synthetic_true_sd",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
