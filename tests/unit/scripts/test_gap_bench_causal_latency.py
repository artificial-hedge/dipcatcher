"""Tests for ``scripts/bench_causal_latency.py``.

The benchmark times the causal-forecast scoring path with seeded SYNTHETIC
inputs. It is a compute-cost measurement, never market evidence, so what
matters for a unit test is only that the workload is deterministic and that
the reported table is well-formed.

``tests/unit/fx1_serve/test_api_audit.py`` is unrelated to this file; this
suite runs in well under a second.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "bench_causal_latency.py"
_spec = importlib.util.spec_from_file_location("gap_bench_causal_latency", _SCRIPT_PATH)
assert _spec and _spec.loader, "bench_causal_latency.py must be importable"
mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_bench_causal_latency"] = mod
_spec.loader.exec_module(mod)


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #


def test_the_workload_is_deterministic_for_a_seed() -> None:
    first = mod._workload(64, seed=11)
    second = mod._workload(64, seed=11)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_a_different_seed_gives_a_different_workload() -> None:
    first = mod._workload(64, seed=11)
    second = mod._workload(64, seed=12)
    assert not np.array_equal(first[0], second[0])


def test_the_workload_has_the_requested_shape() -> None:
    y, mu, sigma, losses = mod._workload(128, seed=3)
    for array in (y, mu, sigma, losses):
        assert array.shape == (128,)
        assert np.all(np.isfinite(array))


def test_the_scale_forecast_is_strictly_positive() -> None:
    """A non-positive scale would make the Student-t CRPS undefined."""
    _y, _mu, sigma, _losses = mod._workload(256, seed=5)
    assert np.all(sigma > 0.0)


def test_the_residuals_are_the_absolute_target_error() -> None:
    y, mu, _sigma, losses = mod._workload(64, seed=7)
    assert np.allclose(losses, np.abs(y - mu))


# --------------------------------------------------------------------------- #
# Timing and formatting
# --------------------------------------------------------------------------- #


def test_the_timer_returns_one_sample_per_repetition() -> None:
    samples = mod._time(lambda: None, reps=5)
    assert len(samples) == 5
    assert all(sample >= 0.0 for sample in samples)


def test_the_timer_reports_milliseconds() -> None:
    samples = mod._time(lambda: sum(range(1000)), reps=3)
    assert all(isinstance(sample, float) for sample in samples)


def test_a_table_row_carries_the_name_and_the_sample_count() -> None:
    row = mod._row("`fit_student_t`", 500, [1.0, 2.0, 3.0])
    assert row.startswith("| `fit_student_t` | 500 |")
    assert row.endswith("|")
    # mean 2.0, median 2.0, max 3.0.
    assert "| 2.00 | 2.00 | 3.00 |" in row


def test_a_row_formats_a_single_sample_without_dividing_by_zero() -> None:
    row = mod._row("x", 10, [4.0])
    assert "| 4.00 | 4.00 | 4.00 |" in row


def test_the_composed_path_runs_end_to_end() -> None:
    """The composed interval is the decision path the benchmark exists for."""
    from quant_fund.metrics.conformal import conformal_quantile
    from quant_fund.metrics.risk_parametric import fit_student_t
    from quant_fund.metrics.scoring import crps_student_t

    _y, _mu, _sigma, losses = mod._workload(256, seed=13)
    fit = fit_student_t(losses)
    quantile = conformal_quantile(losses, 0.1)
    score = crps_student_t(_y, _mu, _sigma, nu=fit.get("nu", 5.0))
    assert np.isfinite(quantile)
    assert np.all(np.isfinite(score))


@pytest.mark.parametrize("n", [16, 64])
def test_the_workload_scales_with_n(n: int) -> None:
    y, mu, sigma, losses = mod._workload(n, seed=1)
    assert y.size == mu.size == sigma.size == losses.size == n
