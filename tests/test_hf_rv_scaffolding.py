"""HF-RV pinning tests (Day Wave 140 design drop).

These tests pin the public surface of ``quant_fund.realized.hf_rv`` and the
minimum-behavior contracts documented in ``docs/HF_RV_DESIGN.md``:

- :mod:`quant_fund.realized.hf_rv` is importable.
- :class:`HFRVResult` is a dataclass with the planned fields.
- :func:`compute_hf_rv` has signature
  ``(bars, asof, available_time, *, window_bars=288, apply_tick_subsample=True, subsample_stride=5, rng_seed=None)``.
- With too-small input (1-bar, empty) the function returns
  ``HFRVResult(honest=False)`` so downstream consumers fail closed
  (matching the design doc's "fail-closed" contract).

The original scaffolding pin (raising :class:`NotImplementedError`) was
retired when the implementation landed; this file now exercises the
public surface plus the two fail-mode-shaped boundaries the scaffolding
tests originally expressed.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pandas as pd

from quant_fund.realized.hf_rv import HFRVResult, compute_hf_rv

EXPECTED_PARAM_NAMES = ["bars", "asof", "available_time"]


def test_hfrvresult_is_dataclass_with_planned_fields() -> None:
    fields = {f.name for f in HFRVResult.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    expected = {
        "rv_5min",
        "bpv",
        "jump_stat",
        "n_jumps",
        "rv_5min_jump_clean",
        "n_obs",
        "asof",
        "available_time",
        "source",
        "source_secondary",
        "clock_drift_seconds",
        "holes_count",
        "honest",
    }
    assert fields == expected, f"HFRVResult fields drifted: {fields ^ expected}"


def test_compute_hf_rv_signature() -> None:
    sig = inspect.signature(compute_hf_rv)
    params = list(sig.parameters)
    assert params == [
        "bars",
        "asof",
        "available_time",
        "window_bars",
        "apply_tick_subsample",
        "subsample_stride",
        "rng_seed",
    ], f"compute_hf_rv signature must match the design doc; got {params}"
    # Keyword-only after the first three
    for name in ("window_bars", "apply_tick_subsample", "subsample_stride", "rng_seed"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only; got {sig.parameters[name].kind}"
        )


def test_compute_hf_rv_defaults() -> None:
    sig = inspect.signature(compute_hf_rv)
    assert sig.parameters["window_bars"].default == 288
    assert sig.parameters["apply_tick_subsample"].default is True
    assert sig.parameters["subsample_stride"].default == 5
    assert sig.parameters["rng_seed"].default is None


def test_compute_hf_rv_runs_and_marks_fail_closed_on_trivial_bars() -> None:
    """A 1-bar frame has zero log returns; the function must fail closed.

    The original scaffolding pin asserted that :func:`compute_hf_rv`
    raised :class:`NotImplementedError` on a 1-bar frame. With the
    implementation landed, the same input shape produces a well-formed
    :class:`HFRVResult` with ``honest=False`` so downstream consumers
    detect fail-closed uniformly.
    """
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    available_time = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    bars = pd.DataFrame(
        {
            "event_time": [asof - pd.Timedelta(minutes=5)],
            "available_time": [available_time],
            "open": [100.0],
            "high": [100.5],
            "low": [99.5],
            "close": [100.2],
        }
    )
    result = compute_hf_rv(bars, asof, available_time)
    assert result.honest is False, (
        "A 1-bar frame has zero log returns; the function must fail closed."
    )
    assert result.n_obs == 0
    assert result.asof == asof


def test_compute_hf_rv_fail_closed_on_empty_bars() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    available_time = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    result = compute_hf_rv(pd.DataFrame(), asof, available_time)
    assert result.honest is False
    assert result.n_obs == 0
    assert result.asof == asof


def test_module_exports_match_design_doc() -> None:
    import quant_fund.realized as realized_mod
    import quant_fund.realized.hf_rv as hf_rv_mod

    assert realized_mod.HFRVResult is HFRVResult
    assert realized_mod.compute_hf_rv is compute_hf_rv
    assert hf_rv_mod.__all__ == ["HFRVResult", "compute_hf_rv"]
