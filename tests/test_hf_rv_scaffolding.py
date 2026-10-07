"""HF-RV scaffolding pinning tests (Day Wave 140 design drop).

These tests are a contract for the follow-up implementation wave. They pin:

- :mod:`quant_fund.realized.hf_rv` is importable.
- :class:`HFRVResult` is a dataclass with the planned fields.
- :func:`compute_hf_rv` has signature
  ``(bars, asof, available_time, *, window_bars=288, apply_tick_subsample=True, subsample_stride=5, rng_seed=None)``.
- :func:`compute_hf_rv` raises :class:`NotImplementedError` with the
  pinned message on every call (scaffolding is intentionally not implemented).
- The function's exception message references the design doc path so a
  future reader can find the contract.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pandas as pd
import pytest

from quant_fund.realized.hf_rv import HFRVResult, compute_hf_rv

EXPECTED_ERROR_SUBSTRING = "hf_rv_committed: see docs/HF_RV_DESIGN.md"
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


def test_compute_hf_rv_raises_not_implemented_with_synthetic_input() -> None:
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
    with pytest.raises(NotImplementedError) as excinfo:
        compute_hf_rv(bars, asof, available_time)
    assert EXPECTED_ERROR_SUBSTRING in str(excinfo.value)


def test_compute_hf_rv_raises_on_empty_bars() -> None:
    asof = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    available_time = pd.Timestamp(datetime(2024, 1, 2, 20, 0, tzinfo=UTC))
    with pytest.raises(NotImplementedError) as excinfo:
        compute_hf_rv(pd.DataFrame(), asof, available_time)
    assert EXPECTED_ERROR_SUBSTRING in str(excinfo.value)


def test_module_exports_match_design_doc() -> None:
    import quant_fund.realized as realized_mod
    import quant_fund.realized.hf_rv as hf_rv_mod

    assert realized_mod.HFRVResult is HFRVResult
    assert realized_mod.compute_hf_rv is compute_hf_rv
    assert hf_rv_mod.__all__ == ["HFRVResult", "compute_hf_rv"]
