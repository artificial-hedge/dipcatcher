"""Boundary audit of the native kernels — verdicts must stay documented."""

from __future__ import annotations

import numpy as np

from quant_fund.native.boundary_audit import _classify, boundary_audit, boundary_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_flat_input_zero_std_is_value_not_flag() -> None:
    cell = _classify("rolling_std", "flat", np.full(24, 1.0), lambda v: np.zeros(24))
    assert cell["verdict"] == "value"  # flat input legitimately yields 0 std


def test_real_signal_silenced_is_flagged() -> None:
    cell = _classify("fake", "mixed", np.linspace(-1, 1, 24), lambda v: np.zeros(24))
    assert cell["verdict"] == "flag:silent_zeros"


def test_moderate_inf_escape_flagged_extreme_noted() -> None:
    moderate = _classify("fake", "m", np.linspace(1, 2, 10), lambda v: np.full(10, np.inf))
    assert moderate["verdict"] == "flag:inf_escape"
    extreme = _classify("fake", "x", np.full(10, 1e308), lambda v: np.full(10, np.inf))
    assert extreme["verdict"] == "note:overflow"


def test_audit_clean_after_bollinger_fix() -> None:
    audit = boundary_audit()
    assert audit["flagged"] == [], audit["flagged"][:5]


def test_bench_seals_and_verifies() -> None:
    receipt = boundary_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "boundary_audit_test.json")
    assert result["valid"], result.get("errors")
