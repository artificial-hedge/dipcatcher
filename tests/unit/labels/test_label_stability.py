"""Tests for labels.label_stability."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quant_fund.labels.label_stability import flip_rate, label_stability_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_flat_path_never_flips() -> None:
    close = np.full(200, 100.0)
    sigma = np.full(200, 0.01)
    events = np.arange(0, 160, 10, dtype=np.intp)
    out = flip_rate(close, sigma, events, 1.5, 1.0, 20, eps=0.05)
    assert out["flip_hi"] == 0.0 and out["flip_lo"] == 0.0


def test_no_observable_events() -> None:
    close = np.linspace(100, 101, 50)
    sigma = np.full(50, 0.01)
    events = np.array([49], dtype=np.intp)  # final bar: label unobservable
    out = flip_rate(close, sigma, events, 1.0, 1.0, 10, eps=0.05)
    assert out["n_observable"] == 0
    assert np.isnan(out["flip_hi"])


def test_bench_seals_and_verifies() -> None:
    receipt = label_stability_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["claim"]["knife_edge"]["flip_lo"] > 0.5
    verdict = verify_receipt_payload(receipt, Path("receipts") / "label_stability.json")
    assert verdict["valid"], verdict["errors"]
