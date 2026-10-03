"""Tests for quant_fund.native.causality_scan."""

from __future__ import annotations

import numpy as np

from quant_fund.native.causality_scan import causality_bench, causality_scan


def test_all_reference_ops_causal():
    report = causality_scan(n=128, extend=32, seed=0)
    for name, r in report.items():
        assert r.get("status") == "causal", f"{name} leaked: {r}"


def test_leak_detection_works():
    # plant a non-causal op: centered moving average
    import quant_fund.native.causality_scan as mod

    def centered(a: np.ndarray) -> np.ndarray:
        k = 5
        pad = k // 2
        out = np.convolve(np.pad(a, pad, mode="edge"), np.ones(k) / k, mode="same")
        return out[pad:] if pad else out

    orig = mod._ops
    try:
        mod._ops = lambda: {**orig(), "centered_plant": centered}
        report = causality_scan(n=64, extend=16, seed=1)
    finally:
        mod._ops = orig
    assert report["centered_plant"]["status"] == "LEAK"
    assert report["centered_plant"]["n_diff"] > 0


def test_bench_sealed_and_deterministic():
    r1 = causality_bench(seed=0)
    r2 = causality_bench(seed=0)
    assert r1["schema"] == "causality_scan.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["claim"]["n_leaks"] == 0
    assert r1 == r2
