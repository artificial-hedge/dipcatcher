"""Tests for quant_fund.pit.asof_audit."""

from __future__ import annotations

from quant_fund.pit.asof_audit import asof_audit_bench, asof_storm_audit


def test_storm_audit_clean_run():
    res = asof_storm_audit(seed=3, n_restatements=4, n_probes=15)
    assert res["probes"] > 0
    assert res["total_violations"] == 0


def test_storm_audit_deterministic():
    r1 = asof_storm_audit(seed=7, n_restatements=3, n_probes=10)
    r2 = asof_storm_audit(seed=7, n_restatements=3, n_probes=10)
    assert r1 == r2


def test_bench_sealed_verdict_ok():
    r = asof_audit_bench(seed=1)
    assert r["schema"] == "asof_audit.v1"
    assert r["data_label"] == "SYNTHETIC"
    assert r["live_pnl_claim"] is False
    assert r["claim"]["verdict"] == "ok"
    assert "receipt_sha256" in r


def test_bench_deterministic_seal():
    r1 = asof_audit_bench(seed=2)
    r2 = asof_audit_bench(seed=2)
    assert r1 == r2
