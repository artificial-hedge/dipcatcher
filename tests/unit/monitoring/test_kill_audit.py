"""Tests for quant_fund.monitoring.kill_audit."""

from __future__ import annotations

from quant_fund.monitoring.kill_audit import kill_audit_bench, kill_switch_audit


def test_full_matrix_clean():
    r = kill_switch_audit()
    assert r["n_checks"] == 57
    assert r["n_violations"] == 0
    assert r["verdict"] == "ok"


def test_spec_catches_regression(monkeypatch):
    import quant_fund.monitoring.kill_audit as mod

    # break one spec cell: pretend ENABLED allows flatten
    broken = dict(mod._SPEC)
    broken["ENABLED"] = (True, False, True)
    monkeypatch.setattr(mod, "_SPEC", broken)
    r = kill_switch_audit()
    assert r["n_violations"] > 0
    assert r["verdict"] == "violations"


def test_bench_sealed():
    r = kill_audit_bench()
    assert r["schema"] == "kill_audit.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r == kill_audit_bench()
