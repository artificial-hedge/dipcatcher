"""Tests for quant_fund.monitoring.alert_budget."""

from __future__ import annotations

import pytest

from quant_fund.monitoring.alert_budget import AlertBudget, alert_budget_bench


def test_wealth_conservation_invariant():
    b = AlertBudget(0.2, ["a", "b", "c", "d"])
    w = b.alarm("a", 0.5)
    s = b.status()
    assert abs(s["wealth_plus_outstanding"] + s["destroyed_total"] - 0.2) < 1e-12
    b.resolve("a", confirmed=False)
    s = b.status()
    assert abs(s["wealth_plus_outstanding"] + s["destroyed_total"] - 0.2) < 1e-12
    assert s["destroyed_total"] == pytest.approx(w)


def test_confirmed_refund_restores_wealth():
    b = AlertBudget(0.4, ["x"])
    w = b.alarm("x", 0.5)
    before = b.rules["x"].wealth
    b.resolve("x", confirmed=True)
    assert b.rules["x"].wealth == pytest.approx(before + w)


def test_frozen_rule_cannot_alarm():
    b = AlertBudget(0.1, ["solo"])
    # burn the wallet: alarm + resolve false until empty
    while b.alarm("solo", 0.5) > 0:
        b.resolve("solo", confirmed=False)
        if b.rules["solo"].wealth <= 0:
            break
    assert b.alarm("solo", 0.5) == 0.0
    assert b.status()["per_rule"]["solo"]["frozen"] is True


def test_destroyed_never_exceeds_alpha():
    b = AlertBudget(0.3, ["a", "b"])
    for _ in range(20):
        for r in ("a", "b"):
            w = b.alarm(r, 0.9)
            if w:
                b.resolve(r, confirmed=False)
    assert b.status()["destroyed_total"] <= 0.3 + 1e-9


def test_resolve_without_pending_raises():
    b = AlertBudget(0.2, ["a"])
    with pytest.raises(ValueError, match="no pending"):
        b.resolve("a", confirmed=True)


def test_bad_init_rejected():
    with pytest.raises(ValueError):
        AlertBudget(0.0, ["a"])
    with pytest.raises(ValueError):
        AlertBudget(0.5, [])


def test_bench_sealed_and_deterministic():
    r1 = alert_budget_bench(horizon=200)
    r2 = alert_budget_bench(horizon=200)
    assert r1["schema"] == "alert_budget.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["live_pnl_claim"] is False
    assert r1["interpretation"]["conservation_bound_holds"]
    assert r1 == r2
