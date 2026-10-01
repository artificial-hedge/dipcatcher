"""Tests for config/impact.py — evidence invalidation engine."""

from __future__ import annotations

import json

from quant_fund.config.impact import (
    config_diff,
    flatten,
    invalidation_report,
    kind_domains,
)


def test_flatten_nested():
    out = flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3})
    assert out == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_diff_reports_domains():
    d = config_diff({"cost": {"half_spread_bps": 5.0}}, {"cost": {"half_spread_bps": 9.0}})
    assert len(d) == 1
    assert d[0]["path"] == "cost.half_spread_bps"
    assert d[0]["domains"] == ["execution_cost"]


def test_runtime_changes_touch_nothing():
    d = config_diff({"runtime": {"n_jobs": 4}}, {"runtime": {"n_jobs": 8}})
    assert d[0]["domains"] == []


def test_kind_classification():
    assert "market_data" in kind_domains("lobster_replay")
    assert "forecast" in kind_domains("coverage_watch")
    assert kind_domains("corpus_epoch") == frozenset()
    assert "risk" in kind_domains("decay_watch")
    # unknown kind with REAL data is conservatively market_data
    assert kind_domains("novel_lane", "MIXED") == frozenset({"market_data"})
    assert kind_domains("novel_lane") == frozenset()


def test_invalidation_report(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"kind": "trade_decomp", "data_label": "MIXED"}))
    (tmp_path / "b.json").write_text(json.dumps({"kind": "coverage_watch", "data_label": "REAL"}))
    (tmp_path / "c.json").write_text(json.dumps({"kind": "repo_integrity"}))
    d = config_diff(
        {"cost": {"half_spread_bps": 5.0}, "train": {"epochs": 3}},
        {"cost": {"half_spread_bps": 9.0}, "train": {"epochs": 3}},
    )
    rep = invalidation_report(d, tmp_path)
    assert rep["touched_domains"] == ["execution_cost"]
    names = {h["receipt"] for h in rep["invalidated"]}
    assert names == {
        "a.json"
    }  # coverage_watch doesn't consume execution_cost; repo_integrity immune


def test_data_change_invalidates_market_consumers(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"kind": "coverage_watch", "data_label": "REAL"}))
    (tmp_path / "b.json").write_text(json.dumps({"kind": "corpus_epoch"}))
    d = config_diff({"data": {"source": "a"}}, {"data": {"source": "b"}})
    rep = invalidation_report(d, tmp_path)
    assert {h["receipt"] for h in rep["invalidated"]} == {"a.json"}


def test_no_change_no_invalidation(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"kind": "trade_decomp"}))
    rep = invalidation_report([], tmp_path)
    assert rep["ok"] and rep["n_invalidated"] == 0
