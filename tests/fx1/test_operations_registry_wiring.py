"""SYNTHETIC registry-wiring tests: newly registered operations are reachable.

These cases invoke the explicit registry (the public entry point) rather than
the module functions directly, so they fail if an implementation is authored
but never registered. Results are correctness fixtures, not market evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.operations.registry import execute_operation, get_operation, list_operations

WIRED_IDS = ("features.rolling_linear_trend", "skills.resolve_security_identity")


def test_wired_operations_are_reachable_through_registry() -> None:
    for operation_id in WIRED_IDS:
        operation = get_operation(operation_id)
        module_name = operation_id.split(".", 1)[1]
        assert operation.id == operation_id
        assert operation.handler.__module__ == f"fx1.operations.{module_name}"


def test_list_operations_surfaces_wired_operations() -> None:
    page = list_operations(limit=100)
    ids = {row["id"] for row in page["results"]}
    assert set(WIRED_IDS) <= ids
    assert page["matching_count"] == len(page["results"])
    assert page["market_evidence"] is False


def test_unknown_operation_ids_remain_fail_closed() -> None:
    with pytest.raises(KeyError):
        get_operation("features.not_a_real_operation")


def test_rolling_linear_trend_matches_known_lines() -> None:
    result = execute_operation(
        "features.rolling_linear_trend",
        {"values": [1.0, 2.0, 3.0, 4.0, 5.0], "window": 3},
        workspace_root=Path("."),
    )
    payload = result["result"]
    assert result["operation_id"] == "features.rolling_linear_trend"
    assert result["research_only"] is True
    assert result["live_pnl_claim"] is False
    assert result["market_evidence"] is False
    assert payload["status"] == ["warmup", "warmup", "finite", "finite", "finite"]
    assert payload["slopes_per_observation"][2:] == pytest.approx([1.0, 1.0, 1.0])
    assert payload["centered_intercepts"][2:] == pytest.approx([2.0, 3.0, 4.0])
    assert payload["residual_scales"][2:] == pytest.approx([0.0, 0.0, 0.0])
    assert payload["r_squared"][2:] == pytest.approx([1.0, 1.0, 1.0])


def test_rolling_linear_trend_marks_constant_windows() -> None:
    result = execute_operation(
        "features.rolling_linear_trend",
        {"values": [4.0, 4.0, 4.0], "window": 3},
        workspace_root=Path("."),
    )
    payload = result["result"]
    assert payload["status"] == ["warmup", "warmup", "constant"]
    assert payload["slopes_per_observation"][2] == 0.0
    assert payload["centered_intercepts"][2] == 4.0
    assert payload["r_squared"][2] is None


def _mapping(
    mapping_id: str,
    security_id: str,
    *,
    ticker: str = "AAPL",
    valid_from: str = "2020-01-01T00:00:00Z",
    valid_to: str | None = None,
    available_time: str | None = None,
    revision_id: str = "r1",
) -> dict[str, str]:
    row = {
        "mapping_id": mapping_id,
        "security_id": security_id,
        "ticker": ticker,
        "exchange": "XNAS",
        "valid_from": valid_from,
        "available_time": available_time or valid_from,
        "revision_id": revision_id,
        "source": "synthetic",
    }
    if valid_to is not None:
        row["valid_to"] = valid_to
    return row


def test_resolve_security_identity_registered_and_resolves_intervals() -> None:
    request = {
        "mappings": [
            _mapping("M1", "SEC_OLD", available_time="2020-01-01T00:00:00Z"),
            _mapping("M1", "SEC_NEW", available_time="2020-06-01T00:00:00Z", revision_id="r2"),
            _mapping("M2", "SEC_B", ticker="MSFT"),
            _mapping("M3", "SEC_C", ticker="TSLA", valid_to="2020-12-31T00:00:00Z"),
        ],
        "queries": [
            {"ticker": "AAPL", "exchange": "XNAS", "decision_time": "2020-03-01T00:00:00Z"},
            {"ticker": "AAPL", "exchange": "XNAS", "decision_time": "2020-09-01T00:00:00Z"},
            {"ticker": "MSFT", "exchange": "XNAS", "decision_time": "2020-03-01T00:00:00Z"},
            {"ticker": "TSLA", "exchange": "XNAS", "decision_time": "2020-12-31T00:00:00Z"},
        ],
    }
    result = execute_operation(
        "skills.resolve_security_identity", request, workspace_root=Path(".")
    )
    payload = result["result"]
    assert result["operation_id"] == "skills.resolve_security_identity"
    assert payload["resolved_count"] == 3
    assert payload["missing_count"] == 1
    assert payload["ambiguous_count"] == 0
    statuses = [resolution["status"] for resolution in payload["resolutions"]]
    assert statuses == ["resolved", "resolved", "resolved", "missing"]
    assert payload["resolutions"][0]["security_id"] == "SEC_OLD"
    assert payload["resolutions"][1]["security_id"] == "SEC_NEW"
    # valid_to is exclusive: a query exactly at the boundary has no mapping.
    assert payload["resolutions"][3]["security_id"] is None


def test_resolve_security_identity_reports_ambiguity() -> None:
    request = {
        "mappings": [
            _mapping("M1", "SEC_A", ticker="AAPL"),
            _mapping("M2", "SEC_B", ticker="AAPL"),
        ],
        "queries": [
            {"ticker": "AAPL", "exchange": "XNAS", "decision_time": "2021-01-01T00:00:00Z"}
        ],
    }
    payload = execute_operation(
        "skills.resolve_security_identity", request, workspace_root=Path(".")
    )["result"]
    resolution = payload["resolutions"][0]
    assert resolution["status"] == "ambiguous"
    assert resolution["security_id"] is None
    assert resolution["candidate_security_count"] == 2
    assert payload["ambiguous_count"] == 1
