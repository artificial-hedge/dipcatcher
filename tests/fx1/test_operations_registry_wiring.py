"""Registry wiring checks for independently implemented fx-1 operations.

These fixtures are SYNTHETIC correctness checks, not market evidence.
"""

from __future__ import annotations

import pytest

from fx1.operations import registry

WIRED_OPERATIONS = (
    ("skills.audit_source_coverage", "skill", "fx1.operations.audit_source_coverage"),
    ("features.bipower_variation", "feature", "fx1.operations.bipower_variation"),
    ("features.permutation_entropy", "feature", "fx1.operations.permutation_entropy"),
)


@pytest.mark.parametrize(("operation_id", "kind", "module"), WIRED_OPERATIONS)
def test_registered_operation_is_reachable(operation_id: str, kind: str, module: str) -> None:
    operation = registry.get_operation(operation_id)
    assert operation.id == operation_id
    assert operation.kind == kind
    assert operation.handler.__module__ == module


def test_list_operations_reports_registered_operations() -> None:
    listing = registry.list_operations(limit=100)
    assert listing["implementation_count"] >= len(WIRED_OPERATIONS)
    ids = {row["id"] for row in listing["results"]}
    assert {operation_id for operation_id, _, _ in WIRED_OPERATIONS} <= ids


def test_bipower_variation_executes_through_registry() -> None:
    outcome = registry.execute_operation(
        "features.bipower_variation", {"returns": [0.01, -0.02, 0.03]}
    )
    assert outcome["operation_id"] == "features.bipower_variation"
    assert outcome["market_evidence"] is False
    assert outcome["live_pnl_claim"] is False
    assert outcome["result"]["bipower_status"] == "finite"


def test_permutation_entropy_executes_through_tool_dispatch() -> None:
    described = registry.invoke_operation_tool(
        "describe_operation", {"operation_id": "features.permutation_entropy"}
    )
    assert described["id"] == "features.permutation_entropy"
    outcome = registry.invoke_operation_tool(
        "execute_operation",
        {
            "operation_id": "features.permutation_entropy",
            "arguments": {
                "values": [1.0, 2.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "minimum_patterns": 1,
            },
        },
    )
    assert outcome["result"]["sample_status"] == "meets_threshold"


def test_audit_source_coverage_executes_through_registry() -> None:
    outcome = registry.execute_operation(
        "skills.audit_source_coverage",
        {
            "expected_pairs": [{"source": "SYNTH", "security_id": "A"}],
            "observations": [
                {
                    "source": "SYNTH",
                    "security_id": "A",
                    "event_time": "2026-01-01T09:00:00Z",
                    "available_time": "2026-01-01T09:01:00Z",
                }
            ],
            "decision_time": "2026-01-01T10:00:00Z",
        },
    )
    assert outcome["result"]["assessment"] == "covered"
    assert outcome["result"]["complete_expected_coverage"] is True
