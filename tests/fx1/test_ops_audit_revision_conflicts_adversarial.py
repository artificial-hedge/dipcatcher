"""Adversarial probes for revision-conflict detection.

Identical repeats must report as exact duplicates, never conflicts; any
contradiction in source, availability, or values must surface as a conflict
field; and value comparison is canonical-JSON type-sensitive so 1, 1.0, true,
and "1" cannot blur into one variant. Groups are bound to exact
(security, event instant, revision) keys — shifting any one escapes the group.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_revision_conflicts as rev,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def record(**updates: object) -> dict[str, object]:
    return {
        "security_id": "s",
        "event_time": "2026-01-01T10:00:00Z",
        "revision_id": "r1",
        "available_time": "2026-01-01T10:05:00Z",
        "source": "x",
        "values": {"k": 1},
        **updates,
    }


def run(records: list[dict[str, object]], context: OperationContext) -> rev.Output:
    return rev.execute(rev.Input.model_validate({"records": records}), context)


def test_identical_repeats_are_exact_duplicates_not_conflicts(
    context: OperationContext,
) -> None:
    result = run([record(), record(), record()], context)
    assert result.conflict_free
    assert not result.unique_and_consistent
    assert result.exact_duplicate_rows == 2
    assert result.exact_duplicate_only_groups == 1
    assert result.findings[0].status == "exact_duplicates_only"


def test_each_conflict_field_isolated(context: OperationContext) -> None:
    by_source = run([record(), record(source="y")], context)
    assert by_source.findings[0].conflict_fields == ["source"]
    by_clock = run([record(), record(available_time="2026-01-01T10:06:00Z")], context)
    assert by_clock.findings[0].conflict_fields == ["available_time"]
    by_values = run([record(), record(values={"k": 2})], context)
    assert by_values.findings[0].conflict_fields == ["values"]
    for result in (by_source, by_clock, by_values):
        assert not result.conflict_free
        assert result.conflicting_revision_groups == 1
        assert result.rows_in_conflicting_groups == 2


def test_all_three_conflict_fields_can_coexist(context: OperationContext) -> None:
    result = run(
        [
            record(),
            record(
                source="y",
                available_time="2026-01-01T10:06:00Z",
                values={"k": 9},
            ),
        ],
        context,
    )
    assert result.findings[0].conflict_fields == [
        "source",
        "available_time",
        "values",
    ]
    assert result.conflict_field_counts == {
        "available_time": 1,
        "source": 1,
        "values": 1,
    }


def test_payload_order_cannot_fake_a_conflict(context: OperationContext) -> None:
    """Canonical JSON binds key order away — reordered dicts are duplicates."""
    result = run(
        [
            record(values={"a": 1, "b": 2}),
            record(values={"b": 2, "a": 1}),
        ],
        context,
    )
    assert result.conflict_free
    assert result.exact_duplicate_rows == 1


def test_value_comparison_is_type_sensitive(context: OperationContext) -> None:
    pairs = [(1, 1.0), (1, True), (1, "1"), (0, False), (None, 0)]
    for left, right in pairs:
        result = run([record(values={"k": left}), record(values={"k": right})], context)
        assert result.findings[0].status == "conflicting_revision", (left, right)
        assert result.findings[0].conflict_fields == ["values"], (left, right)


def test_group_key_escape_defeats_conflict_hiding(context: OperationContext) -> None:
    """Contradictions under a different revision/event/security stay separate
    groups — they cannot merge into a hidden conflict or a fake duplicate."""
    result = run(
        [
            record(values={"k": 1}),
            record(values={"k": 2}, revision_id="r2"),
            record(values={"k": 3}, event_time="2026-01-01T10:01:00Z"),
            record(values={"k": 4}, security_id="other"),
        ],
        context,
    )
    assert result.revision_group_count == 4
    assert result.conflict_free
    assert result.unique_and_consistent
    assert result.findings == []


def test_equivalent_instant_offsets_share_a_group(context: OperationContext) -> None:
    """The same event instant in two offsets is one group — offsets cannot
    escape conflict detection."""
    result = run(
        [
            record(),
            record(event_time="2026-01-01T15:30:00+05:30", values={"k": 2}),
        ],
        context,
    )
    assert result.conflicting_revision_groups == 1
    assert result.findings[0].conflict_fields == ["values"]


def test_variant_row_indices_prefer_differing_variants(
    context: OperationContext,
) -> None:
    result = rev.execute(
        rev.Input.model_validate(
            {
                "records": [
                    record(),
                    record(),
                    record(),
                    record(values={"k": 2}),
                ],
                "max_indices_per_group": 2,
            }
        ),
        context,
    )
    finding = result.findings[0]
    assert finding.row_count == 4
    assert finding.unique_variant_count == 2
    assert finding.exact_duplicate_rows == 2
    assert 3 in finding.row_indices
    assert finding.omitted_row_indices == 2


def test_conflicting_groups_sort_before_duplicate_only(context: OperationContext) -> None:
    result = run(
        [
            record(security_id="z", values={"k": 1}),
            record(security_id="z", values={"k": 1}),
            record(security_id="a", values={"k": 1}),
            record(security_id="a", values={"k": 2}),
        ],
        context,
    )
    assert [f.status for f in result.findings] == [
        "conflicting_revision",
        "exact_duplicates_only",
    ]
    assert result.findings[0].security_id == "a"


@pytest.mark.parametrize(
    "field",
    ["event_time", "available_time"],
)
@pytest.mark.parametrize("clock", ["2026-01-01T10:00:00", "1767223200", "", "soon"])
def test_hostile_clocks_rejected(field: str, clock: object) -> None:
    with pytest.raises(ValidationError):
        rev.Input.model_validate({"records": [record(**{field: clock})]})


def test_hostile_value_cells_rejected() -> None:
    with pytest.raises(ValidationError):
        rev.Input.model_validate({"records": [record(values={"k": [1]})]})
    with pytest.raises(ValidationError):
        rev.Input.model_validate({"records": [record(values={"k": float("nan")})]})
    with pytest.raises(ValidationError):
        rev.Input.model_validate({"records": [record(values={})]})


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    records = [
        record(),
        record(values={"k": 2}),
        record(),
        record(security_id="q"),
    ]
    first = run(records, context)
    second = run(records, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
