"""Reconcile complete snapshot inventories through explicit parent deltas.

A root inventory is a caller-declared baseline and must have no delta. A child
has one exact parent and at most one change per object. expected_parent_version
asserts the prior live or tombstone version; null asserts absence. Every change
introduces a different version. A delete requires a live prior object, while an
upsert may create, replace or resurrect with the exact prior-version precondition.
Tombstones are retained in complete inventories: omission never means deletion.

Object/version declarations are immutable across entries and changes: state,
size, hash label and availability must agree wherever the same identity appears.
State clocks may not regress, parent publication cannot follow child publication,
and every claimed entry/change must be available by its snapshot publication.
Unknown/ambiguous parents, failed preconditions and cycles withhold reconstruction
for descendants. Inventory equality checks occur only after a complete valid
delta application; partial application is never labeled a matching snapshot.

Queries name an exact snapshot and object at a decision clock. They preserve both
the current inventory row and the root-entry/change row introducing that state.
Global findings/summaries audit every declaration, including future metadata.
Queries independently reconstruct visible snapshot declarations: future duplicate
IDs or object-version conflicts cannot invalidate an earlier observable state.
Visible duplicates/conflicts still withhold that state; there is no fallback.
Hashes remain labels; no storage bytes, authenticated version history, atomicity
or external population completeness are verified.

Bounds: 500 snapshots, 20000 inventory entries and 20000 changes in total, 1000
queries at up to 16 distinct decision clocks, 200 findings and 200 query results/page.
Sum of parent inventory sizes, child inventory sizes and delta sizes, multiplied
by one global audit plus the number of requested decision views, is capped at
500000 before reconstruction. This is a conservative charge for every supplied
row at every requested clock, including empty or reused views. Views are cached;
ambiguous parents never select a row.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Digest = Annotated[str, Field(strict=True, pattern=r"\A[0-9a-f]{64}\z")]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("clock must be an explicit timezone-aware ISO datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class ObjectState(InputModel):
    object_id: Name
    version_id: Name
    state: Literal["live", "deleted"]
    size: int | None = Field(default=None, strict=True, ge=0, le=2**63 - 1)
    sha256: Digest | None = None
    available_time: AwareDatetime

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def state_payload(self) -> Self:
        if self.state == "live" and (self.size is None or self.sha256 is None):
            raise ValueError("live states require size and sha256 labels")
        if self.state == "deleted" and (self.size is not None or self.sha256 is not None):
            raise ValueError("tombstones cannot carry live size/hash fields")
        return self


class Change(ObjectState):
    expected_parent_version: Name | None


class Snapshot(InputModel):
    snapshot_id: Name
    parent_snapshot_id: Name | None = None
    available_time: AwareDatetime
    inventory: list[ObjectState] = Field(max_length=20_000)
    changes: list[Change] = Field(default_factory=list, max_length=20_000)

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Query(InputModel):
    snapshot_id: Name
    object_id: Name
    decision_time: AwareDatetime

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    snapshots: list[Snapshot] = Field(min_length=1, max_length=500)
    queries: list[Query] = Field(default_factory=list, max_length=1000)
    max_state_comparisons: int = Field(default=100_000, strict=True, ge=0, le=500_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def reconstruction_budget(self) -> Self:
        entries = sum(len(row.inventory) for row in self.snapshots)
        changes = sum(len(row.changes) for row in self.snapshots)
        if entries > 20_000 or changes > 20_000:
            raise ValueError("at most 20000 inventory entries and 20000 changes are supported")
        parent_sizes: Counter[str] = Counter()
        for row in self.snapshots:
            parent_sizes[row.snapshot_id] += len(row.inventory)
        decisions = {query.decision_time for query in self.queries}
        if len(decisions) > 16:
            raise ValueError("at most 16 distinct snapshot query decision clocks are supported")
        work = (1 + len(decisions)) * (
            entries
            + changes
            + sum(
                parent_sizes[row.parent_snapshot_id]
                for row in self.snapshots
                if row.parent_snapshot_id is not None
            )
        )
        if work > self.max_state_comparisons:
            raise ValueError("declared snapshot reconstruction work exceeds max_state_comparisons")
        return self


class Finding(OutputModel):
    code: str
    snapshot_row_index: int
    inventory_row_index: int | None = None
    change_row_index: int | None = None
    source_snapshot_row_index: int | None = None
    object_id: str | None = None


class SnapshotResult(OutputModel):
    snapshot_row_index: int
    snapshot_id: str
    parent_snapshot_row_index: int | None
    status: Literal["consistent_root", "reconciled_delta", "invalid", "cyclic_or_cycle_dependent"]
    declared_inventory_count: int
    reconstructed_inventory_count: int | None
    live_object_count: int | None
    tombstone_count: int | None
    maximum_source_available_time: datetime | None


class QueryResult(OutputModel):
    query_row_index: int
    status: Literal[
        "unknown_snapshot",
        "ambiguous_snapshot",
        "not_available",
        "invalid_snapshot",
        "absent",
        "live",
        "deleted",
    ]
    snapshot_row_index: int | None
    inventory_row_index: int | None
    version_id: str | None
    size: int | None
    sha256: str | None
    state_available_time: datetime | None
    origin_snapshot_row_index: int | None
    origin_inventory_row_index: int | None
    origin_change_row_index: int | None


class Output(OutputModel):
    passed: bool
    snapshot_count: int
    inventory_entry_count: int
    change_count: int
    declared_reconstruction_work: int
    requested_decision_view_count: int
    consistent_snapshot_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    snapshots: list[SnapshotResult]
    query_count: int
    query_status_counts: dict[str, int]
    queries: list[QueryResult]
    offset: int
    has_more: bool
    object_bytes_verified: Literal[False] = False
    external_snapshot_completeness_verified: Literal[False] = False
    query_reconstruction_scope: Literal["visible_snapshot_declarations_only"] = (
        "visible_snapshot_declarations_only"
    )


@dataclass(frozen=True)
class _Origin:
    state: ObjectState
    snapshot: int
    inventory: int | None
    change: int | None


@dataclass
class _View:
    snapshot_rows: dict[str, list[int]]
    inventories: list[dict[str, int]]
    parents: list[int | None]
    states: list[dict[str, _Origin] | None]
    visited: set[int]


_IssueSink = Callable[[str, int, int | None, int | None, int | None, str | None], None]


def _signature(row: ObjectState) -> tuple[str, int | None, str | None, datetime]:
    return row.state, row.size, row.sha256, row.available_time


def _reconstruct(request: Input, active: set[int], sink: _IssueSink | None) -> _View:
    invalid: set[int] = set()

    def report(
        code: str,
        snapshot: int,
        inventory: int | None = None,
        change: int | None = None,
        source: int | None = None,
        object_id: str | None = None,
    ) -> None:
        invalid.add(snapshot)
        if sink is not None:
            sink(code, snapshot, inventory, change, source, object_id)

    snapshot_rows: dict[str, list[int]] = defaultdict(list)
    inventories: list[dict[str, int]] = [{} for _ in request.snapshots]
    version_signatures: dict[tuple[str, str], tuple[str, int | None, str | None, datetime]] = {}
    conflicts: set[tuple[str, str]] = set()
    occurrences: list[tuple[int, int | None, int | None, ObjectState]] = []
    for index, snapshot in enumerate(request.snapshots):
        if index not in active:
            continue
        snapshot_rows[snapshot.snapshot_id].append(index)
        inventory_counts = Counter(row.object_id for row in snapshot.inventory)
        change_counts = Counter(row.object_id for row in snapshot.changes)
        inventories[index] = {
            row.object_id: local
            for local, row in enumerate(snapshot.inventory)
            if inventory_counts[row.object_id] == 1
        }
        for local, entry in enumerate(snapshot.inventory):
            if inventory_counts[entry.object_id] != 1:
                report(
                    "ambiguous_inventory_object",
                    snapshot=index,
                    inventory=local,
                    object_id=entry.object_id,
                )
            occurrences.append((index, local, None, entry))
        for local, change in enumerate(snapshot.changes):
            if change_counts[change.object_id] != 1:
                report(
                    "multiple_changes_for_object",
                    snapshot=index,
                    change=local,
                    object_id=change.object_id,
                )
            occurrences.append((index, None, local, change))
        if snapshot.parent_snapshot_id is None and snapshot.changes:
            report("root_snapshot_cannot_have_delta", snapshot=index)
    unique = {name: rows[0] for name, rows in snapshot_rows.items() if len(rows) == 1}
    for index, snapshot in enumerate(request.snapshots):
        if index not in active:
            continue
        if len(snapshot_rows[snapshot.snapshot_id]) != 1:
            report("ambiguous_snapshot_id", snapshot=index)
    for index, inventory_index, change_index, entry in occurrences:
        key = (entry.object_id, entry.version_id)
        signature = _signature(entry)
        if key in version_signatures and version_signatures[key] != signature:
            conflicts.add(key)
        version_signatures.setdefault(key, signature)
        if entry.available_time > request.snapshots[index].available_time:
            report(
                "object_state_after_snapshot_publication",
                snapshot=index,
                inventory=inventory_index,
                change=change_index,
                object_id=entry.object_id,
            )
    for index, inventory_index, change_index, entry in occurrences:
        if (entry.object_id, entry.version_id) in conflicts:
            report(
                "immutable_object_version_conflict",
                snapshot=index,
                inventory=inventory_index,
                change=change_index,
                object_id=entry.object_id,
            )

    parents: list[int | None] = [None] * len(request.snapshots)
    children: list[list[int]] = [[] for _ in request.snapshots]
    indegree = [0] * len(request.snapshots)
    for index, snapshot in enumerate(request.snapshots):
        if index not in active:
            continue
        if snapshot.parent_snapshot_id is None:
            continue
        parent = unique.get(snapshot.parent_snapshot_id)
        if parent is None:
            report("unresolved_parent_snapshot", snapshot=index)
            continue
        parents[index] = parent
        children[parent].append(index)
        indegree[index] = 1
        if request.snapshots[parent].available_time > snapshot.available_time:
            report("child_precedes_parent_publication", snapshot=index, source=parent)

    states: list[dict[str, _Origin] | None] = [None] * len(request.snapshots)
    pending = deque(
        index for index, degree in enumerate(indegree) if degree == 0 and index in active
    )
    visited: set[int] = set()
    while pending:
        index = pending.popleft()
        visited.add(index)
        snapshot = request.snapshots[index]
        parent = parents[index]
        if parent is not None and states[parent] is None:
            report("invalid_parent_snapshot", snapshot=index, source=parent)
        if index not in invalid:
            if snapshot.parent_snapshot_id is None:
                states[index] = {
                    entry.object_id: _Origin(entry, index, local, None)
                    for local, entry in enumerate(snapshot.inventory)
                }
            elif parent is not None:
                parent_state = states[parent]
                if parent_state is None:
                    raise ValueError("valid child lacks reconstructed parent state")
                derived = parent_state.copy()
                for local, change in enumerate(snapshot.changes):
                    prior = parent_state.get(change.object_id)
                    prior_version = None if prior is None else prior.state.version_id
                    if change.expected_parent_version != prior_version:
                        report(
                            "delta_parent_version_precondition_failed",
                            snapshot=index,
                            change=local,
                            source=parent,
                            object_id=change.object_id,
                        )
                        continue
                    if change.state == "deleted" and (prior is None or prior.state.state != "live"):
                        report(
                            "delete_requires_live_parent_object",
                            snapshot=index,
                            change=local,
                            source=parent,
                            object_id=change.object_id,
                        )
                        continue
                    if prior is not None and (
                        change.version_id == prior.state.version_id
                        or change.available_time < prior.state.available_time
                    ):
                        report(
                            "delta_version_reuse_or_clock_regression",
                            snapshot=index,
                            change=local,
                            source=parent,
                            object_id=change.object_id,
                        )
                        continue
                    derived[change.object_id] = _Origin(change, index, None, local)
                if index not in invalid:
                    declared = inventories[index]
                    for object_id in sorted(set(derived) - set(declared)):
                        report(
                            "inventory_omits_retained_object_or_tombstone",
                            snapshot=index,
                            source=parent,
                            object_id=object_id,
                        )
                    for object_id in sorted(set(declared) - set(derived)):
                        report(
                            "inventory_unexplained_object",
                            snapshot=index,
                            inventory=declared[object_id],
                            object_id=object_id,
                        )
                    for object_id in sorted(set(derived) & set(declared)):
                        entry = snapshot.inventory[declared[object_id]]
                        expected = derived[object_id].state
                        if entry.version_id != expected.version_id or _signature(
                            entry
                        ) != _signature(expected):
                            report(
                                "inventory_state_mismatch",
                                snapshot=index,
                                inventory=declared[object_id],
                                source=derived[object_id].snapshot,
                                object_id=object_id,
                            )
                    if index not in invalid:
                        states[index] = derived
        for child in children[index]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    for index in sorted(active):
        if index not in visited:
            report("cyclic_or_cycle_dependent_snapshot", snapshot=index, source=parents[index])

    return _View(snapshot_rows, inventories, parents, states, visited)


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []

    def record_issue(
        code: str,
        snapshot: int,
        inventory: int | None,
        change: int | None,
        source: int | None,
        object_id: str | None,
    ) -> None:
        issues[code] += 1
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    snapshot_row_index=snapshot,
                    inventory_row_index=inventory,
                    change_row_index=change,
                    source_snapshot_row_index=source,
                    object_id=object_id,
                )
            )

    all_rows = set(range(len(request.snapshots)))
    global_view = _reconstruct(request, all_rows, record_issue)
    states, parents, visited = global_view.states, global_view.parents, global_view.visited

    summaries: list[SnapshotResult] = []
    for index, snapshot in enumerate(request.snapshots):
        state = states[index]
        status: Literal[
            "consistent_root", "reconciled_delta", "invalid", "cyclic_or_cycle_dependent"
        ] = (
            "cyclic_or_cycle_dependent"
            if index not in visited
            else "invalid"
            if state is None
            else "consistent_root"
            if snapshot.parent_snapshot_id is None
            else "reconciled_delta"
        )
        summaries.append(
            SnapshotResult(
                snapshot_row_index=index,
                snapshot_id=snapshot.snapshot_id,
                parent_snapshot_row_index=parents[index],
                status=status,
                declared_inventory_count=len(snapshot.inventory),
                reconstructed_inventory_count=None if state is None else len(state),
                live_object_count=None
                if state is None
                else sum(origin.state.state == "live" for origin in state.values()),
                tombstone_count=None
                if state is None
                else sum(origin.state.state == "deleted" for origin in state.values()),
                maximum_source_available_time=snapshot.available_time
                if state is not None
                else None,
            )
        )
    results: list[QueryResult] = []
    query_counts: Counter[str] = Counter()
    views: dict[datetime, _View] = {}
    for index, query in enumerate(request.queries):
        if query.decision_time not in views:
            active = {
                row
                for row, snapshot in enumerate(request.snapshots)
                if snapshot.available_time <= query.decision_time
            }
            views[query.decision_time] = (
                global_view if active == all_rows else _reconstruct(request, active, None)
            )
        view = views[query.decision_time]
        rows = view.snapshot_rows.get(query.snapshot_id, [])
        position = rows[0] if len(rows) == 1 else None
        origin: _Origin | None = None
        inventory_row = None
        query_status: Literal[
            "unknown_snapshot",
            "ambiguous_snapshot",
            "not_available",
            "invalid_snapshot",
            "absent",
            "live",
            "deleted",
        ]
        if not rows:
            query_status = (
                "not_available"
                if query.snapshot_id in global_view.snapshot_rows
                else "unknown_snapshot"
            )
        elif position is None:
            query_status = "ambiguous_snapshot"
        elif request.snapshots[position].available_time > query.decision_time:
            query_status = "not_available"
        else:
            queried_state = view.states[position]
            if queried_state is None:
                query_status = "invalid_snapshot"
            else:
                origin = queried_state.get(query.object_id)
                query_status = "absent" if origin is None else origin.state.state
                inventory_row = view.inventories[position].get(query.object_id)
        query_counts[query_status] += 1
        if request.offset <= index < request.offset + request.limit:
            results.append(
                QueryResult(
                    query_row_index=index,
                    status=query_status,
                    snapshot_row_index=position,
                    inventory_row_index=inventory_row,
                    version_id=None if origin is None else origin.state.version_id,
                    size=None if origin is None else origin.state.size,
                    sha256=None if origin is None else origin.state.sha256,
                    state_available_time=None if origin is None else origin.state.available_time,
                    origin_snapshot_row_index=None if origin is None else origin.snapshot,
                    origin_inventory_row_index=None if origin is None else origin.inventory,
                    origin_change_row_index=None if origin is None else origin.change,
                )
            )
    parent_sizes: Counter[str] = Counter()
    for snapshot in request.snapshots:
        parent_sizes[snapshot.snapshot_id] += len(snapshot.inventory)
    entries_count = sum(len(row.inventory) for row in request.snapshots)
    changes_count = sum(len(row.changes) for row in request.snapshots)
    declared_work = (1 + len(views)) * (
        entries_count
        + changes_count
        + sum(
            parent_sizes[row.parent_snapshot_id]
            for row in request.snapshots
            if row.parent_snapshot_id is not None
        )
    )
    return Output(
        passed=not issues,
        snapshot_count=len(request.snapshots),
        inventory_entry_count=entries_count,
        change_count=changes_count,
        declared_reconstruction_work=declared_work,
        requested_decision_view_count=len(views),
        consistent_snapshot_count=sum(state is not None for state in states),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        snapshots=summaries,
        query_count=len(request.queries),
        query_status_counts=dict(sorted(query_counts.items())),
        queries=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.queries),
    )


OPERATION = Operation(
    id="skills.audit_snapshot_manifests",
    kind="skill",
    description="Reconcile complete snapshot inventories through exact parent deltas and retained tombstones, auditing immutable versions, publication and bounded ancestry.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
