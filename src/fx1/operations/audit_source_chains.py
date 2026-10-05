"""Audit source-separated declared sequence/predecessor graphs and caller anchors.

Each source starts at sequence zero with no predecessor, or continues after its
one caller anchor. A link must name a predecessor from the same source at the
immediately preceding sequence. Hash labels identify nodes and may not be reused
with conflicting sequence/predecessor metadata. Duplicate identical metadata is
handled by explicit policy. Fork and cycle checks also inspect inconsistent
metadata rather than silently choosing a representative chain.

Optional payload_hex is hashed as exactly those raw bytes using SHA-256. No
framing, sequence, source or predecessor metadata is added to that preimage.
Consequently even matching supplied bytes do not authenticate chain links,
anchors, source identity or completeness. Missing sequence spans are bounded
by the observed maximum only; no missing tail is inferred.

Cycle diagnostics enumerate deterministic DFS back-edge witnesses, not all
possible graph cycles. Long witnesses retain a bounded path prefix plus the
closing edge and report omitted edge indexes; original input retains the proof.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from itertools import islice
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Digest = Annotated[str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
Sequence = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Payload = Annotated[str, Field(strict=True, max_length=8_192, pattern=r"^(?:[0-9a-f]{2})*$")]


class Anchor(InputModel):
    source_id: Name
    sequence: Sequence
    record_hash: Digest


class Record(InputModel):
    source_id: Name
    sequence: Sequence
    record_hash: Digest
    previous_hash: Digest | None
    payload_hex: Payload | None = None


class Input(InputModel):
    anchors: list[Anchor] = Field(default_factory=list, max_length=1_024)
    records: list[Record] = Field(max_length=10_000)
    duplicate_policy: Literal["reject", "allow_identical_metadata"] = "reject"
    offset: int = Field(default=0, strict=True, ge=0, le=1_024)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indexes: int = Field(default=5, strict=True, ge=2, le=10)

    @model_validator(mode="after")
    def source_and_payload_bounds(self) -> Self:
        sources = {row.source_id for row in self.anchors}
        if len(sources) != len(self.anchors):
            raise ValueError("a source may have only one predecessor anchor")
        sources.update(row.source_id for row in self.records)
        if len(sources) > 1_024:
            raise ValueError("at most 1024 distinct sources are supported")
        if sum(len(row.payload_hex or "") for row in self.records) > 524_288:
            raise ValueError("combined supplied payloads may not exceed 262144 raw bytes")
        return self


class Diagnostic(OutputModel):
    source_id: str
    code: str
    is_violation: bool
    anchor_row_index: int | None
    first_sequence: int | None
    last_sequence: int | None
    record_hash: str | None
    affected_count: int
    record_row_indexes: list[int]
    omitted_record_row_indexes: int


class SourceSummary(OutputModel):
    source_id: str
    anchor_row_index: int | None
    record_row_count: int
    distinct_sequences: int
    distinct_hash_labels: int
    observed_min_sequence: int | None
    observed_max_sequence: int | None
    missing_sequence_count: int
    duplicate_metadata_row_count: int
    tip_hash_count: int
    cycle_back_edge_witness_count: int
    supplied_payload_count: int
    matching_payload_hash_count: int
    issue_counts: dict[str, int]
    violation_count: int
    passed: bool


class Output(OutputModel):
    passed: bool
    source_count: int
    record_row_count: int
    anchor_count: int
    duplicate_policy: str
    hash_scope: Literal["labels_except_optional_raw_payload_sha256"] = (
        "labels_except_optional_raw_payload_sha256"
    )
    chain_metadata_committed_by_payload_hash: Literal[False] = False
    authenticity_established: Literal[False] = False
    missing_tail_inferred: Literal[False] = False
    issue_counts: dict[str, int]
    violation_count: int
    diagnostic_count: int
    omitted_diagnostics: int
    offset: int
    next_offset: int | None
    sources: list[SourceSummary]
    diagnostics: list[Diagnostic]


def execute(request: Input, context: OperationContext) -> Output:
    anchors = {row.source_id: index for index, row in enumerate(request.anchors)}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(request.records):
        grouped[row.source_id].append(index)
    sources = sorted(set(anchors) | set(grouped))
    issues: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    violations: Counter[str] = Counter()
    diagnostics: list[Diagnostic] = []
    summaries: list[SourceSummary] = []

    def report(
        source: str,
        code: str,
        rows: list[int],
        count: int,
        first: int | None = None,
        last: int | None = None,
        digest: str | None = None,
        violation: bool = True,
        omitted: int = 0,
    ) -> None:
        issues[code] += 1
        per_source[source][code] += 1
        violations[source] += violation
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Diagnostic(
                    source_id=source,
                    code=code,
                    is_violation=violation,
                    anchor_row_index=anchors.get(source),
                    first_sequence=first,
                    last_sequence=last,
                    record_hash=digest,
                    affected_count=count,
                    record_row_indexes=rows[: request.max_row_indexes],
                    omitted_record_row_indexes=omitted
                    + max(0, len(rows) - request.max_row_indexes),
                )
            )

    for source_position, source in enumerate(sources):
        indexes = grouped.get(source, [])
        anchor_index = anchors.get(source)
        anchor = request.anchors[anchor_index] if anchor_index is not None else None
        by_sequence: dict[int, list[int]] = defaultdict(list)
        by_hash: dict[str, list[int]] = defaultdict(list)
        metadata: dict[tuple[int, str, str | None], list[int]] = defaultdict(list)
        children: dict[str | None, dict[str, int]] = defaultdict(dict)
        payload_count = matching_payloads = duplicate_rows = 0
        for index in indexes:
            row = request.records[index]
            by_sequence[row.sequence].append(index)
            by_hash[row.record_hash].append(index)
            metadata[(row.sequence, row.record_hash, row.previous_hash)].append(index)
            children[row.previous_hash].setdefault(row.record_hash, index)
            if anchor and (
                row.sequence <= anchor.sequence or row.record_hash == anchor.record_hash
            ):
                report(
                    source,
                    "record_reuses_or_precedes_anchor",
                    [index],
                    1,
                    row.sequence,
                    row.sequence,
                    row.record_hash,
                )
            if row.payload_hex is not None:
                payload_count += 1
                computed = hashlib.sha256(bytes.fromhex(row.payload_hex)).hexdigest()
                matching_payloads += computed == row.record_hash
                if computed != row.record_hash:
                    report(source, "payload_hash_mismatch", [index], 1, digest=row.record_hash)
        for signature, rows in sorted(metadata.items(), key=lambda item: item[1][0]):
            if len(rows) > 1:
                duplicate_rows += len(rows) - 1
                report(
                    source,
                    "duplicate_metadata",
                    rows,
                    len(rows) - 1,
                    signature[0],
                    signature[0],
                    signature[1],
                    request.duplicate_policy == "reject",
                )
        for sequence, rows in sorted(by_sequence.items()):
            variants: dict[tuple[str, str | None], int] = {}
            for index in rows:
                row = request.records[index]
                variants.setdefault((row.record_hash, row.previous_hash), index)
            if len(variants) > 1:
                report(
                    source,
                    "sequence_conflict",
                    list(variants.values()),
                    len(variants),
                    sequence,
                    sequence,
                )
        hash_variants: dict[str, dict[tuple[int, str | None], int]] = {}
        for digest, rows in sorted(by_hash.items()):
            variants_by_hash: dict[tuple[int, str | None], int] = {}
            payloads: dict[str, int] = {}
            for index in rows:
                row = request.records[index]
                variants_by_hash.setdefault((row.sequence, row.previous_hash), index)
                if row.payload_hex is not None:
                    payloads.setdefault(row.payload_hex, index)
            hash_variants[digest] = variants_by_hash
            if len(variants_by_hash) > 1:
                report(
                    source,
                    "hash_label_conflict",
                    list(variants_by_hash.values()),
                    len(variants_by_hash),
                    digest=digest,
                )
            if len(payloads) > 1:
                report(
                    source,
                    "payload_bytes_conflict",
                    list(payloads.values()),
                    len(payloads),
                    digest=digest,
                )
        for predecessor, child_hashes in sorted(
            children.items(), key=lambda item: (item[0] is not None, item[0] or "")
        ):
            if len(child_hashes) > 1:
                report(
                    source,
                    "predecessor_fork",
                    list(child_hashes.values()),
                    len(child_hashes),
                    digest=predecessor,
                )

        for (sequence, digest, predecessor), rows in sorted(
            metadata.items(), key=lambda item: item[1][0]
        ):
            witness = rows[0]
            if predecessor is None:
                if sequence != 0 or anchor is not None:
                    report(source, "unexpected_genesis", [witness], 1, sequence, sequence, digest)
                continue
            if anchor is not None and predecessor == anchor.record_hash:
                if sequence != anchor.sequence + 1:
                    report(
                        source,
                        "anchor_sequence_discontinuity",
                        [witness],
                        1,
                        sequence,
                        sequence,
                        predecessor,
                    )
                continue
            parents = hash_variants.get(predecessor)
            if parents is None:
                report(source, "missing_predecessor", [witness], 1, sequence, sequence, predecessor)
            elif len(parents) > 1:
                parent_rows = [witness, *islice(parents.values(), request.max_row_indexes - 1)]
                report(
                    source,
                    "ambiguous_predecessor",
                    parent_rows,
                    len(parents),
                    sequence,
                    sequence,
                    predecessor,
                    omitted=len(parents) + 1 - len(parent_rows),
                )
            else:
                (parent_sequence, _), parent_index = next(iter(parents.items()))
                if parent_sequence != sequence - 1:
                    report(
                        source,
                        "predecessor_sequence_discontinuity",
                        [parent_index, witness],
                        1,
                        parent_sequence,
                        sequence,
                        predecessor,
                    )

        cursor = anchor.sequence + 1 if anchor else 0
        missing = 0
        for sequence, rows in sorted(by_sequence.items()):
            if sequence < cursor:
                continue
            if sequence > cursor:
                missing += sequence - cursor
                report(
                    source,
                    "missing_sequence_span",
                    [rows[0]],
                    sequence - cursor,
                    cursor,
                    sequence - 1,
                )
            cursor = sequence + 1

        adjacency: dict[str, list[tuple[str, int]]] = {}
        for digest, graph_variants in hash_variants.items():
            unique_edges: dict[str, int] = {}
            for (_, predecessor), index in graph_variants.items():
                if predecessor is not None and predecessor in by_hash:
                    unique_edges.setdefault(predecessor, index)
            adjacency[digest] = sorted(unique_edges.items())
        colors: dict[str, int] = {}
        cycle_count = 0
        for origin in sorted(adjacency):
            if colors.get(origin, 0):
                continue
            stack: list[tuple[str, int]] = [(origin, 0)]
            path_rows: list[int] = []
            positions = {origin: 0}
            colors[origin] = 1
            while stack:
                node, next_edge = stack[-1]
                if next_edge == len(adjacency[node]):
                    colors[node] = 2
                    del positions[node]
                    stack.pop()
                    if path_rows:
                        path_rows.pop()
                    continue
                neighbor, edge_index = adjacency[node][next_edge]
                stack[-1] = node, next_edge + 1
                color = colors.get(neighbor, 0)
                if color == 0:
                    colors[neighbor] = 1
                    positions[neighbor] = len(stack)
                    stack.append((neighbor, 0))
                    path_rows.append(edge_index)
                elif color == 1:
                    cycle_count += 1
                    start = positions[neighbor]
                    cycle_size = len(stack) - start
                    prefix = path_rows[start : start + request.max_row_indexes - 1]
                    witness_rows = [*prefix, edge_index]
                    report(
                        source,
                        "cycle_back_edge",
                        witness_rows,
                        cycle_size,
                        digest=neighbor,
                        omitted=cycle_size - len(witness_rows),
                    )
        if request.offset <= source_position < request.offset + request.limit:
            summaries.append(
                SourceSummary(
                    source_id=source,
                    anchor_row_index=anchor_index,
                    record_row_count=len(indexes),
                    distinct_sequences=len(by_sequence),
                    distinct_hash_labels=len(by_hash),
                    observed_min_sequence=min(by_sequence) if by_sequence else None,
                    observed_max_sequence=max(by_sequence) if by_sequence else None,
                    missing_sequence_count=missing,
                    duplicate_metadata_row_count=duplicate_rows,
                    tip_hash_count=len(set(by_hash) - set(children)),
                    cycle_back_edge_witness_count=cycle_count,
                    supplied_payload_count=payload_count,
                    matching_payload_hash_count=matching_payloads,
                    issue_counts=dict(sorted(per_source[source].items())),
                    violation_count=violations[source],
                    passed=violations[source] == 0,
                )
            )
    diagnostic_count = sum(issues.values())
    end = min(len(sources), request.offset + request.limit)
    return Output(
        passed=sum(violations.values()) == 0,
        source_count=len(sources),
        record_row_count=len(request.records),
        anchor_count=len(request.anchors),
        duplicate_policy=request.duplicate_policy,
        issue_counts=dict(sorted(issues.items())),
        violation_count=sum(violations.values()),
        diagnostic_count=diagnostic_count,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
        offset=request.offset,
        next_offset=end if end < len(sources) else None,
        sources=summaries,
        diagnostics=diagnostics,
    )


OPERATION = Operation(
    id="skills.audit_source_chains",
    kind="skill",
    description=(
        "Audit source-separated sequence/hash-predecessor declarations, caller anchors, "
        "forks, conflicts, missing links and cycles, with optional raw payload hash checks."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
