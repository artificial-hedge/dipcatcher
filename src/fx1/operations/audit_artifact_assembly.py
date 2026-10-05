"""Audit declared byte assembly plans without opening or authenticating any bytes.

Each part copies [source_offset, source_offset+length) from one exact object
version to [target_offset, target_offset+length). Ordinals must be the unique
sequence 0..expected_part_count-1, and their concatenation must start at zero and
end at declared_size. Independent target-interval union checks describe gaps,
overlap and excess. Duplicate object versions, assembly IDs, part IDs or ordinals
are errors; no ambiguous declaration is silently selected.

Source and part metadata must be available by assembly publication. Publications
after decision_time are pending, not fabricated current artifacts. Hashes are
labels: expected_source_sha256 must match the referenced object's label; a whole
source part's label must match that object; a single full assembly part's label
must match the assembly label. The SHA-256 empty-message constant is checked for
declared empty assemblies. Multiple part digests cannot determine the concatenated
digest, so no such comparison is claimed. No bytes, signatures, file existence,
atomic publication or external storage are verified.

Bounds: 5000 objects, 1000 assemblies, 20000 parts and 20000 expected part slots
combined. Sizes/offsets are nonnegative signed-64-bit declarations; interval
arithmetic uses exact Python integers without expanding byte ranges. At most 200
findings and 200 assembly results are returned. All declarations are audited.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Digest = Annotated[str, Field(strict=True, pattern=r"\A[0-9a-f]{64}\z")]
Size = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


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


class SourceObject(InputModel):
    object_id: Name
    version_id: Name
    size: Size
    sha256: Digest
    available_time: AwareDatetime

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Assembly(InputModel):
    assembly_id: Name
    version_id: Name
    declared_size: Size
    sha256: Digest
    expected_part_count: int = Field(strict=True, ge=0, le=20_000)
    available_time: AwareDatetime

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Part(InputModel):
    part_id: Name
    assembly_id: Name
    ordinal: int = Field(strict=True, ge=0, le=19_999)
    source_object_id: Name
    source_version_id: Name
    expected_source_sha256: Digest
    source_offset: Size
    length: int = Field(strict=True, ge=1, le=2**63 - 1)
    target_offset: Size
    part_sha256: Digest
    available_time: AwareDatetime

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    objects: list[SourceObject] = Field(max_length=5000)
    assemblies: list[Assembly] = Field(min_length=1, max_length=1000)
    parts: list[Part] = Field(max_length=20_000)
    decision_time: AwareDatetime
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def expected_work(self) -> Self:
        if sum(row.expected_part_count for row in self.assemblies) > 20_000:
            raise ValueError("at most 20000 expected part slots are supported")
        return self


class Finding(OutputModel):
    code: str
    assembly_row_index: int | None = None
    part_row_index: int | None = None
    other_part_row_index: int | None = None
    object_row_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class AssemblyResult(OutputModel):
    assembly_row_index: int
    assembly_id: str
    version_id: str
    declared_size: int
    observed_part_count: int
    source_references_complete: bool
    maximum_declared_input_available_time: datetime | None
    missing_ordinal_count: int
    excess_ordinal_count: int
    ordinal_concatenation_matches: bool | None
    target_covered_bytes: int
    target_missing_bytes: int
    target_overlap_bytes: int
    target_multiplicity_excess_bytes: int
    target_excess_bytes: int
    hash_label_equation: Literal["empty_assembly", "single_part", "not_derivable"]
    hash_labels_match: bool | None
    available_at_decision: bool
    usable_declaration_at_decision: bool
    passed: bool


class Output(OutputModel):
    passed: bool
    object_count: int
    assembly_count: int
    part_count: int
    usable_declaration_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    assemblies: list[AssemblyResult]
    offset: int
    has_more: bool
    artifact_bytes_verified: Literal[False] = False
    publication_truth_verified: Literal[False] = False
    multipart_hash_derived: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid: set[int] = set()

    def report(
        code: str,
        assembly: int | None = None,
        part: int | None = None,
        other: int | None = None,
        source: int | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        issues[code] += 1
        if assembly is not None:
            invalid.add(assembly)
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    assembly_row_index=assembly,
                    part_row_index=part,
                    other_part_row_index=other,
                    object_row_index=source,
                    start_offset=start,
                    end_offset=end,
                )
            )

    object_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    assembly_rows: dict[str, list[int]] = defaultdict(list)
    for index, obj in enumerate(request.objects):
        object_rows[obj.object_id, obj.version_id].append(index)
    for index, assembly in enumerate(request.assemblies):
        assembly_rows[assembly.assembly_id].append(index)
    unique_objects = {key: rows[0] for key, rows in object_rows.items() if len(rows) == 1}
    unique_assemblies = {key: rows[0] for key, rows in assembly_rows.items() if len(rows) == 1}
    for rows in object_rows.values():
        if len(rows) > 1:
            for index in rows:
                report("ambiguous_object_version", source=index)
    for rows in assembly_rows.values():
        if len(rows) > 1:
            for index in rows:
                report("ambiguous_assembly_id", assembly=index)
    part_counts = Counter(part.part_id for part in request.parts)
    linked: list[list[int]] = [[] for _ in request.assemblies]
    complete = [True] * len(request.assemblies)
    maxima: list[datetime | None] = [None] * len(request.assemblies)
    for index, part in enumerate(request.parts):
        owner = unique_assemblies.get(part.assembly_id)
        if part_counts[part.part_id] > 1:
            report("ambiguous_part_id", assembly=owner, part=index)
        if owner is None:
            report("unresolved_part_assembly", part=index)
            continue
        linked[owner].append(index)
        assembly = request.assemblies[owner]
        if part.available_time > assembly.available_time:
            report("part_declared_after_assembly", assembly=owner, part=index)
        clock = part.available_time
        source = unique_objects.get((part.source_object_id, part.source_version_id))
        if source is None:
            complete[owner] = False
            report("unresolved_part_source_version", assembly=owner, part=index)
        else:
            obj = request.objects[source]
            clock = max(clock, obj.available_time)
            if obj.available_time > assembly.available_time:
                report("source_available_after_assembly", assembly=owner, part=index, source=source)
            if part.source_offset + part.length > obj.size:
                report("part_exceeds_source_extent", assembly=owner, part=index, source=source)
            if part.expected_source_sha256 != obj.sha256:
                report("source_hash_label_mismatch", assembly=owner, part=index, source=source)
            if (
                part.source_offset == 0
                and part.length == obj.size
                and part.part_sha256 != obj.sha256
            ):
                report(
                    "whole_source_part_hash_label_mismatch",
                    assembly=owner,
                    part=index,
                    source=source,
                )
        prior = maxima[owner]
        maxima[owner] = clock if prior is None else max(prior, clock)
        if part.target_offset + part.length > assembly.declared_size:
            report(
                "part_exceeds_target_extent",
                assembly=owner,
                part=index,
                start=part.target_offset,
                end=part.target_offset + part.length,
            )

    results: list[AssemblyResult] = []
    usable_count = 0
    for index, assembly in enumerate(request.assemblies):
        rows = linked[index]
        if len(assembly_rows[assembly.assembly_id]) != 1:
            complete[index] = False
        ordinals: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            ordinals[request.parts[row].ordinal].append(row)
        for same in ordinals.values():
            if len(same) > 1:
                report("duplicate_part_ordinal", assembly=index, part=same[0], other=same[1])
        missing_ordinals = assembly.expected_part_count - sum(
            ordinal < assembly.expected_part_count for ordinal in ordinals
        )
        excess_ordinals = sum(ordinal >= assembly.expected_part_count for ordinal in ordinals)
        if missing_ordinals:
            report("missing_part_ordinals", assembly=index)
        if excess_ordinals:
            report("unexpected_part_ordinals", assembly=index)
        ordered_matches: bool | None = None
        if (
            not missing_ordinals
            and not excess_ordinals
            and all(len(same) == 1 for same in ordinals.values())
            and complete[index]
        ):
            position = 0
            ordered_matches = True
            for ordinal in sorted(ordinals):
                row = ordinals[ordinal][0]
                part = request.parts[row]
                if part.target_offset != position:
                    ordered_matches = False
                    report(
                        "ordinal_target_offset_mismatch",
                        assembly=index,
                        part=row,
                        start=position,
                        end=part.target_offset,
                    )
                position += part.length
            if position != assembly.declared_size:
                ordered_matches = False
                report("concatenated_size_mismatch", assembly=index)

        changes: Counter[int] = Counter()
        for row in rows:
            part = request.parts[row]
            changes[part.target_offset] += 1
            changes[part.target_offset + part.length] -= 1
        changes[0] += 0
        changes[assembly.declared_size] += 0
        previous = active = covered = overlap = multiplicity_excess = outside = 0
        for point in sorted(changes):
            span = point - previous
            inside = max(
                0, min(point, assembly.declared_size) - min(previous, assembly.declared_size)
            )
            if active:
                covered += inside
                outside += span - inside
                multiplicity_excess += inside * (active - 1)
                if active > 1:
                    overlap += inside
            elif inside:
                report(
                    "target_coverage_gap",
                    assembly=index,
                    start=previous,
                    end=min(point, assembly.declared_size),
                )
            active += changes[point]
            previous = point
        if overlap:
            report("overlapping_target_parts", assembly=index)
        equation: Literal["empty_assembly", "single_part", "not_derivable"] = "not_derivable"
        labels_match: bool | None = None
        if not rows and assembly.expected_part_count == 0 and assembly.declared_size == 0:
            equation, labels_match = "empty_assembly", assembly.sha256 == _EMPTY_SHA256
        elif len(rows) == 1 and assembly.expected_part_count == 1 and ordered_matches:
            equation, labels_match = (
                "single_part",
                assembly.sha256 == request.parts[rows[0]].part_sha256,
            )
        if labels_match is False:
            report("assembly_hash_label_mismatch", assembly=index)
        visible = assembly.available_time <= request.decision_time
        usable = visible and index not in invalid
        usable_count += int(usable)
        if request.offset <= index < request.offset + request.limit:
            results.append(
                AssemblyResult(
                    assembly_row_index=index,
                    assembly_id=assembly.assembly_id,
                    version_id=assembly.version_id,
                    declared_size=assembly.declared_size,
                    observed_part_count=len(rows),
                    source_references_complete=complete[index],
                    maximum_declared_input_available_time=maxima[index]
                    if complete[index]
                    else None,
                    missing_ordinal_count=missing_ordinals,
                    excess_ordinal_count=excess_ordinals,
                    ordinal_concatenation_matches=ordered_matches,
                    target_covered_bytes=covered,
                    target_missing_bytes=assembly.declared_size - covered,
                    target_overlap_bytes=overlap,
                    target_multiplicity_excess_bytes=multiplicity_excess,
                    target_excess_bytes=outside,
                    hash_label_equation=equation,
                    hash_labels_match=labels_match,
                    available_at_decision=visible,
                    usable_declaration_at_decision=usable,
                    passed=index not in invalid,
                )
            )
    return Output(
        passed=not issues,
        object_count=len(request.objects),
        assembly_count=len(request.assemblies),
        part_count=len(request.parts),
        usable_declaration_count=usable_count,
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        assemblies=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.assemblies),
    )


OPERATION = Operation(
    id="skills.audit_artifact_assembly",
    kind="skill",
    description="Audit declared byte-part concatenation, source extents/version bindings, target coverage and publication clocks without claiming byte or hash authenticity.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
