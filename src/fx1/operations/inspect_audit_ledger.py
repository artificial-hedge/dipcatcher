"""Stream and diagnose the existing dipcatcher v1 audit-entry hash chain.

Entry bodies use sorted compact ASCII-escaped JSON, excluding entry_hash.
Physical line positions determine expected indices, including malformed lines.
Checks describe the supplied file or segment only. They do not verify signed
checkpoints, Merkle proofs, source truth, research eligibility, or completeness
beyond caller-supplied end anchors. No ledger or lock file is created.
"""

import hashlib
import io
import json
import math
import re
from collections import Counter
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
_KINDS = frozenset({"research_run", "paper_decision", "simulated_order", "fill", "risk_decision"})
_FIELDS = {"v", "index", "kind", "recorded_at", "payload", "prev_hash", "entry_hash"}


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    first_index: int = Field(default=0, strict=True, ge=0, le=10**12)
    expected_previous_hash: Digest = "0" * 64
    expected_final_hash: Digest | None = None
    expected_entry_count: int | None = Field(default=None, strict=True, ge=0, le=100_000)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=50, strict=True, ge=1, le=100)
    max_diagnostics: int = Field(default=50, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def genesis_anchor(self) -> Self:
        if self.first_index == 0 and self.expected_previous_hash != "0" * 64:
            raise ValueError("v1 ledger index zero requires the all-zero genesis hash")
        return self


class Finding(OutputModel):
    physical_line: int | None
    code: str


class EntrySummary(OutputModel):
    physical_line: int
    declared_index: int
    expected_index: int
    kind: str
    recorded_at: str
    payload_field_count: int
    declared_entry_hash: str
    calculated_entry_hash: str
    entry_hash_matches: bool
    previous_link_matches: bool | None
    canonical_line: bool


class Output(OutputModel):
    verification_scope: Literal["supplied_v1_entry_chain_only"] = "supplied_v1_entry_chain_only"
    checkpoint_signatures_verified: Literal[False] = False
    research_eligibility_verified: Literal[False] = False
    assessment: Literal["consistent", "inconsistent", "empty"]
    chain_consistent: bool | None
    first_index: int
    physical_line_count: int
    structurally_parsed_entries: int
    first_declared_hash: str | None
    last_declared_hash: str | None
    expected_previous_hash: str
    expected_final_hash_matches: bool | None
    expected_entry_count_matches: bool | None
    kind_counts: dict[str, int]
    issue_counts: dict[str, int]
    issue_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int
    entries: list[EntrySummary]
    offset: int
    next_offset: int | None
    source_sha256: str
    source_bytes: int


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key")
        result[name] = value
    return result


def _number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if value == 0.0 and any(char in "123456789" for char in text.lower().partition("e")[0]):
        raise ValueError("nonzero JSON number underflows")
    return value


def _reject_constant(text: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _parse(line: bytes) -> dict[str, Any]:
    parsed = json.loads(
        line.decode("utf-8"),
        object_pairs_hook=_object,
        parse_float=_number,
        parse_constant=_reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("entry must be an object")
    pending: list[tuple[object, int]] = [(parsed, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if depth > 64 or count > 100_000:
            raise ValueError("entry exceeds nesting or node budget")
        if isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, dict):
            for key, item in value.items():
                key.encode("utf-8")
                pending.append((item, depth + 1))
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return parsed


def _entry_shape(entry: dict[str, Any]) -> bool:
    return (
        set(entry) == _FIELDS
        and type(entry.get("v")) is int
        and entry["v"] == 1
        and type(entry.get("index")) is int
        and 0 <= entry["index"] <= 10**12 + 100_000
        and isinstance(entry.get("kind"), str)
        and entry["kind"] in _KINDS
        and isinstance(entry.get("recorded_at"), str)
        and 0 < len(entry["recorded_at"]) <= 128
        and isinstance(entry.get("payload"), dict)
        and all(
            isinstance(entry.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", entry[key])
            for key in ("prev_hash", "entry_hash")
        )
    )


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    findings: list[Finding] = []
    page: list[EntrySummary] = []
    line_count = parsed_count = 0
    previous: str | None = request.expected_previous_hash
    first_hash: str | None = None
    last_hash: str | None = None

    def flag(code: str, line_number: int | None) -> None:
        issues[code] += 1
        if len(findings) < request.max_diagnostics:
            findings.append(Finding(physical_line=line_number, code=code))

    with context.open_binary(request.path, suffixes=(".jsonl",), max_bytes=64_000_000) as source:
        buffered = io.BufferedReader(source)
        with buffered as reader:
            while raw := reader.readline(1_000_001):
                line_count += 1
                if len(raw) > 1_000_000 or line_count > 100_000:
                    raise ValueError("ledger exceeds a 1000000-byte line or 100000-line budget")
                terminated = raw.endswith(b"\n")
                if not terminated:
                    flag("missing_final_newline", line_count)
                line = raw[:-1] if terminated else raw
                if b"\r" in line:
                    flag("carriage_return", line_count)
                if not line:
                    flag("blank_line", line_count)
                    previous = None
                    last_hash = None
                    continue
                try:
                    entry = _parse(line)
                except (ValueError, UnicodeError, RecursionError):
                    flag("invalid_json", line_count)
                    previous = None
                    last_hash = None
                    continue
                canonical = _canonical(entry) == line
                if not canonical:
                    flag("noncanonical_line", line_count)
                if not _entry_shape(entry):
                    flag("invalid_entry_shape", line_count)
                    previous = None
                    last_hash = None
                    continue
                parsed_count += 1
                expected_index = request.first_index + line_count - 1
                if entry["index"] != expected_index:
                    flag("index_mismatch", line_count)
                body = {key: value for key, value in entry.items() if key != "entry_hash"}
                actual = hashlib.sha256(_canonical(body)).hexdigest()
                hash_matches = actual == entry["entry_hash"]
                if not hash_matches:
                    flag("entry_hash_mismatch", line_count)
                link_matches = entry["prev_hash"] == previous if previous is not None else None
                if link_matches is False:
                    flag("previous_hash_mismatch", line_count)
                elif link_matches is None:
                    flag("previous_link_unassessed", line_count)
                previous = entry["entry_hash"]
                last_hash = previous
                if line_count == 1:
                    first_hash = previous
                kinds[entry["kind"]] += 1
                if request.offset <= line_count - 1 < request.offset + request.limit:
                    page.append(
                        EntrySummary(
                            physical_line=line_count,
                            declared_index=entry["index"],
                            expected_index=expected_index,
                            kind=entry["kind"],
                            recorded_at=entry["recorded_at"],
                            payload_field_count=len(entry["payload"]),
                            declared_entry_hash=entry["entry_hash"],
                            calculated_entry_hash=actual,
                            entry_hash_matches=hash_matches,
                            previous_link_matches=link_matches,
                            canonical_line=canonical,
                        )
                    )
            source_hash, source_bytes = source.source_sha256, source.bytes_read
    # Empty segments preserve the incoming tip. Malformed final records do not.
    tip = request.expected_previous_hash if line_count == 0 else last_hash
    tail_matches = (
        tip == request.expected_final_hash if request.expected_final_hash is not None else None
    )
    count_matches = (
        parsed_count == request.expected_entry_count
        if request.expected_entry_count is not None
        else None
    )
    if tail_matches is False:
        flag("expected_final_hash_mismatch", None)
    if count_matches is False:
        flag("expected_entry_count_mismatch", None)
    issue_count = sum(issues.values())
    stop = min(line_count, request.offset + request.limit)
    return Output(
        assessment="inconsistent" if issues else "consistent" if line_count else "empty",
        chain_consistent=False if issues else True if line_count else None,
        first_index=request.first_index,
        physical_line_count=line_count,
        structurally_parsed_entries=parsed_count,
        first_declared_hash=first_hash,
        last_declared_hash=last_hash,
        expected_previous_hash=request.expected_previous_hash,
        expected_final_hash_matches=tail_matches,
        expected_entry_count_matches=count_matches,
        kind_counts=dict(sorted(kinds.items())),
        issue_counts=dict(sorted(issues.items())),
        issue_count=issue_count,
        diagnostics=findings,
        omitted_diagnostics=issue_count - len(findings),
        entries=page,
        offset=request.offset,
        next_offset=stop if stop < line_count else None,
        source_sha256=source_hash,
        source_bytes=source_bytes,
    )


OPERATION = Operation(
    id="plugins.inspect_audit_ledger",
    kind="plugin",
    description=(
        "Stream a v1 audit ledger or explicitly anchored segment, checking canonical lines, "
        "entry digests, indices and previous-hash links with bounded diagnostics. Optional "
        "end anchors detect truncation relative to those anchors; no signature or research verification."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
