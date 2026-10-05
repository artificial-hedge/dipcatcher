"""Verify a strict manifest's byte declarations through contained workspace reads.

Manifest format: {"version": 1, "artifacts": [{"path": "relative/path", "size":
123, "sha256": "64 lowercase hex characters"}, ...]}. Paths are relative to
the workspace, not the manifest directory. Portable POSIX spellings and unique
casefolded paths are required; hard-link identity is not inferred. Extra fields,
duplicate JSON keys, floating-point numbers and nonfinite constants are rejected.

Limits: 256 KB UTF-8 manifest, 200 artifacts, 16 MB per artifact, 64 MB declared
aggregate and 64000001 actual aggregate read bytes (64 MB plus one probe byte).
One byte is reserved before each file read for the contained reader's over-limit
probe. Every declaration
receives an observation, including refused/missing/exhausted reads. Failed reads
have no whole-file hash. No artifact is deserialized or executed.

An empty manifest supplies no evidence. Completeness means all listed sizes and
SHA-256 values matched the bytes read; it says nothing about unlisted artifacts,
provenance, research correctness or authenticity. Reads are sequential and do
not constitute an atomic snapshot. Manifest and artifact hashes are byte-exact;
binding hashes use sorted-path compact UTF-8 JSON with sorted object keys and
path/size/sha256 entries. Unread sizes/hashes are null; matching nonempty bundles
have equal binding hashes. Hash equality for two empty lists supplies no evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)


class Artifact(InputModel):
    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(strict=True, ge=0, le=16_000_000)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def portable_path(cls, value: str) -> str:
        value.encode("utf-8")
        components = value.split("/")
        if (
            value.startswith("/")
            or any(component in ("", ".", "..") for component in components)
            or any(
                character in '\\<>:"|?*' or ord(character) < 32 or ord(character) == 127
                for character in value
            )
            or any(
                component.endswith((".", " ")) or PureWindowsPath(component).is_reserved()
                for component in components
            )
        ):
            raise ValueError(
                "artifact paths require portable relative POSIX spelling without ambiguous components"
            )
        return value


class Manifest(InputModel):
    version: int = Field(strict=True, ge=1, le=1)
    artifacts: list[Artifact] = Field(max_length=200)

    @model_validator(mode="after")
    def bounded_unique_artifacts(self) -> Manifest:
        if len({artifact.path.casefold() for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("artifact paths must be unique, including casefolded spellings")
        if sum(artifact.size for artifact in self.artifacts) > 64_000_000:
            raise ValueError("artifact declared sizes exceed the 64 MB aggregate bound")
        return self


Status = Literal[
    "match",
    "size_mismatch",
    "sha256_mismatch",
    "size_and_sha256_mismatch",
    "missing",
    "read_refused",
    "read_error",
    "budget_exhausted",
]


class Observation(OutputModel):
    index: int
    path: str
    declared_size: int
    declared_sha256: str
    status: Status
    observed_size: int | None
    observed_sha256: str | None
    size_matches: bool | None
    sha256_matches: bool | None
    bytes_consumed: int
    error_class: str | None


class Output(OutputModel):
    manifest_version: int
    artifact_count: int
    declared_total_bytes: int
    observations: list[Observation]
    status_counts: dict[str, int]
    fully_read_count: int
    matched_count: int
    complete: bool
    all_declarations_match: bool | None
    manifest_has_artifacts: bool
    aggregate_bytes_consumed: int
    read_budget_bytes: Literal[64_000_001] = 64_000_001
    expected_bindings_sha256: str
    observed_bindings_sha256: str
    binding_hash_convention: Literal["sorted_path_compact_sorted_key_utf8_json"] = (
        "sorted_path_compact_sorted_key_utf8_json"
    )
    read_consistency: Literal["individual_reads_not_atomic_bundle_snapshot"] = (
        "individual_reads_not_atomic_bundle_snapshot"
    )
    unlisted_artifacts_checked: Literal[False] = False
    research_certified: Literal[False] = False
    source_bytes: int
    source_sha256: str


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key in artifact manifest")
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value) > 10:
        raise ValueError("manifest integer spelling exceeds 10 characters")
    return int(value)


def _no_float(value: str) -> None:
    raise ValueError(
        "artifact manifest accepts integer sizes/version and no floating-point constants"
    )


def _manifest(content: bytes) -> Manifest:
    try:
        decoded = content.decode("utf-8")
        # Bound structural nesting before handing the complete text to JSON's
        # recursive parser. Delimiters in strings do not affect this counter.
        depth = 0
        quoted = escaped = False
        for character in decoded:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "[{":
                depth += 1
                if depth > 8:
                    raise ValueError("artifact manifest exceeds eight JSON nesting levels")
            elif character in "]}":
                depth -= 1
        document = json.loads(
            decoded,
            object_pairs_hook=_object,
            parse_int=_integer,
            parse_float=_no_float,
            parse_constant=_no_float,
        )
        return Manifest.model_validate(document)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid strict UTF-8 artifact manifest: {exc}") from exc


def _digest(value: object) -> str:
    content = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".json",), max_bytes=256_000)
    manifest = _manifest(content)
    observations: list[Observation] = []
    consumed = fully_read = matched = 0
    status_counts: dict[str, int] = {}
    for index, artifact in enumerate(manifest.artifacts):
        remaining = 64_000_001 - consumed
        status: Status = "budget_exhausted"
        observed_size: int | None = None
        observed_hash: str | None = None
        size_matches: bool | None = None
        hash_matches: bool | None = None
        bytes_consumed = 0
        error_class: str | None = None
        if remaining > 0:
            try:
                # Reserve one byte for a file that grows past its checked size.
                # The reader's single over-limit probe therefore remains inside
                # the aggregate I/O budget even when this read is refused.
                limit = min(16_000_000, remaining - 1)
                suffix = PurePosixPath(artifact.path).suffix.lower()
                with context.open_binary(
                    artifact.path, suffixes=(suffix,), max_bytes=limit
                ) as stream:
                    try:
                        while stream.read(65_536):
                            pass
                        observed_size = stream.bytes_read
                        observed_hash = stream.source_sha256
                    finally:
                        bytes_consumed = stream.bytes_read
                size_matches = observed_size == artifact.size
                hash_matches = observed_hash == artifact.sha256
                fully_read += 1
                if size_matches and hash_matches:
                    status = "match"
                    matched += 1
                elif not size_matches and not hash_matches:
                    status = "size_and_sha256_mismatch"
                elif not size_matches:
                    status = "size_mismatch"
                else:
                    status = "sha256_mismatch"
            except FileNotFoundError as exc:
                status, error_class = "missing", type(exc).__name__
                observed_size = observed_hash = None
            except ValueError as exc:
                status, error_class = "read_refused", type(exc).__name__
                observed_size = observed_hash = None
            except OSError as exc:
                status, error_class = "read_error", type(exc).__name__
                observed_size = observed_hash = None
        consumed += bytes_consumed
        observations.append(
            Observation(
                index=index,
                path=artifact.path,
                declared_size=artifact.size,
                declared_sha256=artifact.sha256,
                status=status,
                observed_size=observed_size,
                observed_sha256=observed_hash,
                size_matches=size_matches,
                sha256_matches=hash_matches,
                bytes_consumed=bytes_consumed,
                error_class=error_class,
            )
        )
        status_counts[status] = status_counts.get(status, 0) + 1
    expected_bindings = [
        artifact.model_dump() for artifact in sorted(manifest.artifacts, key=lambda item: item.path)
    ]
    observed_bindings = [
        {
            "path": item.path,
            "size": item.observed_size,
            "sha256": item.observed_sha256,
        }
        for item in sorted(observations, key=lambda item: item.path)
    ]
    has_artifacts = bool(manifest.artifacts)
    complete = has_artifacts and matched == len(manifest.artifacts)
    return Output(
        manifest_version=manifest.version,
        artifact_count=len(manifest.artifacts),
        declared_total_bytes=sum(artifact.size for artifact in manifest.artifacts),
        observations=observations,
        status_counts=status_counts,
        fully_read_count=fully_read,
        matched_count=matched,
        complete=complete,
        all_declarations_match=complete
        if has_artifacts and fully_read == len(manifest.artifacts)
        else None,
        manifest_has_artifacts=has_artifacts,
        aggregate_bytes_consumed=consumed,
        expected_bindings_sha256=_digest(expected_bindings),
        observed_bindings_sha256=_digest(observed_bindings),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.inspect_artifact_bundle",
    kind="plugin",
    description="Verify every strict JSON manifest artifact's declared size and SHA-256 through bounded contained streaming reads; report missing/refused/mismatched files and completeness without research certification or atomic-snapshot claims.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
