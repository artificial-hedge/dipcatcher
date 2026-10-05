"""Inspect receipt declarations and hash bindings without certifying research.

Selects receipt-body or hedge-lab path-excluding preimages explicitly, using
UTF-8 canonical JSON (also the legacy ASCII convention for receipt bodies).
This is not verify-research or a lane verifier; declared code/data paths are
never opened. Value selection uses RFC 6901 JSON Pointer strings.
"""

import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    seal_preimage: Literal["receipt_body", "hedge_lab_body"] = "receipt_body"
    pointers: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("pointers")
    @classmethod
    def valid_pointers(cls, pointers: list[str]) -> list[str]:
        if len(set(pointers)) != len(pointers):
            raise ValueError("pointers must be unique")
        for pointer in pointers:
            if len(pointer) > 4096 or (pointer and not pointer.startswith("/")):
                raise ValueError("use bounded RFC 6901 pointer strings, not URI fragments")
            if re.search(r"~(?:[^01]|$)", pointer):
                raise ValueError("JSON Pointer permits only ~0 and ~1 escapes")
            if pointer.count("/") > 64:
                raise ValueError("pointer depth exceeds 64")
        return pointers


class DigestCheck(OutputModel):
    status: Literal["match", "mismatch", "missing_or_invalid", "not_applicable"]
    declared_sha256: str | None
    calculated_sha256: str | None
    calculated_convention: Literal["canonical_json"] = "canonical_json"
    matched_conventions: list[str]


class Selection(OutputModel):
    pointer: str
    status: Literal["found", "missing", "value_too_large"]
    value: JsonValue | None
    canonical_value_sha256: str | None
    canonical_value_bytes: int | None


class Output(OutputModel):
    inspection_scope: Literal["declarations_and_digest_agreement_only"] = (
        "declarations_and_digest_agreement_only"
    )
    research_eligibility_verified: Literal[False] = False
    source_sha256: str
    source_bytes: int
    declared_schema: str | None
    declared_kind: str | None
    declared_data_label: str | None
    declared_verdict: str | None
    declared_live_pnl_claim: bool | None
    selected_seal_preimage: str
    seal_excluded_top_level_fields: list[str]
    receipt_seal: DigestCheck
    environment_fingerprint: DigestCheck
    code_map_fingerprint: DigestCheck
    declared_code_file_count: int | None
    payload_present: bool
    selections: list[Selection]


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("receipt numbers must be finite")
    if value == 0 and any(char in "123456789" for char in text.lower().partition("e")[0]):
        raise ValueError("nonzero receipt number underflows to zero")
    return value


def _reject_constant(text: str) -> None:
    raise ValueError(f"nonfinite JSON constant {text!r}")


def _canonical(value: object, *, ascii_only: bool = False) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_only, allow_nan=False
    ).encode("utf-8")


def _check(declared: object, body: object, *, legacy: bool = False) -> DigestCheck:
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if not isinstance(declared, str) or not _DIGEST.fullmatch(declared):
        return DigestCheck(
            status="missing_or_invalid",
            declared_sha256=None,
            calculated_sha256=actual,
            matched_conventions=[],
        )
    matches = ["canonical_json"] if actual == declared else []
    if legacy and hashlib.sha256(_canonical(body, ascii_only=True)).hexdigest() == declared:
        matches.append("strict_json")
    return DigestCheck(
        status="match" if matches else "mismatch",
        declared_sha256=declared,
        calculated_sha256=actual,
        matched_conventions=matches,
    )


def _unavailable(status: Literal["missing_or_invalid", "not_applicable"]) -> DigestCheck:
    return DigestCheck(
        status=status, declared_sha256=None, calculated_sha256=None, matched_conventions=[]
    )


def _text(document: dict[str, Any], key: str) -> str | None:
    value = document.get(key)
    return value if isinstance(value, str) and len(value) <= 256 else None


def _select(document: dict[str, Any], pointer: str) -> Selection:
    value: Any = document
    for encoded in pointer.split("/")[1:] if pointer else []:
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif (
            isinstance(value, list)
            and re.fullmatch(r"0|[1-9][0-9]*", token)
            and len(token) <= 10
            and int(token) < len(value)
        ):
            value = value[int(token)]
        else:
            return Selection(
                pointer=pointer,
                status="missing",
                value=None,
                canonical_value_sha256=None,
                canonical_value_bytes=None,
            )
    encoded_value = _canonical(value)
    fits = len(encoded_value) <= 16_000
    return Selection(
        pointer=pointer,
        status="found" if fits else "value_too_large",
        value=value if fits else None,
        canonical_value_sha256=hashlib.sha256(encoded_value).hexdigest(),
        canonical_value_bytes=len(encoded_value),
    )


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".json",), max_bytes=2_000_000)
    try:
        document = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object,
            parse_float=_number,
            parse_constant=_reject_constant,
        )
        if not isinstance(document, dict):
            raise ValueError("receipt must be a JSON object")
        pending: list[tuple[object, int]] = [(document, 0)]
        nodes = 0
        while pending:
            value, depth = pending.pop()
            nodes += 1
            if depth > 64 or nodes > 100_000:
                raise ValueError("receipt exceeds 64 levels or 100000 value nodes")
            if isinstance(value, str):
                value.encode("utf-8")
            elif isinstance(value, dict):
                for key, child in value.items():
                    key.encode("utf-8")
                    pending.append((child, depth + 1))
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"receipt JSON is invalid or unsupported: {exc}") from exc
    excluded = ["receipt_sha256"]
    if request.seal_preimage == "hedge_lab_body":
        excluded.extend(("receipt_path", "artifact_path", "metadata_path"))
    body = {key: value for key, value in document.items() if key not in excluded}
    seal = (
        _check(document.get("receipt_sha256"), body, legacy=request.seal_preimage == "receipt_body")
        if "receipt_sha256" in document
        else _unavailable("not_applicable")
    )
    environment = document.get("environment")
    environment_check = _unavailable("not_applicable")
    if "environment" in document:
        environment_check = (
            _check(
                environment.get("fingerprint_sha256"),
                {key: value for key, value in environment.items() if key != "fingerprint_sha256"},
            )
            if isinstance(environment, dict)
            else _unavailable("missing_or_invalid")
        )
    files = document.get("code_files")
    code_check = _unavailable("not_applicable")
    code_count = len(files) if isinstance(files, dict) else None
    if "code_files" in document or "code_sha256" in document:
        code_check = (
            _check(document.get("code_sha256"), files)
            if isinstance(files, dict)
            and files
            and all(
                isinstance(name, str)
                and 0 < len(name) <= 4096
                and isinstance(digest, str)
                and _DIGEST.fullmatch(digest)
                for name, digest in files.items()
            )
            else _unavailable("missing_or_invalid")
        )
    claim = document.get("live_pnl_claim")
    return Output(
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
        declared_schema=_text(document, "schema"),
        declared_kind=_text(document, "kind"),
        declared_data_label=_text(document, "data_label"),
        declared_verdict=_text(document, "verdict"),
        declared_live_pnl_claim=claim if isinstance(claim, bool) else None,
        selected_seal_preimage=request.seal_preimage,
        seal_excluded_top_level_fields=excluded,
        receipt_seal=seal,
        environment_fingerprint=environment_check,
        code_map_fingerprint=code_check,
        declared_code_file_count=code_count,
        payload_present=isinstance(document.get("payload"), dict),
        selections=[_select(document, pointer) for pointer in request.pointers],
    )


OPERATION = Operation(
    id="plugins.inspect_research_receipt",
    kind="plugin",
    description=(
        "Inspect bounded receipt JSON declarations, seal/environment/code-map digest "
        "agreement, and selected RFC6901 values. Rejects duplicate keys and invalid numbers; "
        "does not certify receipt schema, research eligibility, code files or data sources."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
