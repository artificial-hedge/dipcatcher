"""Verify one existing quant_fund.audit v1 Ed25519 checkpoint.

The signed preimage is the repository's ASCII-escaped compact sorted JSON of
v/scheme/key_id/tree_size/merkle_root/timestamp_utc. There is no signed domain
field in this legacy format. expected_domain explicitly selects that convention;
it cannot establish cryptographic domain separation. Caller-supplied identity
labels bind full trusted public-key bytes locally, not an external trust store.

Exact expected tree-size/root anchors and a selected trusted identity are
required. Signature validity, key-hint agreement and anchor agreement are
reported separately. No ledger entries, consistency proof, timestamp freshness,
research claim, private key, release HMAC or Sigstore signature is verified.
Unsigned public-key/Sigstore metadata is returned only as a declaration.

Accepts one .json checkpoint or a selected zero-based .jsonl record: 2 MB source,
1000 JSONL records, 16384 bytes per record, 16 supplied trusted identities.
Blank JSONL records are rejected. All records receive strict JSON syntax checks;
only the selected record receives the checkpoint/schema/signature checks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Hex32 = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
Identity = Annotated[str, Field(min_length=1, max_length=128)]


class TrustedKey(InputModel):
    identity: Identity
    public_key: Hex32


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    record_index: int = Field(default=0, strict=True, ge=0, le=999)
    trusted_keys: list[TrustedKey] = Field(min_length=1, max_length=16)
    expected_identity: Identity
    expected_domain: Literal["quant_fund.audit.checkpoint.v1"]
    expected_tree_size: int = Field(strict=True, ge=1, le=1_000_000_000_000)
    expected_merkle_root: Hex32

    @model_validator(mode="after")
    def unambiguous_keys(self) -> Input:
        if len({key.identity for key in self.trusted_keys}) != len(self.trusted_keys):
            raise ValueError("supplied trusted key identities must be unique")
        if len({key.public_key for key in self.trusted_keys}) != len(self.trusted_keys):
            raise ValueError("one public key cannot have multiple supplied identities")
        if self.expected_identity not in {key.identity for key in self.trusted_keys}:
            raise ValueError("expected_identity must identify one supplied trusted key")
        return self


class Checkpoint(InputModel):
    v: int = Field(strict=True, ge=1, le=1)
    scheme: Literal["ed25519"]
    key_id: str = Field(min_length=16, max_length=16, pattern=r"^[0-9a-f]{16}$")
    tree_size: int = Field(strict=True, ge=1, le=1_000_000_000_000)
    merkle_root: Hex32
    timestamp_utc: str = Field(min_length=1, max_length=128)
    signature: str = Field(min_length=128, max_length=128, pattern=r"^[0-9a-f]{128}$")
    public_key: Hex32
    sigstore_bundle_json: str | None = Field(default=None, max_length=4096)
    sigstore_identity: str | None = Field(default=None, max_length=512)
    sigstore_issuer: str | None = Field(default=None, max_length=512)

    @field_validator("timestamp_utc")
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if "T" not in value or parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("checkpoint timestamp_utc must be an aware UTC ISO timestamp")
        return value  # Preserve the exact signed string, including its spelling.


class UnsignedMetadata(OutputModel):
    public_key: str
    sigstore_bundle_json: str | None
    sigstore_identity: str | None
    sigstore_issuer: str | None


class Output(OutputModel):
    record_index: int
    record_count: int
    declared_tree_size: int
    declared_merkle_root: str
    declared_timestamp_utc: str
    declared_key_id: str
    expected_identity: str
    expected_domain: str
    supplied_key_sha256: str
    signature_valid_under_supplied_key: bool
    signature_valid_under_embedded_key: bool
    embedded_key_matches_supplied_key: bool
    key_id_matches_supplied_key: bool
    expected_tree_size_matches: bool
    expected_merkle_root_matches: bool
    checkpoint_matches_supplied_expectations: bool
    unsigned_metadata: UnsignedMetadata
    canonical_record_spelling: bool
    signed_payload_sha256: str
    signed_payload_bytes: int
    domain_cryptographically_bound: Literal[False] = False
    trust_scope: Literal["agreement_with_caller_supplied_key_and_anchors"] = (
        "agreement_with_caller_supplied_key_and_anchors"
    )
    ledger_completeness_verified: Literal[False] = False
    ledger_consistency_verified: Literal[False] = False
    timestamp_freshness_checked: Literal[False] = False
    research_certified: Literal[False] = False
    source_bytes: int
    source_sha256: str


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("checkpoint JSON contains a duplicate key")
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value) > 13:
        raise ValueError("checkpoint JSON integer exceeds 13 characters")
    return int(value)


def _reject_number(value: str) -> None:
    raise ValueError(
        "checkpoint JSON does not support floating-point values or nonfinite constants"
    )


def _parse(record: bytes) -> dict[str, Any]:
    if not record or len(record) > 16_384:
        raise ValueError("checkpoint JSON records must contain 1 through 16384 bytes")
    try:
        result = json.loads(
            record.decode("utf-8"),
            object_pairs_hook=_object,
            parse_int=_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid strict UTF-8 checkpoint JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("checkpoint JSON record must be an object")
    # The existing checkpoint format is flat. Check every JSONL record before
    # selection so nested records cannot acquire checkpoint semantics by accident.
    if any(isinstance(value, (list, dict)) for value in result.values()):
        raise ValueError("checkpoint records must be flat JSON objects")
    try:
        for key, value in result.items():
            key.encode("utf-8")
            if isinstance(value, str):
                value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("checkpoint strings must not contain unpaired Unicode surrogates") from exc
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _verify(public_key: bytes, signature: bytes, payload: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError):
        return False
    return True


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".json", ".jsonl"), max_bytes=2_000_000)
    if request.path.lower().endswith(".jsonl"):
        if content.count(b"\n") > 1000:
            raise ValueError("checkpoint JSONL exceeds 1000 records")
        records = content.split(b"\n")
        if records[-1:] == [b""]:
            records.pop()
        if not 1 <= len(records) <= 1000:
            raise ValueError("checkpoint JSONL must contain 1 through 1000 records")
        records = [record.removesuffix(b"\r") for record in records]
    else:
        records = [content]
    if request.record_index >= len(records):
        raise ValueError("checkpoint record_index is outside the source")
    selected: dict[str, Any] = {}
    for index, record in enumerate(records):
        parsed = _parse(record)
        if index == request.record_index:
            selected = parsed
    checkpoint = Checkpoint.model_validate(selected)
    payload = _canonical(
        {
            key: selected[key]
            for key in ("v", "scheme", "key_id", "tree_size", "merkle_root", "timestamp_utc")
        }
    )
    trusted = next(key for key in request.trusted_keys if key.identity == request.expected_identity)
    trusted_bytes = bytes.fromhex(trusted.public_key)
    embedded_bytes = bytes.fromhex(checkpoint.public_key)
    signature = bytes.fromhex(checkpoint.signature)
    key_hash = hashlib.sha256(trusted_bytes).hexdigest()
    supplied_valid = _verify(trusted_bytes, signature, payload)
    embedded_valid = (
        supplied_valid
        if embedded_bytes == trusted_bytes
        else _verify(embedded_bytes, signature, payload)
    )
    key_matches = embedded_bytes == trusted_bytes
    id_matches = checkpoint.key_id == key_hash[:16]
    size_matches = checkpoint.tree_size == request.expected_tree_size
    root_matches = checkpoint.merkle_root == request.expected_merkle_root
    return Output(
        record_index=request.record_index,
        record_count=len(records),
        declared_tree_size=checkpoint.tree_size,
        declared_merkle_root=checkpoint.merkle_root,
        declared_timestamp_utc=checkpoint.timestamp_utc,
        declared_key_id=checkpoint.key_id,
        expected_identity=request.expected_identity,
        expected_domain=request.expected_domain,
        supplied_key_sha256=key_hash,
        signature_valid_under_supplied_key=supplied_valid,
        signature_valid_under_embedded_key=embedded_valid,
        embedded_key_matches_supplied_key=key_matches,
        key_id_matches_supplied_key=id_matches,
        expected_tree_size_matches=size_matches,
        expected_merkle_root_matches=root_matches,
        checkpoint_matches_supplied_expectations=all(
            (supplied_valid, key_matches, id_matches, size_matches, root_matches)
        ),
        unsigned_metadata=UnsignedMetadata(
            public_key=checkpoint.public_key,
            sigstore_bundle_json=checkpoint.sigstore_bundle_json,
            sigstore_identity=checkpoint.sigstore_identity,
            sigstore_issuer=checkpoint.sigstore_issuer,
        ),
        canonical_record_spelling=records[request.record_index] == _canonical(selected),
        signed_payload_sha256=hashlib.sha256(payload).hexdigest(),
        signed_payload_bytes=len(payload),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.verify_signed_checkpoint",
    kind="plugin",
    description="Verify an audit-v1 Ed25519 checkpoint against caller-pinned identity/key bytes and exact size/root anchors; separately report unsigned metadata and the format's absent domain binding, without ledger or research certification.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
