"""Verify an ordered audit-tree inclusion path against a caller-pinned root/size.

Conventions match quant_fund.audit.merkle and RFC 6962 section 2.1.1: SHA256
leaf prefix 00, ordered internal-node prefix 01, and bottom-up minimal siblings.
This is not the sorted-leaf proofcore tree. A recursive split plan determines
the exact sibling orientations and required path length without tree allocation.
Missing or extra siblings fail before hashing the proof. Empty trees have no
inclusion proof; a one-leaf tree requires an empty path.

The caller supplies either exact leaf-preimage bytes as lowercase hex or an
already prefixed SHA256 leaf digest. Digest-only mode does not validate a leaf
preimage. Roots and sizes are required caller assertions, never obtained from
the proof. No signature, checkpoint identity, log completeness, inclusion of
other leaves, leaf payload semantics or research claim is certified.

Limits: sizes/indices through signed 63-bit integers, 64 siblings, 65536-byte
leaf preimages. Proof work is logarithmic in the declared size.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Digest = Annotated[str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
Outcome = Literal[
    "verified",
    "empty_tree",
    "index_out_of_range",
    "proof_too_short",
    "proof_too_long",
    "root_mismatch",
]


class Input(InputModel):
    expected_tree_size: int = Field(strict=True, ge=0, le=(1 << 63) - 1)
    expected_root: Digest
    leaf_index: int = Field(strict=True, ge=0, le=(1 << 63) - 1)
    leaf_encoding: Literal["hex_preimage", "sha256_leaf_digest"]
    leaf_value: str = Field(strict=True, max_length=131_072, pattern=r"^[0-9a-f]*$")
    proof: list[Digest] = Field(max_length=64)

    @model_validator(mode="after")
    def leaf_bytes(self) -> Self:
        if len(self.leaf_value) % 2:
            raise ValueError("leaf hex must encode whole bytes")
        if self.leaf_encoding == "sha256_leaf_digest" and len(self.leaf_value) != 64:
            raise ValueError("a SHA256 leaf digest requires exactly 32 bytes")
        return self


class Output(OutputModel):
    verified: bool
    outcome: Outcome
    expected_tree_size: int
    expected_root: str
    leaf_index: int
    leaf_digest: str
    leaf_preimage_hashed: bool
    leaf_preimage_bytes: int | None
    expected_proof_nodes: int | None
    supplied_proof_nodes: int
    consumed_proof_nodes: int
    computed_root: str | None
    root_matches: bool | None
    conventions: Literal["quant_fund.audit.ordered_rfc6962_sha256"] = (
        "quant_fund.audit.ordered_rfc6962_sha256"
    )
    anchor_scope: Literal["caller_supplied_root_and_size"] = "caller_supplied_root_and_size"
    signature_verified: Literal[False] = False
    ledger_completeness_verified: Literal[False] = False
    leaf_payload_semantics_verified: Literal[False] = False
    research_certified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    supplied_leaf = bytes.fromhex(request.leaf_value)
    hashed_preimage = request.leaf_encoding == "hex_preimage"
    leaf = hashlib.sha256(b"\x00" + supplied_leaf).digest() if hashed_preimage else supplied_leaf
    outcome: Outcome
    required: int | None = None
    computed: bytes | None = None
    root_matches: bool | None = None
    consumed = 0
    if request.expected_tree_size == 0:
        outcome = "empty_tree"
    elif request.leaf_index >= request.expected_tree_size:
        outcome = "index_out_of_range"
    else:
        # Record root-to-leaf orientations using the largest strict power-of-two
        # split, then reverse them to consume the bottom-up path exactly once.
        left_siblings: list[bool] = []
        width, position = request.expected_tree_size, request.leaf_index
        while width > 1:
            split = 1 << ((width - 1).bit_length() - 1)
            sibling_is_left = position >= split
            left_siblings.append(sibling_is_left)
            if sibling_is_left:
                position -= split
                width -= split
            else:
                width = split
        required = len(left_siblings)
        if len(request.proof) < required:
            outcome = "proof_too_short"
        elif len(request.proof) > required:
            outcome = "proof_too_long"
        else:
            computed = leaf
            for sibling_is_left, sibling_hex in zip(
                reversed(left_siblings), request.proof, strict=True
            ):
                sibling = bytes.fromhex(sibling_hex)
                left, right = (sibling, computed) if sibling_is_left else (computed, sibling)
                computed = hashlib.sha256(b"\x01" + left + right).digest()
                consumed += 1
            root_matches = hmac.compare_digest(computed, bytes.fromhex(request.expected_root))
            outcome = "verified" if root_matches else "root_mismatch"
    return Output(
        verified=outcome == "verified",
        outcome=outcome,
        expected_tree_size=request.expected_tree_size,
        expected_root=request.expected_root,
        leaf_index=request.leaf_index,
        leaf_digest=leaf.hex(),
        leaf_preimage_hashed=hashed_preimage,
        leaf_preimage_bytes=len(supplied_leaf) if hashed_preimage else None,
        expected_proof_nodes=required,
        supplied_proof_nodes=len(request.proof),
        consumed_proof_nodes=consumed,
        computed_root=computed.hex() if computed is not None else None,
        root_matches=root_matches,
    )


OPERATION = Operation(
    id="skills.verify_merkle_inclusion",
    kind="skill",
    description="Verify a minimal ordered SHA256 audit inclusion path against caller-pinned tree size/root, with exact sibling consumption and explicit leaf-preimage versus digest semantics.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
