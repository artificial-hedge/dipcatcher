"""Verify an audit-tree prefix proof against two caller-pinned root/size anchors.

Hashes and minimal bottom-up SUBPROOF ordering match quant_fund.audit.merkle
and RFC 6962 section 2.1.2. This independent verifier inverts the recursive
proof construction to rebuild both roots; it does not call the legacy verifier.
Every supplied node must be consumed exactly once. Paths need at most 64 nodes
for bounded signed-63-bit sizes; recursion depth is at most 63.

Equal sizes require an empty path and equal roots and prove equality only.
Any zero-size anchor must additionally equal SHA256(empty); this deliberately
strengthens the legacy verifier, which accepts arbitrary equal roots at 0/0.
As in the existing audit convention, growth from zero to a positive size is
unsupported even with the canonical empty root. Shrinking trees fail explicitly.
No signature, leaf validity, complete ledger history, availability, external
trust store, checkpoint identity or research claim is verified.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Digest = Annotated[str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
Outcome = Literal[
    "verified",
    "equal_roots",
    "shrinking_tree",
    "noncanonical_empty_root",
    "empty_growth_unsupported",
    "proof_too_short",
    "proof_too_long",
    "old_root_mismatch",
    "new_root_mismatch",
    "both_roots_mismatch",
]


class Input(InputModel):
    expected_old_size: int = Field(strict=True, ge=0, le=(1 << 63) - 1)
    expected_new_size: int = Field(strict=True, ge=0, le=(1 << 63) - 1)
    expected_old_root: Digest
    expected_new_root: Digest
    proof: list[Digest] = Field(max_length=64)


class Output(OutputModel):
    verified: bool
    outcome: Outcome
    proof_scope: Literal["append_only_prefix", "root_equality_only", "not_verified"]
    expected_old_size: int
    expected_new_size: int
    expected_old_root: str
    expected_new_root: str
    empty_anchors_canonical: bool | None
    expected_proof_nodes: int | None
    supplied_proof_nodes: int
    consumed_proof_nodes: int
    computed_old_root: str | None
    computed_new_root: str | None
    old_root_matches: bool | None
    new_root_matches: bool | None
    conventions: Literal["quant_fund.audit.ordered_rfc6962_sha256"] = (
        "quant_fund.audit.ordered_rfc6962_sha256"
    )
    zero_size_policy: Literal["canonical_empty_roots_required_zero_growth_unsupported"] = (
        "canonical_empty_roots_required_zero_growth_unsupported"
    )
    anchor_scope: Literal["caller_supplied_roots_and_sizes"] = "caller_supplied_roots_and_sizes"
    signature_verified: Literal[False] = False
    ledger_completeness_verified: Literal[False] = False
    leaf_payload_semantics_verified: Literal[False] = False
    research_certified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    old_size, new_size = request.expected_old_size, request.expected_new_size
    old_root, new_root = (
        bytes.fromhex(request.expected_old_root),
        bytes.fromhex(request.expected_new_root),
    )
    empty = hashlib.sha256(b"").digest()
    empty_canonical = (
        (
            (old_size != 0 or hmac.compare_digest(old_root, empty))
            and (new_size != 0 or hmac.compare_digest(new_root, empty))
        )
        if 0 in (old_size, new_size)
        else None
    )
    outcome: Outcome
    required: int | None = None
    old_calculated: bytes | None = None
    new_calculated: bytes | None = None
    old_matches: bool | None = None
    new_matches: bool | None = None
    consumed = 0
    if old_size > new_size:
        outcome = "shrinking_tree"
    elif empty_canonical is False:
        outcome = "noncanonical_empty_root"
    elif old_size == new_size:
        required = 0
        if request.proof:
            outcome = "proof_too_long"
        else:
            old_calculated = new_calculated = old_root
            old_matches = True
            new_matches = hmac.compare_digest(old_root, new_root)
            outcome = "equal_roots" if new_matches else "new_root_mismatch"
    elif old_size == 0:
        outcome = "empty_growth_unsupported"
    else:
        # Count the exact nodes before consuming any, including the seed hash
        # emitted when the old prefix ends in an incomplete right subtree.
        prefix, width, complete = old_size, new_size, True
        required = 0
        while prefix != width:
            required += 1
            split = 1 << ((width - 1).bit_length() - 1)
            if prefix <= split:
                width = split
            else:
                prefix -= split
                width -= split
                complete = False
        required += int(not complete)
        if len(request.proof) < required:
            outcome = "proof_too_short"
        elif len(request.proof) > required:
            outcome = "proof_too_long"
        else:
            siblings = [bytes.fromhex(node) for node in request.proof]

            def next_node() -> bytes:
                nonlocal consumed
                node = siblings[consumed]
                consumed += 1
                return node

            def rebuild(prefix: int, width: int, complete: bool) -> tuple[bytes, bytes]:
                if prefix == width:
                    seed = old_root if complete else next_node()
                    return seed, seed
                split = 1 << ((width - 1).bit_length() - 1)
                if prefix <= split:
                    old_left, new_left = rebuild(prefix, split, complete)
                    right = next_node()
                    return old_left, hashlib.sha256(b"\x01" + new_left + right).digest()
                old_right, new_right = rebuild(prefix - split, width - split, False)
                left = next_node()
                return (
                    hashlib.sha256(b"\x01" + left + old_right).digest(),
                    hashlib.sha256(b"\x01" + left + new_right).digest(),
                )

            old_calculated, new_calculated = rebuild(old_size, new_size, True)
            if consumed != len(siblings):
                raise ValueError("internal consistency path accounting failed")
            old_matches = hmac.compare_digest(old_calculated, old_root)
            new_matches = hmac.compare_digest(new_calculated, new_root)
            if old_matches and new_matches:
                outcome = "verified"
            elif not old_matches and not new_matches:
                outcome = "both_roots_mismatch"
            else:
                outcome = "new_root_mismatch" if old_matches else "old_root_mismatch"
    return Output(
        verified=outcome in ("verified", "equal_roots"),
        outcome=outcome,
        proof_scope="append_only_prefix"
        if outcome == "verified"
        else "root_equality_only"
        if outcome == "equal_roots"
        else "not_verified",
        expected_old_size=old_size,
        expected_new_size=new_size,
        expected_old_root=request.expected_old_root,
        expected_new_root=request.expected_new_root,
        empty_anchors_canonical=empty_canonical,
        expected_proof_nodes=required,
        supplied_proof_nodes=len(request.proof),
        consumed_proof_nodes=consumed,
        computed_old_root=old_calculated.hex() if old_calculated is not None else None,
        computed_new_root=new_calculated.hex() if new_calculated is not None else None,
        old_root_matches=old_matches,
        new_root_matches=new_matches,
    )


OPERATION = Operation(
    id="skills.verify_merkle_consistency",
    kind="skill",
    description="Independently rebuild both audit Merkle roots from an exactly consumed prefix proof and caller-pinned sizes/roots; reject noncanonical empty anchors and explicitly distinguish equality from growth.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
