"""W5 contracts smoke tests (DESIGN.md §3, §8.1).

The contracts module is the coordination point for all five workstreams;
these tests pin the canonical-hash determinism contract (§8.1.3: no reliance
on dict/set iteration order) and the Merkle helper's domain separation.

The Merkle v2 block pins the malleability fix (docs/SOTA/12 N8/N9,
docs/SOTA/23 §1): v1 stays byte-identical against captured fixtures (sealed
``proofcore/1`` bundles reference v1 roots — immutable evidence must not
move), while v2 is order-preserving, count-committing, duplicate-rejecting,
and domain-separated from v1.
"""

from __future__ import annotations

import pytest

from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    HASH_HEX_LEN,
    MERKLE_ALGORITHM_V1,
    MERKLE_ALGORITHM_V2,
    SCHEMA_VERSION,
    ProofError,
    canonical_json_bytes,
    merkle_inclusion_proof_hex_v2,
    merkle_root_from_inclusion_hex_v2,
    merkle_root_hex,
    merkle_root_hex_v2,
    sha256_hex_bytes,
    sha256_hex_json,
)


def test_schema_version_and_genesis() -> None:
    assert SCHEMA_VERSION == "proofcore/1"
    assert GENESIS_HASH == "0" * 64


def test_canonical_json_is_key_order_independent() -> None:
    """§8.1.3: permuted insertion order must hash identically (PYTHONHASHSEED
    cannot save us post-interpreter-start, so sorting is the contract)."""
    a = {"z": 1, "a": [3, 2, 1], "m": {"b": True, "a": None}}
    b = {"m": {"a": None, "b": True}, "a": [3, 2, 1], "z": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert sha256_hex_json(a) == sha256_hex_json(b)


def test_canonical_json_tight_separators() -> None:
    assert canonical_json_bytes({"a": 1, "b": [2]}) == b'{"a":1,"b":[2]}'


def test_sha256_hex_bytes_length() -> None:
    assert len(sha256_hex_bytes(b"")) == HASH_HEX_LEN


def test_merkle_root_empty_hashes_empty_string() -> None:
    assert merkle_root_hex([]) == sha256_hex_bytes(b"")


def test_merkle_root_order_independent() -> None:
    # v1 semantics, pinned on purpose: sealed proofcore/1 bundles commit
    # sorted-leaf roots and v1 must stay byte-identical forever (docs/SOTA/23
    # §1). Order-independence is a v1 *defect* for ordered logs — new code
    # uses merkle_root_hex_v2, tested below.
    leaves = [sha256_hex_bytes(bytes([i])) for i in range(5)]
    assert merkle_root_hex(leaves) == merkle_root_hex(list(reversed(leaves)))


def test_merkle_root_rejects_non_hex_leaves() -> None:
    with pytest.raises(ProofError):
        merkle_root_hex(["not-a-digest"])


def test_merkle_single_leaf_is_domain_separated() -> None:
    leaf = sha256_hex_bytes(b"x")
    root = merkle_root_hex([leaf])
    assert root == sha256_hex_bytes(b"PC:leaf:" + bytes.fromhex(leaf))
    assert root != leaf  # second-preimage resistance by domain separation


# ---------------------------------------------------------------------------
# Merkle v1 backward compatibility — CAPTURED PRE-CHANGE FIXTURES
#
# Values below were captured on the *unmodified* implementation (before v2
# existed) and cross-check against docs/SOTA/12 §2.2 N8's published
# measurement (root prefix d44f177398ab19ee). Sealed proofcore/1 bundles
# reference these roots; they are immutable evidence and must never move.
# ---------------------------------------------------------------------------

#: sha256("a".."d") — the reproduction leaf set used in docs/SOTA/12 §7.
FIXTURE_LEAVES_ABC: tuple[str, ...] = (
    "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d",
    "2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6",
)
FIXTURE_LEAF_D = "18ac3e7343f016890c510e93f935261169d9e3f565436429830faf0934f4f8e4"

#: v1 root of [sha256("a"), sha256("b"), sha256("c")] — captured pre-change.
V1_ROOT_ABC = "d44f177398ab19eeacbe97b08164e533e2694582aa862e792c43a1bba49f36ce"
#: v1 root of the four-leaf set [a, b, c, d] — captured pre-change.
V1_ROOT_ABCD = "c0d6aea8ab0fe5f262fa32d56b76d1db03e2b3b68aad41042cb563c2664975f4"
#: v1 root of [sha256(bytes([i])) for i in range(5)] — captured pre-change.
V1_ROOT_FIVE_BYTES = "ecfe42dd418e219ab23480d3f27b2f71575cfa7af73da0565e380883ff6ee8e3"


def test_merkle_v1_matches_captured_pre_change_fixture() -> None:
    """Regression fixture: v1 output is byte-identical to the pre-change build."""
    assert merkle_root_hex(list(FIXTURE_LEAVES_ABC)) == V1_ROOT_ABC
    assert merkle_root_hex([*FIXTURE_LEAVES_ABC, FIXTURE_LEAF_D]) == V1_ROOT_ABCD
    five = [sha256_hex_bytes(bytes([i])) for i in range(5)]
    assert merkle_root_hex(five) == V1_ROOT_FIVE_BYTES


def test_merkle_v1_empty_root_unchanged() -> None:
    assert merkle_root_hex([]) == sha256_hex_bytes(b"")
    assert merkle_root_hex([]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_merkle_v1_malleability_is_reproduced_and_documented() -> None:
    """Pins the *defect* so nobody mistakes v1 for a commitment.

    Re-appending the maximum leaf leaves the v1 root unchanged (N8), and a
    permutation leaves it unchanged (N9). Both are exactly why v2 exists; the
    v1 behaviour is preserved because sealed bundles reference it.
    """
    leaves = list(FIXTURE_LEAVES_ABC)
    assert merkle_root_hex(leaves) == merkle_root_hex([*leaves, max(leaves)])
    assert merkle_root_hex(leaves) == merkle_root_hex(list(reversed(leaves)))


# ---------------------------------------------------------------------------
# Merkle v2 — order-preserving, count-committing, extension-resistant
# ---------------------------------------------------------------------------


def test_merkle_v2_rejects_extension_by_the_max_leaf() -> None:
    """The v1 attack, closed.

    v1 let an attacker re-append the maximum leaf and keep the root. Under v2
    that exact construction is a duplicate and is rejected outright — the
    attack cannot even be expressed. (A *distinct* new leaf that is larger
    than every existing one is covered by the count-commitment test below.)
    """
    leaves = list(FIXTURE_LEAVES_ABC)
    # v1 silently returns the same root for L and L+[max(L)]:
    assert merkle_root_hex(leaves) == merkle_root_hex([*leaves, max(leaves)])
    # v2 refuses to build it at all:
    with pytest.raises(ProofError, match="duplicate"):
        merkle_root_hex_v2([*leaves, max(leaves)])


def test_merkle_v2_extension_by_a_distinct_larger_leaf_changes_root() -> None:
    """Count-commitment defeats extension by a genuinely new (larger) leaf."""
    leaves = list(FIXTURE_LEAVES_ABC)
    bigger = "f" * 64  # valid lowercase sha256-shaped digest, larger than every leaf
    assert bigger not in leaves and bigger > max(leaves)
    assert merkle_root_hex_v2([*leaves, bigger]) != merkle_root_hex_v2(leaves)


def test_merkle_v2_rejects_extension_by_any_leaf() -> None:
    """Stronger than the brief: appending *any* distinct leaf moves the root."""
    leaves = list(FIXTURE_LEAVES_ABC)
    base = merkle_root_hex_v2(leaves)
    for extra in (FIXTURE_LEAF_D, sha256_hex_bytes(b"zzz"), "f" * 64):
        assert extra not in leaves
        assert merkle_root_hex_v2([*leaves, extra]) != base


def test_merkle_v2_is_order_sensitive() -> None:
    """Permuting leaves changes the root (kills N9)."""
    leaves = list(FIXTURE_LEAVES_ABC)
    base = merkle_root_hex_v2(leaves)
    for permutation in (
        [leaves[2], leaves[1], leaves[0]],
        [leaves[1], leaves[0], leaves[2]],
        [leaves[0], leaves[2], leaves[1]],
    ):
        assert merkle_root_hex_v2(permutation) != base


def test_merkle_v2_commits_the_leaf_count() -> None:
    assert merkle_root_hex_v2([FIXTURE_LEAVES_ABC[0]]) != merkle_root_hex_v2(
        [FIXTURE_LEAVES_ABC[0], FIXTURE_LEAVES_ABC[1]]
    )


def test_merkle_v2_rejects_duplicate_leaves() -> None:
    """The extension construction cannot even be expressed under v2."""
    leaves = list(FIXTURE_LEAVES_ABC)
    with pytest.raises(ProofError, match="duplicate"):
        merkle_root_hex_v2([*leaves, leaves[0]])
    with pytest.raises(ProofError, match="duplicate"):
        merkle_root_hex_v2([leaves[0], leaves[0]])


def test_merkle_v2_domain_separated_from_v1() -> None:
    """v1 and v2 roots over the same leaves can never collide."""
    leaves = list(FIXTURE_LEAVES_ABC)
    assert merkle_root_hex_v2(leaves) != merkle_root_hex(leaves)
    assert merkle_root_hex_v2([]) != merkle_root_hex([])
    assert merkle_root_hex_v2(leaves[:1]) != merkle_root_hex(leaves[:1])
    # v2's leaf/node tags are structurally distinct from v1's PC:leaf:/PC:node:.
    leaf_tagged = sha256_hex_bytes(b"PC2:leaf:3:0:" + bytes.fromhex(leaves[0]))
    assert merkle_root_hex_v2(leaves) != leaf_tagged
    assert merkle_root_hex_v2([leaves[0]]) == sha256_hex_bytes(
        b"PC2:leaf:1:0:" + bytes.fromhex(leaves[0])
    )


def test_merkle_v2_empty_root_is_tagged_not_the_bare_empty_hash() -> None:
    assert merkle_root_hex_v2([]) == sha256_hex_bytes(b"PC2:tree:0")
    assert merkle_root_hex_v2([]) != sha256_hex_bytes(b"")


def test_merkle_v2_rejects_non_hex_leaves() -> None:
    with pytest.raises(ProofError):
        merkle_root_hex_v2(["not-a-digest"])
    with pytest.raises(ProofError):
        merkle_root_hex_v2(["AB" * 32])


def test_merkle_v2_is_deterministic_over_widths() -> None:
    for count in range(1, 9):
        leaves = [sha256_hex_bytes(f"leaf-{count}-{i}".encode()) for i in range(count)]
        assert merkle_root_hex_v2(leaves) == merkle_root_hex_v2(list(leaves))
        assert len(merkle_root_hex_v2(leaves)) == HASH_HEX_LEN


def test_merkle_algorithm_tags_are_distinct() -> None:
    assert MERKLE_ALGORITHM_V1 == "merkle-sorted-v1"
    assert MERKLE_ALGORITHM_V2 == "merkle-ordered-v2"
    assert MERKLE_ALGORITHM_V1 != MERKLE_ALGORITHM_V2


def test_merkle_v2_inclusion_proof_round_trips() -> None:
    for count in range(1, 9):
        leaves = [sha256_hex_bytes(f"leaf-{count}-{i}".encode()) for i in range(count)]
        root = merkle_root_hex_v2(leaves)
        for index in range(count):
            proof = merkle_inclusion_proof_hex_v2(leaves, index)
            assert merkle_root_from_inclusion_hex_v2(leaves[index], index, count, proof) == root


def test_merkle_v2_inclusion_proof_rejects_wrong_coordinates() -> None:
    leaves = list(FIXTURE_LEAVES_ABC) + [FIXTURE_LEAF_D]
    root = merkle_root_hex_v2(leaves)
    proof = merkle_inclusion_proof_hex_v2(leaves, 1)
    # A different leaf at the same index does not rebuild the root.
    assert merkle_root_from_inclusion_hex_v2(leaves[2], 1, len(leaves), proof) != root
    # A wrong index does not rebuild the root either.
    assert merkle_root_from_inclusion_hex_v2(leaves[1], 2, len(leaves), proof) != root
    # Truncated, padded, or malformed paths fail closed.
    with pytest.raises(ProofError, match="requires exactly"):
        merkle_root_from_inclusion_hex_v2(leaves[1], 1, len(leaves), proof[:-1])
    with pytest.raises(ProofError, match="requires exactly"):
        merkle_root_from_inclusion_hex_v2(leaves[1], 1, len(leaves), [*proof, proof[0]])
    with pytest.raises(ProofError):
        merkle_root_from_inclusion_hex_v2(leaves[1], 1, len(leaves), ["zz"])
    with pytest.raises(ProofError):
        merkle_root_from_inclusion_hex_v2("bad", 1, len(leaves), proof)
    with pytest.raises(ProofError):
        merkle_root_from_inclusion_hex_v2(leaves[1], 9, len(leaves), proof)
    with pytest.raises(ProofError):
        merkle_inclusion_proof_hex_v2(leaves, 9)
    with pytest.raises(ProofError):
        merkle_inclusion_proof_hex_v2([], 0)
