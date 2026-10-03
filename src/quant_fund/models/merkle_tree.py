"""Merkle tree + inclusion proofs — SYNTHETIC (sha256).

Verified: root recomputation, proof validity for every leaf, tamper
detection.
"""

from __future__ import annotations

import hashlib
import random


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def root(leaves: list[bytes]) -> bytes:
    lvl = [_h(x) for x in leaves]
    while len(lvl) > 1:
        if len(lvl) % 2:
            lvl.append(lvl[-1])
        lvl = [_h(lvl[i] + lvl[i + 1]) for i in range(0, len(lvl), 2)]
    return lvl[0] if lvl else b""


def prove(leaves: list[bytes], i: int) -> list[tuple[bytes, int]]:
    """Proof: list of (sibling_hash, side) — side 1 = sibling on right."""
    lvl = [_h(x) for x in leaves]
    proof: list[tuple[bytes, int]] = []
    idx = i
    while len(lvl) > 1:
        if len(lvl) % 2:
            lvl.append(lvl[-1])
        sib = idx ^ 1
        proof.append((lvl[sib], int(sib > idx)))
        idx //= 2
        lvl = [_h(lvl[j] + lvl[j + 1]) for j in range(0, len(lvl), 2)]
    return proof


def verify(leaf: bytes, proof: list[tuple[bytes, int]], rt: bytes) -> bool:
    acc = _h(leaf)
    for sib, side in proof:
        acc = _h(acc + sib) if side else _h(sib + acc)
    return acc == rt


def bench_merkle_tree(seed: int = 20261231 + 340) -> dict[str, float]:
    rng = random.Random(seed)
    proof_ok = tamper_ok = 0
    trials = 30
    total_leaves = 0
    proof_hits = 0
    for _ in range(trials):
        n = rng.randrange(1, 40)
        leaves = [rng.randbytes(32) for _ in range(n)]
        rt = root(leaves)
        for i in range(n):
            total_leaves += 1
            proof_hits += int(verify(leaves[i], prove(leaves, i), rt))
        # tamper: flip one leaf → root changes
        bad = leaves[:]
        bad[0] = b"\x00" + bad[0][1:]
        tamper_ok += int(root(bad) != rt)
        # wrong proof fails
        if n >= 2:
            proof_ok += int(not verify(leaves[0], prove(leaves, 1), rt))
        else:
            proof_ok += 1
    return {
        "synthetic_proof_valid": float(proof_hits / max(1, total_leaves)),
        "synthetic_wrong_proof_rejected": float(proof_ok / trials),
        "synthetic_tamper_detected": float(tamper_ok / trials),
    }
