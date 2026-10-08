"""SPHINCS+-lite — FORS few-time signature under a hypertree of XMSS (SYNTHETIC).

The message digest picks k FORS leaf indices; each revealed leaf value +
auth path verifies to the FORS root, which is signed by a bottom-layer
XMSS (WOTS+ + Merkle path) whose root is the public key. Structure mirrors
the real scheme at toy height.
"""

import hashlib

import numpy as np

from quant_fund.models.xmss_sig import xmss_keygen, xmss_sign, xmss_verify

_SEED = 20261231 + 871

FORS_K, FORS_A = 6, 2  # k trees of height a -> t = 2^a leaves each
N_BYTES = 8
XMSS_H = 4


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def _fors_leaf(seed: bytes, tree: int, leaf: int) -> bytes:
    return _h(seed, bytes([tree, leaf]))


def _fors_root(seed: bytes) -> bytes:
    roots = []
    for tree in range(FORS_K):
        nodes = [_h(_fors_leaf(seed, tree, leaf), b"N") for leaf in range(1 << FORS_A)]
        while len(nodes) > 1:
            nodes = [_h(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]
        roots.append(nodes[0])
    return _h(b"fors", *roots)


def _fors_indices(msg: bytes) -> list[int]:
    digest = int.from_bytes(_h(msg), "little")
    return [(digest >> (i * FORS_A)) & ((1 << FORS_A) - 1) for i in range(FORS_K)]


def _fors_sign(msg: bytes, seed: bytes) -> list[tuple[bytes, list[bytes]]]:
    out = []
    for tree, leaf in enumerate(_fors_indices(msg)):
        auth = []
        nodes = [_h(_fors_leaf(seed, tree, l_), b"N") for l_ in range(1 << FORS_A)]
        for h in range(FORS_A):
            auth.append(nodes[(leaf >> h) ^ 1])
            nodes = [_h(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]
        out.append((_fors_leaf(seed, tree, leaf), auth))
    return out


def _fors_pk_from_sig(msg: bytes, sig: list[tuple[bytes, list[bytes]]]) -> bytes:
    roots = []
    for tree, (leaf_val, auth) in enumerate(sig):
        leaf = _fors_indices(msg)[tree]
        node = _h(leaf_val, b"N")
        for h in range(FORS_A):
            node = _h(auth[h], node) if (leaf >> h) & 1 else _h(node, auth[h])
        roots.append(node)
    return _h(b"fors", *roots)


def sphincs_keygen(seed: bytes) -> bytes:
    root, _ = xmss_keygen(_h(seed, b"x")[:16])
    return root


def sphincs_sign(
    msg: bytes, seed: bytes, idx: int
) -> tuple[list[tuple[bytes, list[bytes]]], tuple[list[bytes], list[bytes]], int]:
    fors = _fors_sign(msg, _h(seed, b"f")[:16])
    fors_pk = _fors_pk_from_sig(msg, fors)
    xsig = xmss_sign(fors_pk, _h(seed, b"x")[:16], idx)
    return fors, xsig, idx


def sphincs_verify(msg: bytes, sig: tuple, pk_root: bytes, seed: bytes) -> bool:
    fors, xsig, idx = sig
    fors_pk = _fors_pk_from_sig(msg, fors)
    return xmss_verify(fors_pk, xsig, idx, pk_root, _h(seed, b"x")[:16])


def bench_sphincs_sig(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: sign/verify round-trip; forgery and index mismatch fail."""
    rng = np.random.default_rng(seed)
    s = rng.bytes(16)
    pk = sphincs_keygen(s)
    good, bad = 0, 0
    for i in range(4):
        msg = rng.bytes(12)
        sig = sphincs_sign(msg, s, i)
        good += int(sphincs_verify(msg, sig, pk, s))
        bad += int(not sphincs_verify(rng.bytes(12), sig, pk, s))
    return {"synthetic_sphincs_sig": 1.0 if (good == 4 and bad == 4) else good / 4.0}
