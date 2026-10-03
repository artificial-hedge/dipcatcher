"""XMSS — extended Merkle signature scheme over WOTS+.

WOTS+ (w=16, n=8-byte digests) one-time signatures plus an h-level Merkle
tree of WOTS+ public keys. Signing reveals one leaf's WOTS+ sig + auth
path; verification recomputes the leaf then walks the path to the root.
"""

import hashlib

import numpy as np

_SEED = 20261231 + 870

N_BYTES, W = 8, 16
LEN1 = 2 * N_BYTES  # ceil(8n / log2 w) = 16
LEN2 = 2  # floor(log2(len1*(w-1)) / log2 w) + 1 = 2
LEN = LEN1 + LEN2
XMSS_H = 4


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def _chain(x: bytes, i: int, s: int, masks: list[bytes]) -> bytes:
    for j in range(i, i + s):
        x = _h(bytes([a ^ b for a, b in zip(x, masks[j], strict=True)]), x)
    return x


def _wots_keygen(seed: bytes, masks: list[bytes]) -> tuple[list[bytes], bytes]:
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"w")[:4], "little"))
    sk = [rng.bytes(32) for _ in range(LEN)]
    pk_elems = [_chain(x, 0, W - 1, masks) for x in sk]
    return sk, _h(b"pk", *pk_elems)


def _msg_chain_lens(msg: bytes) -> list[int]:
    nibbles: list[int] = []
    for byte in _h(msg)[:N_BYTES]:
        nibbles += [byte >> 4, byte & 0xF]
    csum = sum(W - 1 - v for v in nibbles)
    csum_bytes = csum.to_bytes(2, "big")
    for byte in csum_bytes:
        nibbles += [byte >> 4, byte & 0xF]
    return nibbles[:LEN]


def wots_sign(msg: bytes, seed: bytes, masks: list[bytes]) -> list[bytes]:
    sk, _ = _wots_keygen(seed, masks)
    lens = _msg_chain_lens(msg)
    return [_chain(sk[i], 0, lens[i], masks) for i in range(LEN)]


def wots_pk_from_sig(msg: bytes, sig: list[bytes], masks: list[bytes]) -> bytes:
    lens = _msg_chain_lens(msg)
    pk_elems = [_chain(sig[i], lens[i], W - 1 - lens[i], masks) for i in range(LEN)]
    return _h(b"pk", *pk_elems)


def _masks(seed: bytes) -> list[bytes]:
    rng = np.random.default_rng(int.from_bytes(_h(seed, b"mk")[:4], "little"))
    return [rng.bytes(32) for _ in range(W)]


def _tree(leaf_pks: list[bytes]) -> list[list[bytes]]:
    levels = [leaf_pks]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([_h(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)])
    return levels


def xmss_keygen(seed: bytes) -> tuple[bytes, list[bytes]]:
    masks = _masks(seed)
    leaf_seeds = [_h(seed, bytes([i])) for i in range(1 << XMSS_H)]
    leaf_pks = [_wots_keygen(ls, masks)[1] for ls in leaf_seeds]
    root = _tree(leaf_pks)[-1][0]
    return root, leaf_seeds


def xmss_sign(msg: bytes, seed: bytes, idx: int) -> tuple[list[bytes], list[bytes]]:
    masks = _masks(seed)
    leaf_seeds = [_h(seed, bytes([i])) for i in range(1 << XMSS_H)]
    leaf_pks = [_wots_keygen(ls, masks)[1] for ls in leaf_seeds]
    sig = wots_sign(msg, leaf_seeds[idx], masks)
    levels = _tree(leaf_pks)
    auth = [levels[h][(idx >> h) ^ 1] for h in range(XMSS_H)]
    return sig, auth


def xmss_verify(
    msg: bytes, sig: tuple[list[bytes], list[bytes]], idx: int, root: bytes, seed: bytes
) -> bool:
    wsig, auth = sig
    masks = _masks(seed)
    node = wots_pk_from_sig(msg, wsig, masks)
    for h in range(XMSS_H):
        node = _h(auth[h], node) if (idx >> h) & 1 else _h(node, auth[h])
    return node == root


def bench_xmss_sig(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: every tree leaf signs+verifies; forgery and wrong index fail."""
    rng = np.random.default_rng(seed)
    seed_bytes = rng.bytes(16)
    root, _ = xmss_keygen(seed_bytes)
    good, bad = 0, 0
    n_leaf = 1 << XMSS_H
    for i in range(n_leaf):
        msg = rng.bytes(12)
        sig = xmss_sign(msg, seed_bytes, i)
        good += int(xmss_verify(msg, sig, i, root, seed_bytes))
        bad += int(not xmss_verify(msg, sig, (i + 1) % n_leaf, root, seed_bytes))
        bad += int(not xmss_verify(rng.bytes(12), sig, i, root, seed_bytes))
    total_bad = 2 * n_leaf
    return {"synthetic_xmss_sig": 1.0 if (good == n_leaf and bad == total_bad) else good / n_leaf}
