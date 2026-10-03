"""SYNTHETIC Merkle signature scheme — many OTS pubkeys in one tree.

Tree of OTS verification keys; each signature = OTS sig + auth path.
Verifies: valid sig+path accepted, wrong index/path rejected, tree root
commits to all leaves.
"""

from __future__ import annotations

import random

from quant_fund.models.winternitz_ots import _digest, _h, keygen, sign, verify

_HMASK = (1 << 64) - 1


def _hh(a: int, b: int) -> int:
    # order-sensitive toy node hash (a XOR-swap would make it commutative)
    return _h(a & _HMASK, (b + 0x9E3779B97F4A7C15) & _HMASK) & _HMASK


def _tree_root(leaves: list[int]) -> list[int]:
    level = leaves
    while len(level) > 1:
        level = [_hh(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level


def _auth_path(leaves: list[int], idx: int) -> list[int]:
    path = []
    level = leaves
    i = idx
    while len(level) > 1:
        path.append(level[i ^ 1])
        i //= 2
        level = [_hh(level[j], level[j + 1]) for j in range(0, len(level), 2)]
    return path


def _root_from(leaf: int, idx: int, path: list[int]) -> int:
    x, i = leaf, idx
    for sib in path:
        x = _hh(x, sib) if i % 2 == 0 else _hh(sib, x)
        i //= 2
    return x


def bench_merkle_ots(seed: int = 20261231 + 422) -> dict[str, float]:
    rng = random.Random(seed)
    ok = path_ok = wrong = 0
    trials = 20
    for _ in range(trials):
        n_leaf = 1 << rng.randrange(3, 5)
        keys = [keygen(rng) for _ in range(n_leaf)]
        pks = [_h(0, pk[0]) for _, pk, _ in keys]
        root = _tree_root(pks)[0]
        idx = rng.randrange(n_leaf)
        sk, pk, salt = keys[idx]
        dig = _digest(rng)
        sig = sign(sk, salt, dig)
        leaf = _h(0, pk[0])
        path = _auth_path(pks, idx)
        ok += int(verify(sig, salt, dig, pk) and _root_from(leaf, idx, path) == root)
        # wrong index → different root
        wrong_idx = (idx + 1) % n_leaf
        path_ok += int(_root_from(leaf, wrong_idx, path) != root or leaf == pks[wrong_idx])
        # corrupt path element → reject
        path2 = list(path)
        path2[0] ^= 1
        wrong += int(_root_from(leaf, idx, path2) != root)
    return {
        "synthetic_sig_and_path_valid": float(ok / trials),
        "synthetic_wrong_index_fails": float(path_ok / trials),
        "synthetic_corrupt_path_fails": float(wrong / trials),
    }
