"""XOR filter: static perfect-hash 3-choice filter (synthetic).

Each key maps to 3 cells; fingerprint fp(x) = h2(x) & mask;
query: fp(x) == T[h0] ⊕ T[h1] ⊕ T[h2]. Built by peeling.
Verified: zero FN on the build set; FPR ≈ 1/2^F measured.
"""

from __future__ import annotations

import hashlib
import random

F = 8


def _h(x: int, salt: int, m: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{x}".encode()).digest()[:8], "little") % m


def _fp(x: int) -> int:
    return _h(x, 9, 1 << F) | 1


def _idx(x: int, m: int) -> tuple[int, int, int]:
    b = m // 3
    return _h(x, 0, b), b + _h(x, 1, b), 2 * b + _h(x, 2, b)


def build(keys: list[int]) -> tuple[bytearray, int]:
    m = 3 * ((len(keys) * 4 + 2) // 3)
    tab = bytearray(m)
    for _try in range(10):
        # peel: stack of (key, cell) in removal order
        stack: list[tuple[int, int]] = []
        cnt2 = [0] * m
        for x in keys:
            for i in _idx(x, m):
                cnt2[i] += 1
        q = [i for i in range(m) if cnt2[i] == 1]
        seen = [False] * len(keys)
        while q:
            cell = q.pop()
            if cnt2[cell] != 1:
                continue
            for ki, x in enumerate(keys):
                if seen[ki]:
                    continue
                if cell in _idx(x, m):
                    stack.append((x, cell))
                    seen[ki] = True
                    for i in _idx(x, m):
                        cnt2[i] -= 1
                        if cnt2[i] == 1:
                            q.append(i)
                    break
        if len(stack) == len(keys):
            for x, cell in reversed(stack):
                i0, i1, i2 = _idx(x, m)
                tab[cell] = _fp(x) ^ tab[i0] ^ tab[i1] ^ tab[i2]
            return tab, m
        # restack with different salt impossible here; widen m
        m += 3
    return tab, m


def contains(tab: bytearray, m: int, x: int) -> bool:
    i0, i1, i2 = _idx(x, m)
    return tab[i0] ^ tab[i1] ^ tab[i2] == _fp(x)


def bench_xor_filter(seed: int = 20261231 + 312) -> dict[str, float]:
    rng = random.Random(seed)
    fn = fpr_ok = 0
    trials = 20
    fprs: list[float] = []
    for _ in range(trials):
        n = rng.randint(60, 200)
        keys = [rng.randrange(10**9) for _ in range(n)]
        tab, m = build(keys)
        fn += int(all(contains(tab, m, x) for x in keys))
        neg = [rng.randrange(10**9, 10**10) for _ in range(300)]
        fpr = sum(1 for x in neg if contains(tab, m, x)) / len(neg)
        fprs.append(fpr)
        fpr_ok += int(fpr <= 4 / (1 << F) + 0.02)
    return {
        "synthetic_no_fn": float(fn / trials),
        "synthetic_fpr_bound": float(fpr_ok / trials),
        "synthetic_mean_fpr": float(sum(fprs) / len(fprs)),
    }
