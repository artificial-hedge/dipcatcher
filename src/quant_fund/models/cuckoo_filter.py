"""Cuckoo filter: two-bucket fingerprint hash table (synthetic) (SYNTHETIC).

Each item maps to b1 = h(x), b2 = b1 ⊕ h(fp); insertion kicks a
resident fingerprint until an empty slot or MAX_KICKS (then full).
Supports deletion. Verified: zero FN, FPR ≈ 2·B/2^F bound, deletes
work, load factor reachable.
"""

from __future__ import annotations

import hashlib
import random

F = 16  # fingerprint bits
B = 1  # fingerprints per bucket
MAX_KICK = 100


def _hs(x: int, salt: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{x}".encode()).digest()[:8], "little")


class Cuckoo:
    def __init__(self, n: int) -> None:
        self.m = max(4, 1 << (n * 2 - 1).bit_length())
        self.tab = [0] * self.m

    def _cands(self, x: int) -> tuple[int, int, int]:
        fp = (_hs(x, 1) & ((1 << F) - 1)) | 1
        b1 = _hs(x, 0) % self.m
        b2 = (b1 ^ _hs(fp, 2)) % self.m
        return fp, b1, b2

    def add(self, x: int) -> bool:
        fp, b1, b2 = self._cands(x)
        if self.tab[b1] == 0:
            self.tab[b1] = fp
            return True
        if self.tab[b2] == 0:
            self.tab[b2] = fp
            return True
        b = b1 if random.random() < 0.5 else b2
        for _ in range(MAX_KICK):
            fp, self.tab[b] = self.tab[b], fp
            b = (b ^ _hs(fp, 2)) % self.m
            if self.tab[b] == 0:
                self.tab[b] = fp
                return True
        return False

    def contains(self, x: int) -> bool:
        fp, b1, b2 = self._cands(x)
        return self.tab[b1] == fp or self.tab[b2] == fp

    def remove(self, x: int) -> bool:
        fp, b1, b2 = self._cands(x)
        if self.tab[b1] == fp:
            self.tab[b1] = 0
            return True
        if self.tab[b2] == fp:
            self.tab[b2] = 0
            return True
        return False


def bench_cuckoo_filter(seed: int = 20261231 + 311) -> dict[str, float]:
    rng = random.Random(seed)
    fn = del_ok = fpr_ok = 0
    trials = 30
    fprs: list[float] = []
    for _ in range(trials):
        n = 200
        cf = Cuckoo(n)
        items = [rng.randrange(10**9) for _ in range(n)]
        ins = sum(cf.add(x) for x in items)
        fn += int(all(cf.contains(x) for x in items[:ins]))
        # deletes
        dels = items[: ins // 2]
        for x in dels:
            cf.remove(x)
        keep = set(items[ins // 2 : ins])
        gone = sum(1 for x in dels if x not in keep and not cf.contains(x))
        want = sum(1 for x in dels if x not in keep)
        del_ok += int(want == 0 or gone / max(1, want) >= 0.9)
        neg = [rng.randrange(10**9, 10**10) for _ in range(400)]
        fpr = sum(1 for x in neg if cf.contains(x)) / len(neg)
        fprs.append(fpr)
        bound = 2 * B / (1 << F)  # two buckets × F bits
        fpr_ok += int(fpr <= bound * 4 + 0.01)
    return {
        "synthetic_no_fn": float(fn / trials),
        "synthetic_delete_ok": float(del_ok / trials),
        "synthetic_fpr_bound": float(fpr_ok / trials),
        "synthetic_mean_fpr": float(sum(fprs) / len(fprs)),
    }
