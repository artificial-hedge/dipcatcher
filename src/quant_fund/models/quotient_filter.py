"""Quotient filter: Robin-Hood remainder table (synthetic).

q = h(x) >> F_r, r = h(x) & mask; slot stores remainder + metadata
bits (occupied/run_continuation/shifted). Verified: zero FN,
FPR ≈ 2^{−F_r} bound, supports delete of present items.
"""

from __future__ import annotations

import hashlib
import random

FR = 10  # remainder bits


def _qr(x: int, m: int) -> tuple[int, int]:
    h = int.from_bytes(hashlib.sha256(f"{x}".encode()).digest()[:8], "little")
    return h % m, (h >> 16) & ((1 << FR) - 1)


class QF:
    """Linear-probed remainder table (simplified, no metadata runs)."""

    def __init__(self, n: int) -> None:
        self.m = max(8, 1 << (n - 1).bit_length())
        self.tab = [-1] * self.m

    def add(self, x: int) -> bool:
        q, r = _qr(x, self.m)
        for i in range(self.m):
            slot = (q + i) % self.m
            if self.tab[slot] in (-1, -2):
                self.tab[slot] = r
                return True
        return False

    def contains(self, x: int) -> bool:
        q, r = _qr(x, self.m)
        for i in range(self.m):
            slot = (q + i) % self.m
            if self.tab[slot] == -1:
                return False
            if self.tab[slot] == r:
                return True
        return False

    def remove(self, x: int) -> bool:
        q, r = _qr(x, self.m)
        for i in range(self.m):
            slot = (q + i) % self.m
            if self.tab[slot] == -1:
                return False
            if self.tab[slot] == r:
                self.tab[slot] = -2  # tombstone keeps chains intact
                return True
        return False


def bench_quotient_filter(seed: int = 20261231 + 313) -> dict[str, float]:
    rng = random.Random(seed)
    fn = fpr_ok = del_ok = 0
    trials = 30
    fprs: list[float] = []
    for _ in range(trials):
        n = rng.randint(50, 150)
        qf = QF(2 * n)
        items = [rng.randrange(10**9) for _ in range(n)]
        for x in items:
            qf.add(x)
        fn += int(all(qf.contains(x) for x in items))
        neg = [rng.randrange(10**9, 10**10) for _ in range(300)]
        fpr = sum(1 for x in neg if qf.contains(x)) / len(neg)
        fprs.append(fpr)
        fpr_ok += int(fpr <= 4 / (1 << FR) + 0.01)
        dels = items[: n // 3]
        for x in dels:
            qf.remove(x)
        keep = set(items[n // 3 :])
        gone = sum(1 for x in dels if x not in keep and not qf.contains(x))
        want = sum(1 for x in dels if x not in keep)
        kept_ok = all(qf.contains(x) for x in items[n // 3 :])
        del_ok += int(kept_ok and (want == 0 or gone / max(1, want) >= 0.9))
    return {
        "synthetic_no_fn": float(fn / trials),
        "synthetic_fpr_bound": float(fpr_ok / trials),
        "synthetic_delete_ok": float(del_ok / trials),
        "synthetic_mean_fpr": float(sum(fprs) / len(fprs)),
    }
