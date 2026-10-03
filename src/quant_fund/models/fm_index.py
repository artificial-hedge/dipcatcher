"""SYNTHETIC FM-index backward search.

BWT + C-table + Occ checkpoints for O(pattern)-time exact-match counting —
the engine inside BWA/bowtie aligners. Verified against naive find on
random strings.
"""

from __future__ import annotations

import random


def _bwt(s: str) -> tuple[str, list[int]]:
    """BWT with terminal $ sentinel; returns (bwt string, suffix array)."""
    s = s + "$"
    sa = sorted(range(len(s)), key=lambda i: s[i:])
    bwt = "".join(s[i - 1] if i > 0 else "$" for i in sa)
    return bwt, sa


class FMIndex:
    def __init__(self, s: str, occ_step: int = 8):
        self.s = s + "$"
        self.bwt, self.sa = _bwt(s)
        self.alpha = sorted(set(self.bwt))
        counts = {c: self.bwt.count(c) for c in self.alpha}
        self.C: dict[str, int] = {}
        tot = 0
        for c in self.alpha:
            self.C[c] = tot
            tot += counts[c]
        # Occ checkpoints: occ[c][i] = count of c in bwt[:i*step]
        self.step = occ_step
        self.occ: dict[str, list[int]] = {c: [0] for c in self.alpha}
        for c in self.alpha:
            running = 0
            for i, ch in enumerate(self.bwt):
                if ch == c:
                    running += 1
                if (i + 1) % occ_step == 0:
                    self.occ[c].append(running)
            self.occ[c].append(running)

    def _occ(self, c: str, i: int) -> int:
        """count of c in bwt[:i]"""
        k, rem = divmod(i, self.step)
        base = self.occ[c][k] if k < len(self.occ[c]) else self.occ[c][-1]
        start = k * self.step
        return base + self.bwt[start:i].count(c)

    def count(self, p: str) -> int:
        """Backward search: number of occurrences of p."""
        lo, hi = 0, len(self.bwt)
        for c in reversed(p):
            if c not in self.C:
                return 0
            lo = self.C[c] + self._occ(c, lo)
            hi = self.C[c] + self._occ(c, hi)
            if lo >= hi:
                return 0
        return hi - lo


def bench_fm_index(seed: int = 20261231 + 513) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "ACG"
    ok = 0
    n = 60
    for _ in range(n):
        s = "".join(rng.choice(alpha) for _ in range(rng.randrange(15, 40)))
        fm = FMIndex(s, occ_step=4)
        p = "".join(rng.choice(alpha) for _ in range(rng.randrange(1, 6)))
        truth = s.count(p) + (1 if p == "$" else 0) - (s.count(p) if p.endswith("$") else 0)
        truth = sum(1 for i in range(len(s) - len(p) + 1) if s[i : i + len(p)] == p) + (
            1 if (s + "$").find(p) >= 0 and "$" in p else 0
        )
        ok += int(fm.count(p) == truth)
    # locate: suffix-array interval rows point at real occurrences
    loc = 0
    for _ in range(30):
        s = "".join(rng.choice(alpha) for _ in range(30))
        fm = FMIndex(s)
        p = "AC"
        lo, hi = 0, len(fm.bwt)
        for c in reversed(p):
            lo = fm.C[c] + fm._occ(c, lo)
            hi = fm.C[c] + fm._occ(c, hi)
        loc += int(all(fm.s[fm.sa[r] :].startswith(p) for r in range(lo, hi)))
    return {
        "synthetic_count_exact": ok / n,
        "synthetic_interval_locates": loc / 30,
    }
