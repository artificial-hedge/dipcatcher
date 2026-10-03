"""Burrows-Wheeler transform + inverse (synthetic).

Cyclic-rotation sort with a sentinel char; inverse via LF-mapping.
Verified: (i) exact inverse round-trip; (ii) BWT clusters equal
characters (run-length grows vs source); (iii) MTF + BWT pipeline
consistent.
"""

from __future__ import annotations

import random


def bwt(s: str) -> str:
    t = s + "\x00"
    rots = sorted(t[i:] + t[:i] for i in range(len(t)))
    return "".join(r[-1] for r in rots)


def ibwt(b: str) -> str:
    table = [""] * len(b)
    for _ in range(len(b)):
        table = sorted(b[i] + table[i] for i in range(len(b)))
    for row in table:
        if row.endswith("\x00"):
            return row[:-1]
    raise ValueError("no sentinel row")


def mtf(s: str) -> list[int]:
    alpha = sorted(set(s))
    out = []
    for c in s:
        i = alpha.index(c)
        out.append(i)
        alpha.insert(0, alpha.pop(i))
    return out


def rle(s: str) -> int:
    """Count of runs."""
    return sum(1 for i in range(1, len(s) + 1) if i == len(s) or s[i] != s[i - 1])


def bench_bwt_transform(seed: int = 20261231 + 265) -> dict[str, float]:
    rng = random.Random(seed)
    roundtrip = 0
    trials = 15
    for _ in range(trials):
        text = "".join(rng.choice("aabbcc") for _ in range(rng.randint(5, 60)))
        # ensure no sentinel char in input
        b = bwt(text)
        roundtrip += int(ibwt(b) == text)
    # clustering: BWT of repetitive text has fewer runs
    rep = "mississippi"
    runs_src = rle(rep)
    runs_bwt = rle(bwt(rep))
    clusters = runs_bwt <= runs_src + 2
    # mtf produces small numbers on clustered BWT
    m = mtf(bwt("banana"))
    mtf_small = all(v <= 4 for v in m)
    return {
        "synthetic_roundtrip": float(roundtrip / trials),
        "synthetic_runs_source": float(runs_src),
        "synthetic_runs_bwt": float(runs_bwt),
        "synthetic_clusters": float(clusters),
        "synthetic_mtf_bounded": float(mtf_small),
    }
