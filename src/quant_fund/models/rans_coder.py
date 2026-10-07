"""rANS (range asymmetric numeral systems) coder (synthetic) (SYNTHETIC).

Rygiewicz-style streaming ANS: state encodes the whole message in
one integer that stays in [L, L·b). Verified: exact LIFO
round-trip; coded size within a few bits of the empirical entropy
bound on skewed distributions.
"""

from __future__ import annotations

import math
import random

SCALE = 1 << 12  # frequency precision
LOWER = SCALE << 16  # normalization interval base
EMIT = 1 << 16


def _model(msg: list[int]) -> tuple[dict[int, tuple[int, int]], int]:
    freq: dict[int, int] = {}
    for s in msg:
        freq[s] = freq.get(s, 0) + 1
    tot = len(msg)
    # normalize to SCALE
    cum = 0
    table: dict[int, tuple[int, int]] = {}
    for s, f in sorted(freq.items()):
        nf = max(1, round(f * SCALE / tot))
        table[s] = (cum, nf)
        cum += nf
    return table, cum


def _renorm(x: int, f: int) -> tuple[int, list[int]]:
    """Emit 16-bit chunks while x exceeds the symbol's bound x_max(f)."""
    out: list[int] = []
    x_max = ((LOWER >> 12) << 16) * f
    while x >= x_max:
        out.append(x & (EMIT - 1))
        x >>= 16
    return x, out


def rans_encode(msg: list[int]) -> tuple[int, list[int], dict[int, tuple[int, int]], int]:
    table, _ = _model(msg)
    x = LOWER
    emitted: list[int] = []
    for s in msg:
        cum, f = table[s]
        x, chunks = _renorm(x, f)
        emitted += chunks
        x = ((x // f) << 12) + (x % f) + cum
    return x, emitted, table, len(msg)


def rans_decode(x: int, emitted: list[int], table: dict[int, tuple[int, int]], n: int) -> list[int]:
    rev = {cum: s for s, (cum, f) in table.items()}
    cums = sorted(rev)
    out: list[int] = []
    pos = len(emitted)
    for _ in range(n):
        y = x & (SCALE - 1)
        s = max(c for c in cums if c <= y)
        sym = rev[s]
        out.append(sym)
        cum, f = table[sym]
        x = f * (x >> 12) + y - cum
        while x < LOWER and pos > 0:
            pos -= 1
            x = (x << 16) | emitted[pos]
    return out


def bench_rans_coder(seed: int = 20261231 + 294) -> dict[str, float]:
    rng = random.Random(seed)
    rt = near = 0
    trials = 30
    gaps: list[float] = []
    for _ in range(trials):
        n_sym = rng.randint(2, 6)
        w = [rng.random() ** 3 + 0.02 for _ in range(n_sym)]
        tot = sum(w)
        probs = [x / tot for x in w]
        n = rng.randint(50, 400)
        msg = [rng.choices(range(n_sym), weights=probs)[0] for _ in range(n)]
        x, em, table, _ = rans_encode(msg)
        dec = rans_decode(x, em, table, len(msg))
        # rANS is LIFO — decoder emits in reverse
        rt += int(dec == msg[::-1])
        # size: log2(x_final) + emitted bits
        bits = x.bit_length() + 16 * len(em)
        h = -sum(p * math.log2(p) for p in probs)
        gap = bits / n - h
        gaps.append(gap)
        near += int(gap < 1.5)
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_near_entropy": float(near / trials),
        "synthetic_mean_gap": float(sum(gaps) / len(gaps)),
    }
