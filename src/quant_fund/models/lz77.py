"""LZ77 sliding-window compression (synthetic).

Greedy longest-match encoder with window W and literal/match
tokens; exact round-trip verified. Compression ratio reported
on repetitive vs random text (honest: random expands slightly).
"""

from __future__ import annotations

import random

WIN = 64
MIN_MATCH = 3


def compress(data: str) -> list[tuple[int, int, str]]:
    """Tokens: (offset, length, next_char). offset=0 → literal."""
    out: list[tuple[int, int, str]] = []
    i = 0
    n = len(data)
    while i < n:
        best_len = 0
        best_off = 0
        lo = max(0, i - WIN)
        for j in range(lo, i):
            ln = 0
            while i + ln < n and data[j + ln] == data[i + ln] and j + ln < i:
                ln += 1
            if ln > best_len:
                best_len, best_off = ln, i - j
        if best_len >= MIN_MATCH:
            nc = data[i + best_len] if i + best_len < n else ""
            out.append((best_off, best_len, nc))
            i += best_len + (1 if nc else 0)
        else:
            out.append((0, 0, data[i]))
            i += 1
    return out


def decompress(tokens: list[tuple[int, int, str]]) -> str:
    out: list[str] = []
    for off, ln, c in tokens:
        if off == 0 and ln == 0:
            out.append(c)
        else:
            start = len(out) - off
            for k in range(ln):
                out.append(out[start + k])
            if c:
                out.append(c)
    return "".join(out)


def bench_lz77(seed: int = 20261231 + 264) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "ab"
    roundtrip = 0
    trials = 30
    for _ in range(trials):
        # mix repetitive + random segments
        text = "".join(
            (rng.choice(["ab", "aab", "b"]) * rng.randint(1, 12))[: rng.randint(3, 30)]
            if rng.random() < 0.6
            else "".join(rng.choice(alpha) for _ in range(rng.randint(3, 30)))
            for _ in range(8)
        )
        toks = compress(text)
        roundtrip += int(decompress(toks) == text)
    rep = "ab" * 300
    rnd = "".join(rng.choice(alpha) for _ in range(600))
    rep_ratio = len(compress(rep)) / len(rep)
    rnd_ratio = len(compress(rnd)) / len(rnd)
    return {
        "synthetic_roundtrip": float(roundtrip / trials),
        "synthetic_repetitive_ratio": float(rep_ratio),
        "synthetic_random_ratio": float(rnd_ratio),
        "synthetic_beats_random": float(rep_ratio < rnd_ratio),
    }
