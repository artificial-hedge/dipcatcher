"""LZW dictionary compression (synthetic).

Classic Welch LZW: encoder grows dictionary on miss, decoder
reconstructs with the KwKwK edge case. Verified: exact round-trip
on random + repetitive corpora, code count below literal length on
redundant text.
"""

from __future__ import annotations

import random


def lzw_encode(data: str) -> list[int]:
    dic = {c: i for i, c in enumerate(sorted(set(data)))}
    nxt = len(dic)
    out: list[int] = []
    w = ""
    for c in data:
        wc = w + c
        if wc in dic:
            w = wc
        else:
            out.append(dic[w])
            dic[wc] = nxt
            nxt += 1
            w = c
    if w:
        out.append(dic[w])
    return out


def lzw_decode(codes: list[int], alphabet: list[str]) -> str:
    dic = {i: c for i, c in enumerate(alphabet)}
    nxt = len(dic)
    w = dic[codes[0]]
    out = [w]
    for c in codes[1:]:
        if c in dic:
            entry = dic[c]
        elif c == nxt:
            entry = w + w[0]
        else:
            raise ValueError("bad LZW code")
        out.append(entry)
        dic[nxt] = w + entry[0]
        nxt += 1
        w = entry
    return "".join(out)


def bench_lzw_compress(seed: int = 20261231 + 292) -> dict[str, float]:
    rng = random.Random(seed)
    rt = 0
    trials = 40
    gains: list[float] = []
    better = 0
    for _ in range(trials):
        alpha = "abc"
        parts = []
        for _ in range(rng.randint(5, 15)):
            if rng.random() < 0.5:
                parts.append(rng.choice(["abab", "abcabc", "aaaa", "b"]) * rng.randint(2, 5))
            else:
                parts.append("".join(rng.choice(alpha) for _ in range(rng.randint(2, 10))))
        data = "".join(parts)
        codes = lzw_encode(data)
        dec = lzw_decode(codes, sorted(set(data)))
        rt += int(dec == data)
        # compare symbol count vs literal length
        ratio = len(codes) / max(1, len(data))
        gains.append(ratio)
        better += int(len(codes) <= len(data))
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_mean_ratio": float(sum(gains) / len(gains)),
        "synthetic_beats_literal": float(better / trials),
    }
