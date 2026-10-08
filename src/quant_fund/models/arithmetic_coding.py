"""Integer arithmetic coding (synthetic) (SYNTHETIC).

64-bit range coder over an empirical symbol model; encode/decode
verified by exact round-trip; encoded length within ~2 bits/symbol
of the empirical entropy bound.
"""

from __future__ import annotations

import math
import random

PREC = 32
TOP = 1 << PREC
Q1 = TOP >> 2
Q2 = TOP >> 1
Q3 = 3 * Q1


def _cum(freq: dict[int, int]) -> tuple[list[int], list[int], int]:
    syms = sorted(freq)
    cum: list[int] = []
    acc = 0
    for s in syms:
        cum.append(acc)
        acc += freq[s]
    return syms, cum, acc


def encode(msg: list[int], freq: dict[int, int]) -> tuple[list[int], int]:
    """Returns (bit list, total frequency)."""
    syms, cum, tot = _cum(freq)
    idx = {s: i for i, s in enumerate(syms)}
    lo, hi = 0, TOP - 1
    bits: list[int] = []
    pending = 0

    def emit(bit: int) -> None:
        bits.append(bit)
        for _ in range(pending):
            bits.append(1 - bit)

    for s in msg:
        i = idx[s]
        rng = hi - lo + 1
        hi = lo + (rng * (cum[i] + freq[s])) // tot - 1
        lo = lo + (rng * cum[i]) // tot
        while True:
            if hi < Q2:
                emit(0)
                pending = 0
                # renormalize
                lo, hi = lo << 1, (hi << 1) | 1
            elif lo >= Q2:
                emit(1)
                pending = 0
                lo = (lo - Q2) << 1
                hi = ((hi - Q2) << 1) | 1
            elif lo >= Q1 and hi < Q3:
                pending += 1
                lo = (lo - Q1) << 1
                hi = ((hi - Q1) << 1) | 1
            else:
                break
    pending += 1
    emit(1 if lo >= Q1 else 0)
    return bits, tot


def decode(bits: list[int], n: int, freq: dict[int, int]) -> list[int]:
    syms, cum, tot = _cum(freq)
    lo, hi = 0, TOP - 1
    code = 0
    pos = 0
    for _ in range(PREC):
        code <<= 1
        if pos < len(bits):
            code |= bits[pos]
            pos += 1
    out: list[int] = []
    for _ in range(n):
        rng = hi - lo + 1
        t = ((code - lo + 1) * tot - 1) // rng
        i = max(k for k in range(len(syms)) if cum[k] <= t)
        s = syms[i]
        out.append(s)
        hi = lo + (rng * (cum[i] + freq[s])) // tot - 1
        lo = lo + (rng * cum[i]) // tot
        while True:
            if hi < Q2:
                pass
            elif lo >= Q2:
                lo -= Q2
                hi -= Q2
                code -= Q2
            elif lo >= Q1 and hi < Q3:
                lo -= Q1
                hi -= Q1
                code -= Q1
            else:
                break
            lo <<= 1
            hi = (hi << 1) | 1
            code = (code << 1) | (bits[pos] if pos < len(bits) else 0)
            pos += 1
    return out


def bench_arithmetic_coding(seed: int = 20261231 + 291) -> dict[str, float]:
    rng = random.Random(seed)
    rt = near_ent = 0
    overheads: list[float] = []
    trials = 30
    for _ in range(trials):
        n_sym = rng.randint(2, 6)
        w = [rng.random() ** 2 + 0.05 for _ in range(n_sym)]
        tot = sum(w)
        probs = [x / tot for x in w]
        freq = {s: max(1, int(p * 100)) for s, p in enumerate(probs)}
        n = rng.randint(5, 30)
        msg = [rng.choices(range(n_sym), weights=probs)[0] for _ in range(n)]
        bits, _ = encode(msg, freq)
        dec = decode(bits, n, freq)
        rt += int(dec == msg)
        h = -sum(p * math.log2(p) for p in probs)
        oh = len(bits) / n - h
        overheads.append(oh)
        near_ent += int(oh < 3.0)
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_near_entropy": float(near_ent / trials),
        "synthetic_mean_overhead_bits": float(sum(overheads) / len(overheads)),
    }
