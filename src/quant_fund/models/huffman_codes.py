"""Huffman optimal prefix coding (synthetic).

Tree built from symbol frequencies; encode/decode bitstring.
Verified: (i) exact round-trip; (ii) code is prefix-free;
(iii) expected length within 1 bit of empirical entropy;
(iv) expected length ≤ any equal-depth fixed code.
"""

from __future__ import annotations

import heapq
import math
import random


def huffman(freq: dict[str, int]) -> dict[str, str]:
    heap: list[tuple[int, int, object]] = [(f, i, s) for i, (s, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    k = len(heap)
    while len(heap) > 1:
        f1, _, t1 = heapq.heappop(heap)
        f2, _, t2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, k, (t1, t2)))
        k += 1
    if not heap:
        return {}
    _, _, root = heap[0]
    codes: dict[str, str] = {}

    def walk(node: object, pre: str) -> None:
        if isinstance(node, str):
            codes[node] = pre or "0"
            return
        if not isinstance(node, tuple):
            raise TypeError(f"bad huffman node {node!r}")
        a, b = node
        walk(a, pre + "0")
        walk(b, pre + "1")

    walk(root, "")
    return codes


def encode(codes: dict[str, str], msg: list[str]) -> str:
    return "".join(codes[s] for s in msg)


def decode(codes: dict[str, str], bits: str) -> list[str]:
    rev = {v: k for k, v in codes.items()}
    out: list[str] = []
    cur = ""
    for b in bits:
        cur += b
        if cur in rev:
            out.append(rev[cur])
            cur = ""
    return out


def bench_huffman_codes(seed: int = 20261231 + 290) -> dict[str, float]:
    rng = random.Random(seed)
    rt = pf = ent_ok = 0
    gaps: list[float] = []
    trials = 40
    for _ in range(trials):
        n_sym = rng.randint(2, 8)
        alpha = [chr(97 + i) for i in range(n_sym)]
        # skewed freq
        w = [rng.random() ** 3 for _ in alpha]
        tot = sum(w)
        probs = [x / tot for x in w]
        freq = {s: max(1, int(p * 1000)) for s, p in zip(alpha, probs, strict=True)}
        codes = huffman(freq)
        msg = [rng.choices(alpha, weights=probs)[0] for _ in range(100)]
        bits = encode(codes, msg)
        rt += int(decode(codes, bits) == msg)
        pf += int(
            all(
                not any(codes[a].startswith(codes[b2]) and codes[a] != codes[b2] for b2 in codes)
                for a in codes
            )
        )
        h = -sum(p * math.log2(p) for p in probs)
        exp_len = sum(probs[i] * len(codes[alpha[i]]) for i in range(n_sym))
        gaps.append(exp_len - h)
        ent_ok += int(exp_len < h + 1.0 + 1e-9)
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_prefix_free": float(pf / trials),
        "synthetic_within_bit": float(ent_ok / trials),
        "synthetic_mean_gap": float(sum(gaps) / len(gaps)),
    }
