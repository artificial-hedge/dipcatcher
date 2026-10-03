"""SYNTHETIC Mealy↔Moore transducer equivalence.

Runs random Mealy machines, converts each to an equivalent Moore machine
(state×output lifting), and verifies identical output streams on random
input sequences.
"""

from __future__ import annotations

import random

# Mealy: delta[(q,in)] = q', out[(q,in)] = symbol
Mealy = tuple[int, dict[tuple[int, str], int], dict[tuple[int, str], str], int]
Moore = tuple[
    int,
    dict[tuple[int, str], int],
    dict[int, str],  # output per state
    int,
    str | None,  # output produced before start state (None → skip)
]


def run_mealy(m: Mealy, inputs: list[str]) -> list[str]:
    _, d, o, q = m
    out = []
    for c in inputs:
        out.append(o[(q, c)])
        q = d[(q, c)]
    return out


def run_moore(mm: Moore, inputs: list[str]) -> list[str]:
    _, d, o, q, _pre = mm
    out = []
    for c in inputs:
        q = d[(q, c)]
        out.append(o[q])
    return out


def mealy_to_moore(m: Mealy, alpha: tuple[str, ...]) -> Moore:
    """Standard conversion: Moore state = (mealy_state, output_emitted_on_entry)."""
    n, d, o, s = m
    outputs = sorted({o[k] for k in o})
    if not outputs:
        outputs = ["0"]
    idx = {(q, out): i for i, (q, out) in enumerate((q, out) for q in range(n) for out in outputs)}
    # start state: use (s, first output symbol) — entry output discarded by convention
    start = idx[(s, outputs[0])]
    dd: dict[tuple[int, str], int] = {}
    oo: dict[int, str] = {}
    for (q, out), i in idx.items():
        oo[i] = out
        for a in alpha:
            nq = d[(q, a)]
            dd[(i, a)] = idx[(nq, o[(q, a)])]
    return len(idx), dd, oo, start, None


def _rand_mealy(rng: random.Random, n: int, alpha: tuple[str, ...], outs: tuple[str, ...]) -> Mealy:
    d = {(q, a): rng.randrange(n) for q in range(n) for a in alpha}
    o = {(q, a): rng.choice(outs) for q in range(n) for a in alpha}
    return n, d, o, 0


def bench_mealy_moore(seed: int = 20261231 + 504) -> dict[str, float]:
    rng = random.Random(seed)
    alpha, outs = ("a", "b"), ("0", "1")
    eq = 0
    n = 40
    for _ in range(n):
        m = _rand_mealy(rng, rng.randrange(2, 6), alpha, outs)
        mm = mealy_to_moore(m, alpha)
        seq = [rng.choice(alpha) for _ in range(rng.randrange(1, 20))]
        eq += int(run_mealy(m, seq) == run_moore(mm, seq))
    # direct run sanity: stepwise state tracking
    m = _rand_mealy(rng, 4, alpha, outs)
    seq = [rng.choice(alpha) for _ in range(15)]
    q = m[3]
    manual = []
    for c in seq:
        manual.append(m[2][(q, c)])
        q = m[1][(q, c)]
    direct = manual == run_mealy(m, seq)
    return {
        "synthetic_conversion_equivalent": eq / n,
        "synthetic_run_correct": float(direct),
    }
