"""Busy beaver Σ(n) exact values by enumeration of n-state 2-symbol TMs (SYNTHETIC)."""

from __future__ import annotations

import itertools


def run_tm(
    tm: dict[tuple[int, int], tuple[int, int, int]], max_steps: int = 200
) -> tuple[int, int]:
    """Run TM (state,symbol)->(write,move±1,next). Returns (ones, steps) or (-1,-1) if no halt."""
    tape: dict[int, int] = {}
    pos, state, steps = 0, 0, 0
    while steps < max_steps:
        sym = tape.get(pos, 0)
        tr = tm.get((state, sym))
        if tr is None:
            return sum(tape.values()), steps
        w, mv, nxt = tr
        tape[pos] = w
        pos += mv
        state = nxt
        steps += 1
    return -1, -1


def enumerate_tms(n_states: int):
    """All n-state 2-symbol TMs with explicit halt transition (state -1)."""
    trans = [(s, c) for s in range(n_states) for c in (0, 1)]
    opts = [(w, m, ns) for w in (0, 1) for m in (-1, 1) for ns in range(-1, n_states)]
    for combo in itertools.product(opts, repeat=len(trans)):
        yield dict(zip(trans, combo, strict=True))


def busy_beaver_sigma(n_states: int, max_steps: int = 200) -> int:
    best = 0
    for tm in enumerate_tms(n_states):
        ones, steps = run_tm(tm, max_steps)
        if steps >= 0 and ones > best:
            best = ones
    return best


def _bench_busy_beaver(seed: int = 0) -> float:
    checks = []
    # BB(1) = 1
    checks.append(busy_beaver_sigma(1) == 1)
    # direct: 1-state TM (0,0)->(1,1,-1) writes one 1 then halts
    tm = {(0, 0): (1, 1, -1), (0, 1): (0, 1, 0)}
    ones, steps = run_tm(tm)
    checks.append(ones == 1 and steps == 1)
    # non-halting TM returns -1
    tm_loop = {(0, 0): (0, 1, 0), (0, 1): (0, 1, 0)}
    checks.append(run_tm(tm_loop, 50) == (-1, -1))
    return sum(checks) / len(checks)


def bench_busy_beaver(seed: int = 0) -> dict[str, float]:
    return {"synthetic_busy_beaver": _bench_busy_beaver(seed)}
