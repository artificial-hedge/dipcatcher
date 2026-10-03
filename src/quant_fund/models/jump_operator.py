"""Turing jump: K = halting set of plain TMs, verified c.e.-not-decidable toy (SYNTHETIC)."""

from __future__ import annotations


def run_tm(table: dict[tuple[int, int], tuple[int, int, int]], steps: int = 200) -> int:
    """Deterministic 1-tape TM; returns steps taken if halts else -1."""
    state, head = 0, 0
    tape: dict[int, int] = {}
    for t in range(steps):
        sym = tape.get(head, 0)
        key = (state, sym)
        if key not in table:
            return t
        state, sym, move = table[key]
        tape[head] = sym
        head += move
    return -1


def diagonal_set(programs: list[dict], steps: int = 200) -> frozenset[int]:
    """K = {e : phi_e(e) halts} (with input ignored / input = e written on tape)."""
    out = set()
    for e, tbl in enumerate(programs):
        if run_tm(tbl, steps) >= 0:
            out.add(e)
    return frozenset(out)


def _bench_jump_operator(seed: int = 0) -> float:
    checks = []
    # run_tm halts when (state,sym) missing from table; empty table halts at t=0
    loop_tbl = {(0, 0): (0, 0, 1), (0, 1): (0, 1, 1)}  # moves right forever
    flip_tbl = {
        (0, 0): (1, 1, 1),
        (1, 0): (2, 1, 0),
    }  # writes 1 then reads state1: (1,0) missing -> halt t=2
    checks.append(run_tm({}, 10) == 0)
    checks.append(run_tm(loop_tbl, 50) == -1)
    checks.append(run_tm(flip_tbl, 50) == 2)
    K = diagonal_set([{}, loop_tbl, flip_tbl])
    checks.append(frozenset({0, 2}) == K)
    # K is undecidable bounded-toy: complement membership can't be verified by bounded run
    checks.append(1 not in K)
    return float(sum(checks) / len(checks))


def bench_jump_operator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jump_operator": _bench_jump_operator(seed)}
