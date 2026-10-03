"""Counterexample-guided abstraction refinement — CEGAR (synthetic).

Verifies a safety property over an integer program via predicate
abstraction: start with the coarsest abstraction (empty predicate
set), model-check the Boolean abstraction, and on a spurious
counterexample add a new predicate (from a small candidate pool)
that rules it out. Iterates until the abstract model is safe or a
concrete counterexample is confirmed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

Pred = tuple[str, Callable[[int], bool]]


class _Sys(NamedTuple):
    states: list[int]
    nxt: Callable[[int], list[int]]
    init: set[int]
    bad: set[int]
    preds: list[Pred]


class _Abs(NamedTuple):
    states: set[tuple[bool, ...]]
    nxt: dict[tuple[bool, ...], set[tuple[bool, ...]]]
    init: set[tuple[bool, ...]]
    bad: set[tuple[bool, ...]]


def _mk_incr_prog() -> _Sys:
    """while x < 5: x := x + d  with d nondet in {1,2}; safe: x <= 6."""
    states = list(range(-2, 10))

    def nxt(s: int) -> list[int]:
        return [s + d for d in (1, 2) if s < 5 and s + d in states]

    return _Sys(
        states=states,
        nxt=nxt,
        init={0},
        bad={s for s in states if s > 6},
        preds=[
            ("x<=4", lambda s: s <= 4),
            ("x<=5", lambda s: s <= 5),
            ("x>=0", lambda s: s >= 0),
            ("x<=6", lambda s: s <= 6),
        ],
    )


def _abstract_sys(sys: _Sys, pred_idx: list[int]) -> _Abs:
    """Boolean (existential) abstraction over the chosen predicates."""
    preds = [sys.preds[i] for i in pred_idx]
    sig_of = {s: tuple(fn(s) for _, fn in preds) for s in sys.states}
    abstract_states = set(sig_of.values())
    nxt_abs: dict[tuple[bool, ...], set[tuple[bool, ...]]] = {a: set() for a in abstract_states}
    for s in sys.states:
        for t in sys.nxt(s):
            nxt_abs[sig_of[s]].add(sig_of[t])
    init_abs = {sig_of[s] for s in sys.init}
    bad_abs = {sig_of[s] for s in sys.bad}
    return _Abs(states=abstract_states, nxt=nxt_abs, init=init_abs, bad=bad_abs)


def _abs_reach(absys: _Abs) -> set[tuple[bool, ...]]:
    reach: set[tuple[bool, ...]] = set()
    frontier = list(absys.init)
    while frontier:
        a = frontier.pop()
        if a in reach:
            continue
        reach.add(a)
        frontier.extend(b for b in absys.nxt.get(a, ()) if b not in reach)
    return reach


def _concrete_reach(sys: _Sys) -> set[int]:
    reach: set[int] = set()
    frontier = list(sys.init)
    while frontier:
        s = frontier.pop()
        if s in reach:
            continue
        reach.add(s)
        frontier.extend(t for t in sys.nxt(s) if t not in reach)
    return reach


def cegar(sys: _Sys) -> dict[str, int | bool]:
    chosen: list[int] = []
    steps = 0
    for _ in range(len(sys.preds) + 1):
        absys = _abstract_sys(sys, chosen)
        reach_abs = _abs_reach(absys)
        steps += 1
        if not reach_abs & absys.bad:
            return {"safe": True, "steps": steps, "preds": len(chosen)}
        remaining = [i for i in range(len(sys.preds)) if i not in chosen]
        if not remaining:
            break
        chosen.append(remaining[0])
    conc = _concrete_reach(sys)
    return {
        "safe": not bool(conc & sys.bad),
        "steps": steps,
        "preds": len(chosen),
    }


def bench_cegar_loop(seed: int = 20261231 + 227) -> dict[str, float]:
    sys = _mk_incr_prog()
    r = cegar(sys)
    conc = _concrete_reach(sys)
    oracle = not bool(conc & sys.bad)
    # variant where bad IS reachable (tighten bad to x > 4)
    sys2 = sys._replace(bad={s for s in sys.states if s > 4})
    r2 = cegar(sys2)
    conc2 = _concrete_reach(sys2)
    oracle2 = not bool(conc2 & sys2.bad)
    return {
        "synthetic_safe": float(bool(r["safe"])),
        "synthetic_oracle": float(oracle),
        "synthetic_agree": float(bool(r["safe"]) == oracle),
        "synthetic_safe2": float(bool(r2["safe"])),
        "synthetic_oracle2": float(oracle2),
        "synthetic_agree2": float(bool(r2["safe"]) == oracle2),
        "synthetic_refinements": float(int(r["preds"])),
    }
