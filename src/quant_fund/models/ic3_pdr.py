"""IC3 / property-directed reachability — backward PDR (synthetic) (SYNTHETIC).

On a small finite-state transition system, decides safety of
``init`` vs ``bad`` by backward reachability: computes the iterated
pre-image of ``bad`` (P_0 = bad, P_{i+1} = P_i ∪ pre(P_i)) and
declares unsafe iff init intersects some P_i. Fixpoint terminates
the computation. This is the standard backward-IC3 view — dual to
forward BFS — so agreement with forward enumeration is a genuine
soundness check of both directions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

State = tuple[int, int, int]


class _Sys(NamedTuple):
    states: list[State]
    nxt: Callable[[State], list[State]]
    init: set[State]
    bad: set[State]


def _mk_shift_sys() -> _Sys:
    """Synthetic bit-register: (a,b,c) in {0,1}^3, shift right each step
    with a fresh nondeterministic input bit; init a=b=c=0."""
    states = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    def nxt(s: State) -> list[State]:
        a, b, c = s
        return [(nb, a, b) for nb in (0, 1)]

    init = {(0, 0, 0)}
    bad = {s for s in states if s[0] == 1 and s[1] == 1 and s[2] == 0}
    return _Sys(states=states, nxt=nxt, init=init, bad=bad)


def _bfs_reach(sys: _Sys) -> set[State]:
    reach: set[State] = set()
    frontier = list(sys.init)
    while frontier:
        s = frontier.pop()
        if s in reach:
            continue
        reach.add(s)
        frontier.extend(t for t in sys.nxt(s) if t not in reach)
    return reach


def _pre_image(sys: _Sys, target: set[State]) -> set[State]:
    return {s for s in sys.states if any(t in target for t in sys.nxt(s))}


def ic3(sys: _Sys) -> dict[str, int | bool]:
    """Backward PDR: iterated pre-image until fixpoint or init hit."""
    p = set(sys.bad)
    depth = 0
    while True:
        if p & sys.init:
            return {"safe": False, "depth": depth}
        new = p | _pre_image(sys, p)
        depth += 1
        if new == p:
            return {"safe": True, "depth": depth}
        p = new


def bench_ic3_pdr(seed: int = 20261231 + 222) -> dict[str, float]:
    sys = _mk_shift_sys()
    r = ic3(sys)
    reach = _bfs_reach(sys)
    oracle_safe = not bool(reach & sys.bad)

    # second instance: bad = all-ones — also reachable, still must agree
    sys2 = sys._replace(bad={(1, 1, 1)})
    r2 = ic3(sys2)
    reach2 = _bfs_reach(sys2)
    oracle2 = not bool(reach2 & sys2.bad)
    return {
        "synthetic_ic3_safe": float(bool(r["safe"])),
        "synthetic_oracle_safe": float(oracle_safe),
        "synthetic_agree": float(bool(r["safe"]) == oracle_safe),
        "synthetic_ic3_safe2": float(bool(r2["safe"])),
        "synthetic_oracle_safe2": float(oracle2),
        "synthetic_agree2": float(bool(r2["safe"]) == oracle2),
        "synthetic_depth": float(int(r["depth"])),
    }
