"""Termination via affine ranking functions (synthetic) (SYNTHETIC).

For simple integer loops, synthesizes an affine ranking function
``R(s) = w·s + c`` such that (i) R >= 0 on all states and
(ii) R strictly decreases across every transition. Existence decided
by LP feasibility over the bounded state abstraction; verdict
cross-checked by bounded execution with fuel.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import linprog

State1 = int
State2 = tuple[int, int]


def _mk_loop(kind: str) -> tuple[list, Callable, set]:
    if kind == "countdown":
        # while x>0: x:=x-1
        states: list = list(range(0, 12))
        nxt: Callable = lambda s: [s - 1] if s > 0 else []  # noqa: E731
        return states, nxt, {10}
    # while x<10 and y>0: x:=x+2; y:=y-1
    states = [(x, y) for x in range(0, 12) for y in range(0, 6)]
    nxt = lambda s: [(s[0] + 2, s[1] - 1)] if s[0] < 10 and s[1] > 0 else []  # noqa: E731
    return states, nxt, {(0, 5)}


def _synth_rank(states: list, nxt: Callable) -> dict[str, object]:
    """Find w (dim) and c s.t. w·s+c >= 0 and w·s' < w·s for all s->s'."""
    pts = np.asarray([s if isinstance(s, tuple) else (s,) for s in states], dtype=float)
    dim = pts.shape[1]
    rows_strict = []
    rows_nonneg = []
    for i, s in enumerate(states):
        sp = pts[i]
        for t in nxt(s):
            tp = np.asarray(t if isinstance(t, tuple) else (t,), dtype=float)
            rows_strict.append(np.append(sp - tp, 0.0))
        rows_nonneg.append(np.append(sp, 1.0))
    a_strict = np.asarray(rows_strict) if rows_strict else np.zeros((0, dim + 1))
    a_nn = np.asarray(rows_nonneg)
    A_ub = -np.vstack([a_strict, a_nn])
    b_ub = np.concatenate([-np.ones(len(a_strict)), np.zeros(len(a_nn))])
    res = linprog(np.zeros(dim + 1), A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method="highs")
    term = res.status == 0
    w = res.x[:dim] if term else np.zeros(dim)
    c = float(res.x[dim]) if term else 0.0
    return {"terminates": term, "w": w, "c": c}


def _bounded_term(init: set, nxt: Callable, fuel: int = 400) -> bool:
    for s0 in init:
        stack = [s0]
        steps = 0
        while stack and steps < fuel:
            s = stack.pop()
            stack.extend(nxt(s))
            steps += 1
        if steps >= fuel:
            return False
    return True


def bench_ranking_function(seed: int = 20261231 + 226) -> dict[str, float]:
    agree = 0
    details = []
    for kind in ("countdown", "twovar"):
        states, nxt, init = _mk_loop(kind)
        r = _synth_rank(states, nxt)
        oracle = _bounded_term(init, nxt)
        ok = bool(r["terminates"]) == oracle
        agree += int(ok)
        details.append(float(bool(r["terminates"])))

    # nonterminating cycle: x := (x+1) mod 6 — no ranking function exists
    cyc_states = list(range(6))
    cyc_nxt: Callable = lambda s: [(s + 1) % 6]  # noqa: E731
    r3 = _synth_rank(cyc_states, cyc_nxt)
    seen: set[int] = set()
    s = 0
    nonterm = False
    for _ in range(50):
        if s in seen:
            nonterm = True
            break
        seen.add(s)
        s = cyc_nxt(s)[0]
    agree += int((not bool(r3["terminates"])) == nonterm)
    return {
        "synthetic_agree": float(agree / 3),
        "synthetic_countdown_term": details[0],
        "synthetic_twovar_term": details[1],
        "synthetic_cycle_rejected": float(not bool(r3["terminates"])),
    }
