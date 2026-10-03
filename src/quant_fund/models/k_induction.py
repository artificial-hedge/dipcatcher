"""k-induction safety checker (synthetic).

Proves a state predicate ``phi`` is an invariant of a small integer
transition system by the classic two-obligation scheme:

  base    : phi holds at every state reached in 0..k-1 steps
  inductive : for every trajectory of length k starting anywhere,
              phi(s_0..s_{k-1}) => phi(s_k)

States live in a bounded box (integer counters clipped to
[-B, B]) so both obligations are decided by exhaustive enumeration —
this is the textbook semantics of k-induction on a finite-state
abstraction, not an approximation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

State = tuple[int, int]


class _Sys(NamedTuple):
    states: list[State]
    nxt: Callable[[State], list[State]]
    phi: Callable[[State], bool]
    init: Callable[[State], bool]


def _mk_guard_loop() -> _Sys:
    """Synthetic program: x+=1 while x<y, y stable (guarded counter)."""

    # transitions on (x, y) with x,y in [-4, 8]
    def nxt(s: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = s
        out = []
        if x < y:
            out.append((min(8, x + 1), y))
        else:
            out.append((x, y))
        return out

    states = [(x, y) for x in range(-4, 9) for y in range(-4, 9)]
    return _Sys(
        states=states,
        nxt=nxt,
        phi=lambda s: s[0] <= max(0, s[1]) + 1,
        init=lambda s: s[0] <= s[1],
    )


def _reachable(states: list[State], nxt, init) -> set[State]:
    reach: set[tuple[int, int]] = set()
    frontier = [s for s in states if init(s)]
    while frontier:
        s = frontier.pop()
        if s in reach:
            continue
        reach.add(s)
        frontier.extend(t for t in nxt(s) if t not in reach)
    return reach


def _k_induction(sys: _Sys, k: int) -> dict[str, bool | int]:
    states = sys.states
    nxt, phi, init = sys.nxt, sys.phi, sys.init

    # base: phi holds along every init-trajectory prefix of length < k
    base_ok = True
    for s0 in states:
        if not init(s0):
            continue
        traj = [s0]
        for _ in range(k - 1):
            succ = nxt(traj[-1])
            if not succ:
                break
            traj.append(succ[0])
        if not all(phi(t) for t in traj):
            base_ok = False
            break

    # inductive: phi closed under k-step suffix from arbitrary start
    ind_ok = True
    for s0 in states:
        # enumerate all trajectories of length <= k from s0 (loop-free system
        # picks the deterministic successor; here nxt returns all options)
        trajs = [[s0]]
        for _ in range(k):
            trajs = [t + [u] for t in trajs for u in nxt(t[-1])]
            if not trajs:
                break
        for t in trajs:
            if len(t) == k + 1 and all(phi(u) for u in t[:-1]) and not phi(t[-1]):
                ind_ok = False
                break
        if not ind_ok:
            break

    reach = _reachable(states, nxt, init)
    safe = all(phi(s) for s in reach)
    return {"base_ok": base_ok, "ind_ok": ind_ok, "safe": safe, "reach": len(reach)}


def bench_k_induction(seed: int = 20261231 + 221) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sys = _mk_guard_loop()
    r1 = _k_induction(sys, k=1)
    r3 = _k_induction(sys, k=3)

    # randomized perturbation: shrink state box, oracle by BFS
    agree = 0
    trials = 25
    for _ in range(trials):
        lo = int(rng.integers(-6, -2))
        sys2 = _Sys(
            states=[(x, y) for x in range(lo, 9) for y in range(lo, 9)],
            nxt=sys.nxt,
            phi=sys.phi,
            init=sys.init,
        )
        rr = _k_induction(sys2, k=2)
        agree += int((rr["base_ok"] and rr["ind_ok"]) == rr["safe"])
    return {
        "synthetic_base_ok_k1": float(r1["base_ok"]),
        "synthetic_ind_ok_k1": float(r1["ind_ok"]),
        "synthetic_ind_ok_k3": float(r3["ind_ok"]),
        "synthetic_safe": float(r3["safe"]),
        "synthetic_agree": float(agree / trials),
        "synthetic_reach": float(r3["reach"]),
    }
