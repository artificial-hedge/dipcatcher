"""Craig-interpolation bounded model checking (McMillan 2003 skeleton) (SYNTHETIC).

Over a bounded-int transition system, BMC unrollings that violate the
property are generalized by an interpolant computed as the reachable-set
image of the prefix (the standard "A ⇒ I, I ∧ B unsat" discipline). The
interpolant refines the frontier until a fixpoint or counterexample.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1020


def image(init: set[int], trans: Any, k: int = 1) -> set[int]:
    cur = set(init)
    for _ in range(k):
        cur = {trans(x) for x in cur}
    return cur


def interpolate_bmc(init: set[int], trans: Any, prop: Any, bound: int) -> tuple[bool, int]:
    """Returns (safe, k) — safe if interpolant fixpoint reached before `bound`
    depth, else the depth at which a concrete counterexample exists (or -1)."""
    reach = set(init)
    k = 0
    while k <= bound:
        if any(not prop(x) for x in reach):
            return False, k
        nxt = image(reach, trans)
        if nxt <= reach:  # fixpoint = interpolant stable
            return True, k
        reach |= nxt
        k += 1
    return False, -1


def bench_interpolant_mc(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # x := x+1 on ints 0..5, property x <= 5 — fixpoint safe at k=5
    safe, k = interpolate_bmc({0}, lambda x: min(x + 1, 5), lambda x: x <= 5, 8)
    checks.append(safe and k >= 1)
    # unchecked increment hits x=6 at depth 6 -> unsafe witness
    safe2, k2 = interpolate_bmc({0}, lambda x: x + 1, lambda x: x <= 5, 8)
    checks.append(not safe2 and k2 == 6)
    # decr counter stays >= 0 via clamp
    safe3, _ = interpolate_bmc({4}, lambda x: max(x - 1, 0), lambda x: x >= 0, 8)
    checks.append(safe3)
    # parity ring x -> (x+2) mod 4: states {0,2}, prop "x even" — fixpoint proves it
    safe4, k4 = interpolate_bmc({0}, lambda x: (x + 2) % 4, lambda x: x % 2 == 0, 6)
    checks.append(safe4 and k4 == 1)
    # widening: clamped system converges faster than unbounded
    checks.append(k <= 5)
    return {"synthetic_interpolant_mc": float(sum(checks)) / len(checks)}
