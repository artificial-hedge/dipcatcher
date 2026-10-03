"""Nondeterministic Büchi automaton emptiness via nested DFS.

NBA: states, alphabet-agnostic transitions, initial set, accepting set.
Language nonempty iff some reachable cycle contains an accept state —
checked by classic nested DFS (outer finds accept, inner confirms a path
back to it).
"""

from __future__ import annotations

_SEED = 20261231 + 1043


def is_empty(nstates: int, init: set[int], acc: set[int], succ: dict[int, list[int]]) -> bool:
    """True iff L(A) = ∅ — no reachable accept-state cycle."""
    return not _has_accepting_cycle(nstates, init, acc, succ)


def _has_accepting_cycle(n: int, init: set[int], acc: set[int], succ: dict[int, list[int]]) -> bool:
    outer: set[int] = set()
    for s in init:
        if s in outer:
            continue
        stack = [(s, iter(succ.get(s, [])))]
        outer.add(s)
        while stack:
            v, it = stack[-1]
            adv = False
            for u in it:
                if u in acc and u == v:
                    return True
                if u in acc and _reaches(u, v, acc, succ, set()):
                    return True
                if u not in outer:
                    outer.add(u)
                    stack.append((u, iter(succ.get(u, []))))
                    adv = True
                    break
            if not adv:
                stack.pop()
    return False


def _reaches(src: int, dst: int, acc: set[int], succ: dict[int, list[int]], seen: set[int]) -> bool:
    """Path src~>dst of length >=1 (nontrivial cycle check)."""
    wl = [src]
    seen = set(seen) | {src}
    while wl:
        v = wl.pop()
        for u in succ.get(v, []):
            if u == dst:
                return True
            if u not in seen:
                seen.add(u)
                wl.append(u)
    return False


def bench_nba_emptiness(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # s0 -> s1(acc) -> s1: nonempty (s1 self loop accepting)
    checks.append(not is_empty(2, {0}, {1}, {0: [1], 1: [1]}))
    # s0 -> s1(acc) dead end: nonempty? Büchi needs infinite run -> empty
    checks.append(is_empty(2, {0}, {1}, {0: [1], 1: []}))
    # reachable cycle with no accept: empty
    checks.append(is_empty(3, {0}, {2}, {0: [1], 1: [1]}))
    # accept state unreachable: empty
    checks.append(is_empty(3, {0}, {2}, {0: [1], 1: [1], 2: [2]}))
    # accept on a non-trivial cycle reachable from init: nonempty
    checks.append(not is_empty(3, {0}, {1}, {0: [1], 1: [2], 2: [1]}))
    return {"synthetic_nba_emptiness": float(sum(checks)) / len(checks)}
