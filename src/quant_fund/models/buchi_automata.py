"""Büchi automata: lasso acceptance on ultimately-periodic words + emptiness (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 551


class Buchi:
    def __init__(self) -> None:
        self.trans: dict[tuple[int, str], list[int]] = {}
        self.start = 0
        self.final: set[int] = set()

    def step(self, states: set[int], sym: str) -> set[int]:
        out: set[int] = set()
        for s in states:
            out.update(self.trans.get((s, sym), []))
        return out


def accepts_up(aut: Buchi, prefix: list[str], cycle: list[str]) -> bool:
    """Acceptance on prefix·cycle^ω: a final (phase,state) pair recurs."""
    states: set[int] = {aut.start}
    for s in prefix:
        states = aut.step(states, s)
    if not states:
        return False
    # phase graph over (state, phase): single-run reachability
    n_ph = len(cycle)
    nq = (
        max(
            aut.final | aut.start.__class__(s for s in aut.trans)
            if False
            else {aut.start}
            | {k[0] for k in aut.trans}
            | {t for vs in aut.trans.values() for t in vs}
        )
        + 1
    )
    bound = 2 * nq * n_ph + 4
    # BFS from each state in `states` (at phase 0): find reachable final node
    # lying on a cycle.
    for s0 in states:
        # forward reachability from (s0, 0)
        front = {(s0, 0)}
        reached: set[tuple[int, int]] = set()
        for _ in range(bound * n_ph):
            nxt: set[tuple[int, int]] = set()
            for st, ph in front:
                for t in aut.trans.get((st, cycle[ph]), []):
                    nxt.add((t, (ph + 1) % n_ph))
            front = nxt - reached
            reached |= nxt
            if not front:
                break
        for f, ph in reached:
            if f not in aut.final:
                continue
            # does (f, ph) reach itself in ≥1 step?
            front2 = {
                t2 for t in aut.trans.get((f, cycle[ph]), []) for t2 in [(t, (ph + 1) % n_ph)]
            }
            seen2: set[tuple[int, int]] = set()
            for _ in range(bound * n_ph):
                if (f, ph) in front2:
                    return True
                seen2 |= front2
                nxt2: set[tuple[int, int]] = set()
                for st, p2 in front2:
                    for t in aut.trans.get((st, cycle[p2]), []):
                        nxt2.add((t, (p2 + 1) % n_ph))
                front2 = nxt2 - seen2
                if not front2:
                    break
    return False


def bench_buchi_automata(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # automaton: accept words with infinitely many 'a'
    a1 = Buchi()
    a1.start = 0
    a1.trans = {(0, "a"): [1], (0, "b"): [0], (1, "a"): [1], (1, "b"): [0]}
    a1.final = {1}
    n = 60
    hits = 0
    for _ in range(n):
        cyc = [rng.choice(["a", "b"]) for _ in range(rng.randint(1, 4))]
        pref = [rng.choice(["a", "b"]) for _ in range(rng.randint(0, 3))]
        truth = "a" in cyc  # infinitely many a iff cycle contains a
        hits += int(accepts_up(a1, pref, cyc) == truth)
    # second: infinitely many alternations — accept if cycle has both a and b
    a2 = Buchi()
    a2.start = 0
    a2.trans = {
        (0, "a"): [1],
        (0, "b"): [0],
        (1, "a"): [1],
        (1, "b"): [2],
        (2, "a"): [1],
        (2, "b"): [2],
    }
    a2.final = {2}
    hits2 = 0
    for _ in range(n):
        cyc = [rng.choice(["a", "b"]) for _ in range(rng.randint(1, 4))]
        pref = [rng.choice(["a", "b"]) for _ in range(rng.randint(0, 2))]
        w = [str(x) for x in pref + cyc + cyc]
        truth = any(a == "a" and b == "b" for a, b in zip(w, w[1:], strict=False))
        hits2 += int(accepts_up(a2, pref, cyc) == truth)
    return {
        "synthetic_inf_a": float(hits / n),
        "synthetic_inf_both": float(hits2 / n),
    }
