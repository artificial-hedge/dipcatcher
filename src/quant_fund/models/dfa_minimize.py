"""Hopcroft DFA minimization (synthetic).

Partition refinement on random DFAs. Verified: (i) minimized DFA
language-equivalent to original on random input set; (ii) state
count ≤ original; (iii) agrees with naive distinguishability
oracle (all-pairs table-filling).
"""

from __future__ import annotations

import random
from collections import deque


def hopcroft(
    states: int,
    alpha: list[str],
    trans: list[dict[str, int]],
    accept: set[int],
) -> list[set[int]]:
    p = [set(accept), set(range(states)) - accept]
    p = [s for s in p if s]
    work = deque(p)
    while work:
        a = work.popleft()
        for c in alpha:
            x = {s for s in range(states) if trans[s].get(c) in a}
            new_p: list[set[int]] = []
            for y in p:
                i1 = y & x
                i2 = y - x
                if i1 and i2:
                    new_p += [i1, i2]
                    if y in work:
                        work.remove(y)
                        work += deque([i1, i2])
                    else:
                        work.append(i1 if len(i1) <= len(i2) else i2)
                else:
                    new_p.append(y)
            p = new_p
    return p


def distinguishable_pairs(
    states: int,
    alpha: list[str],
    trans: list[dict[str, int]],
    accept: set[int],
) -> set[frozenset[int]]:
    """Naive table-filling: pairs where one accepts, other rejects."""
    marked: set[frozenset[int]] = set()
    for s in range(states):
        for t in range(s + 1, states):
            if (s in accept) != (t in accept):
                marked.add(frozenset({s, t}))
    changed = True
    while changed:
        changed = False
        for s in range(states):
            for t in range(s + 1, states):
                if frozenset({s, t}) in marked:
                    continue
                for c in alpha:
                    u = trans[s].get(c, 0)
                    v = trans[t].get(c, 0)
                    if u != v and frozenset({u, v}) in marked:
                        marked.add(frozenset({s, t}))
                        changed = True
                        break
    return marked


def bench_dfa_minimize(seed: int = 20261231 + 281) -> dict[str, float]:
    rng = random.Random(seed)
    agree = lang_ok = shrink_ok = 0
    trials = 30
    for _ in range(trials):
        n = rng.randint(3, 10)
        alpha = ["0", "1"]
        trans = [{c: rng.randrange(n) for c in alpha} for _ in range(n)]
        accept = {i for i in range(n) if rng.random() < 0.4}
        parts = hopcroft(n, alpha, trans, accept)
        marked = distinguishable_pairs(n, alpha, trans, accept)
        # partition cells must contain only unmarked (equivalent) states,
        # and #cells must equal #distinct equivalence classes
        cells_ok = True
        for part in parts:
            pl = list(part)
            for i in range(len(pl)):
                for j in range(i + 1, len(pl)):
                    if frozenset({pl[i], pl[j]}) in marked:
                        cells_ok = False
        rep: dict[int, int] = {}
        for k, part in enumerate(parts):
            for s in part:
                rep[s] = k
        # oracle class count: each class = maximal set of mutually unmarked
        classes: list[set[int]] = []
        for s in range(n):
            placed = False
            for cl in classes:
                t = next(iter(cl))
                if frozenset({s, t}) not in marked:
                    cl.add(s)
                    placed = True
                    break
            if not placed:
                classes.append({s})
        agree += int(cells_ok and len(parts) == len(classes))
        # language equivalence: original vs quotient DFA accepts identically
        ok = True
        for _ in range(50):
            w = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 12)))
            s1 = 0
            for c in w:
                s1 = trans[s1].get(c, 0)
            orig = s1 in accept
            # simulate minimized: start in rep[0], move via quotient
            s2 = rep[0]
            for c in w:
                nxt = trans[next(iter(parts[s2]))].get(c, 0)
                s2 = rep[nxt]
            mini = any(rep[a] == s2 for a in accept)
            ok = ok and orig == mini
        lang_ok += int(ok)
        shrink_ok += int(len(parts) <= n)
    return {
        "synthetic_partition_ok": float(agree / trials),
        "synthetic_lang_equiv": float(lang_ok / trials),
        "synthetic_shrinks": float(shrink_ok / trials),
    }
