"""SYNTHETIC DFA equivalence + minimization.

Table-filling distinguishability for minimization; product-construction
emptiness oracle for equivalence.  Checks: minimized DFA ≡ original, and
perturbed DFAs are correctly detected as inequivalent.
"""

from __future__ import annotations

import random
from collections import deque


def minimize(
    n_states: int,
    delta: dict[tuple[int, str], int],
    accept: set[int],
    start: int,
    alpha: tuple[str, ...],
) -> tuple[int, dict[tuple[int, str], int], set[int], int]:
    """Partition-refinement minimization; returns canonical DFA."""
    alive = {q for q in range(n_states) if _reachable(delta, start, {q}, alpha)}
    part: list[set[int]] = [{q for q in alive if q in accept}, alive - accept]
    part = [p for p in part if p]
    changed = True
    while changed:
        changed = False
        new_part: list[set[int]] = []
        block = {q: i for i, p in enumerate(part) for q in p}
        for p in part:
            buckets: dict[tuple, set[int]] = {}
            for q in p:
                sig = tuple(block.get(delta.get((q, a), -1), -1) for a in alpha)
                buckets.setdefault(sig, set()).add(q)
            new_part.extend(buckets.values())
            if len(buckets) > 1:
                changed = True
        part = new_part
    block = {q: i for i, p in enumerate(part) for q in p}
    nd = {(block[q], a): block[delta[(q, a)]] for q in alive for a in alpha if (q, a) in delta}
    nacc = {block[q] for q in alive if q in accept}
    return len(part), nd, nacc, block[start]


def _reachable(
    delta: dict[tuple[int, str], int], start: int, targets: set[int], alpha: tuple[str, ...]
) -> bool:
    seen, dq = {start}, deque([start])
    while dq:
        q = dq.popleft()
        if q in targets:
            return True
        for a in alpha:
            nq = delta.get((q, a))
            if nq is not None and nq not in seen:
                seen.add(nq)
                dq.append(nq)
    return False


def equivalent(
    d1: tuple[int, dict[tuple[int, str], int], set[int], int],
    d2: tuple[int, dict[tuple[int, str], int], set[int], int],
    alpha: tuple[str, ...],
) -> bool:
    """Language equivalence via product DFA: disagreeing-accept reachable?"""
    _, t1, a1, s1 = d1
    _, t2, a2, s2 = d2
    seen: set[tuple[int | None, int | None]] = {(s1, s2)}
    dq: deque[tuple[int | None, int | None]] = deque([(s1, s2)])
    while dq:
        p, q = dq.popleft()
        if (p in a1) != (q in a2):
            return False
        for c in alpha:
            np_ = t1.get((p, c)) if p is not None else None
            nq = t2.get((q, c)) if q is not None else None
            if np_ is not None and nq is not None and (np_, nq) not in seen:
                seen.add((np_, nq))
                dq.append((np_, nq))
            elif (np_ is None) != (nq is None):
                # one side dies: equivalent only if survivor rejects forever —
                # treat dead as non-accepting sink and keep checking
                s_a = np_ if np_ is not None else nq
                if s_a in (a1 if np_ is not None else a2):
                    return False
                pair = (np_, nq)
                if pair not in seen:
                    seen.add(pair)
                    dq.append(pair)
    return True


def _rand_dfa(
    rng: random.Random, n: int, alpha: tuple[str, ...]
) -> tuple[int, dict[tuple[int, str], int], set[int], int]:
    delta = {(q, a): rng.randrange(n) for q in range(n) for a in alpha}
    accept = {q for q in range(n) if rng.random() < 0.3}
    return n, delta, accept, 0


def bench_dfa_equiv(seed: int = 20261231 + 503) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = ("a", "b")
    tp = 0
    n = 40
    for _ in range(n):
        d = _rand_dfa(rng, rng.randrange(3, 8), alpha)
        dm = minimize(*d, alpha)
        tp += int(equivalent((d[0], d[1], d[2], d[3]), dm, alpha))
    det = 0
    for _ in range(n):
        d = _rand_dfa(rng, 5, alpha)
        d2n, d2t, d2a, d2s = d
        d2t = dict(d2t)
        q = rng.randrange(5)
        c = rng.choice(alpha)
        d2t[(q, c)] = (d2t[(q, c)] + 1) % 5
        # oracle: brute-force enumerate strings ≤ 8 for disagreement
        diff = _diff_exists(d, (d2n, d2t, d2a, d2s), alpha, 8)
        det += int(equivalent(d, (d2n, d2t, d2a, d2s), alpha) == (not diff))
    return {
        "synthetic_minimize_equivalent": tp / n,
        "synthetic_equiv_matches_oracle": det / n,
    }


def _diff_exists(
    d1: tuple[int, dict[tuple[int, str], int], set[int], int],
    d2: tuple[int, dict[tuple[int, str], int], set[int], int],
    alpha: tuple[str, ...],
    maxlen: int,
) -> bool:
    for length in range(maxlen + 1):
        for bits in range(1 << length):
            s = [alpha[(bits >> k) & 1] for k in range(length)]
            a1 = _accepts(d1, s)
            a2 = _accepts(d2, s)
            if a1 != a2:
                return True
    return False


def _accepts(d: tuple[int, dict[tuple[int, str], int], set[int], int], s: list[str]) -> bool:
    _, t, acc, q = d
    for c in s:
        q = t.get((q, c), -1)
        if q < 0:
            return False
    return q in acc
