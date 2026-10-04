"""Ground Horn-clause (CHC-style) solver by forward-chaining saturation.

Clauses: head <- body_1 ∧ ... ∧ body_n over a finite fact domain; queries
ask whether a fact is derivable. Encodes loop-invariant CHCs over a bounded
integer domain — the pattern used by CHC solvers (Z3 spacer, Eldarica) at
the ground level.
"""

from __future__ import annotations

_SEED = 20261231 + 1019

Clause = tuple[list[int], int]  # (body fact ids, head fact id)


def saturate(facts0: set[int], clauses: list[Clause]) -> set[int]:
    facts = set(facts0)
    changed = True
    while changed:
        changed = False
        for body, head in clauses:
            if head not in facts and all(b in facts for b in body):
                facts.add(head)
                changed = True
    return facts


def bench_horn_clauses(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # encode inv(x) for x=0..10 under loop x:=0; while x<10: x++
    # facts: inv(x) for x in 0..10, safe
    N = 10

    def inv(x: int) -> int:
        return x  # fact id = x

    SAFE = 100
    UNSAFE = 101
    clauses: list[Clause] = [
        ([], inv(0)),  # init
    ]
    clauses += [([inv(x)], inv(x + 1)) for x in range(N)]  # induct step
    clauses.append(([inv(N)], SAFE))  # exit implies safe
    facts = saturate(set(), clauses)
    checks.append(SAFE in facts)
    checks.append(UNSAFE not in facts)
    checks.append(inv(N) in facts)
    # broken step clause (skip x=4) loses inductiveness -> UNSAFE-safe fails
    clauses2: list[Clause] = [([], inv(0))] + [([inv(x)], inv(x + 1)) for x in range(N) if x != 4]
    clauses2.append(([inv(N)], SAFE))
    facts2 = saturate(set(), clauses2)
    checks.append(SAFE not in facts2)
    checks.append(inv(4) in facts2)
    # transitivity through derived facts: f0->f1->f2->GOAL
    c3: list[Clause] = [([], 0), ([0], 1), ([1], 2), ([2], 3)]
    checks.append(3 in saturate(set(), c3))
    return {"synthetic_horn_clauses": float(sum(checks)) / len(checks)}
