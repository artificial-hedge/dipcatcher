"""Unit propagation engine: closure under unit clauses, counts
propagated assignments and detects forced-variable fraction on a
clause set generated with a planted satisfying assignment.
"""

import numpy as np


def _propagate(clauses: list[list[int]], n_var: int) -> tuple[dict[int, bool], int]:
    a: dict[int, bool] = {}
    props = 0
    changed = True
    while changed:
        changed = False
        for c in clauses:
            sat = any(a.get(abs(lit)) == (lit > 0) for lit in c)
            if sat:
                continue
            un = [lit for lit in c if abs(lit) not in a]
            if len(un) == 0:
                return a, props
            if len(un) == 1:
                v = abs(un[0])
                a[v] = un[0] > 0
                props += 1
                changed = True
    return a, props


def bench_unit_propagation(seed: int = 5905) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_var = 12
    plant = {v: bool(rng.random() < 0.5) for v in range(1, n_var + 1)}
    # clauses satisfied under plant; add unit clauses to force a prefix
    clauses = []
    for _ in range(30):
        vs = rng.choice(n_var, 3, replace=False) + 1
        lits = [int(v) * (1 if plant[int(v)] else -1) for v in vs]
        # flip signs so plant satisfies (at least one lit true)
        if not any(plant[abs(lit)] == (lit > 0) for lit in lits):
            lits[0] = abs(lits[0]) * (1 if plant[abs(lits[0])] else -1)
        clauses.append(lits)
    forced = [int(v) * (1 if plant[v] else -1) for v in range(1, 5)]
    clauses += [[f] for f in forced]
    a, props = _propagate(clauses, n_var)
    consistent = all(plant[v] == a.get(v, plant[v]) for v in a)
    return {
        "synthetic_up_props": float(props),
        "synthetic_up_assigned": float(len(a)),
        "synthetic_up_consistent": float(consistent),
    }
