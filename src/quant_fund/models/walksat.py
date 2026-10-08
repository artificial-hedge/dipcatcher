"""WalkSAT stochastic local search on random 3-SAT (SYNTHETIC).

Pick a random unsatisfied clause, flip a variable in it — free choice
with prob p (random walk), min-conflict otherwise. Bench: solve rate
at the phase transition and agreement with brute force.
"""

import numpy as np


def _sat_count(clauses: list[list[int]], a: dict[int, bool]) -> int:
    return sum(1 for c in clauses if any(a[abs(lit)] == (lit > 0) for lit in c))


def _walksat(
    clauses: list[list[int]], n_var: int, rng: np.random.Generator, max_flips: int = 600
) -> bool:
    a = {v: bool(rng.random() < 0.5) for v in range(1, n_var + 1)}
    for _ in range(max_flips):
        bad = [c for c in clauses if not any(a[abs(lit)] == (lit > 0) for lit in c)]
        if not bad:
            return True
        c = bad[rng.integers(len(bad))]
        if rng.random() < 0.5:
            v = abs(c[rng.integers(3)])
        else:
            v = min(
                (abs(lit) for lit in c),
                key=lambda vv: -(_sat_count(clauses, {**a, vv: not a[vv]})),
            )
        a[v] = not a[v]
    return _sat_count(clauses, a) == len(clauses)


def _rand_3sat(rng: np.random.Generator, n_var: int, m: int) -> list[list[int]]:
    return [
        [
            int(v) * int(s)
            for v, s in zip(
                rng.choice(n_var, 3, replace=False) + 1, rng.choice([-1, 1], 3), strict=True
            )
        ]
        for _ in range(m)
    ]


def bench_walksat(seed: int = 5903) -> dict[str, float]:
    import itertools

    rng = np.random.default_rng(seed)
    n_var, m = 10, 30
    trials = 40
    sat_ct, agree = 0, 0
    for _ in range(trials):
        cl = _rand_3sat(rng, n_var, m)
        got = _walksat(cl, n_var, rng)
        truth = any(
            all(any(vals[abs(lit) - 1] == (lit > 0) for lit in c) for c in cl)
            for vals in itertools.product([False, True], repeat=n_var)
        )
        sat_ct += int(got)
        agree += int(got == truth)
    return {
        "synthetic_ws_solve_rate": sat_ct / trials,
        "synthetic_ws_agree": agree / trials,
    }
