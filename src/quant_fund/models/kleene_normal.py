"""Kleene T-predicate and normal form (SYNTHETIC)."""

from __future__ import annotations


def t_pred(e: int, x: int, steps: int) -> int | None:
    """Toy T(e, x, s): returns output iff program e halts on x within
    s steps; programs are tiny arithmetic maps."""
    prog = _program(e)
    r = prog(x, steps)
    return None if r is None else int(r)


def _program(e: int):
    def run(x: int, s: int) -> int | None:
        if e == 0:  # successor program
            return x + 1 if s >= 1 else None
        if e == 1:  # doubling program needing 2 steps
            return 2 * x if s >= 2 else None
        if e == 2:  # never halts
            return None
        return x if s >= 1 else None

    return run


def u_extract(res: int | None) -> int:
    """U: extract output from a computation result (identity here)."""
    return 0 if res is None else res


def _bench_kleene_normal(seed: int = 0) -> float:
    checks = []
    # successor halts with 1 step
    checks.append(t_pred(0, 4, 1) == 5)
    # doubling needs 2 steps
    checks.append(t_pred(1, 3, 1) is None)
    checks.append(t_pred(1, 3, 2) == 6)
    # non-halting program yields None
    checks.append(t_pred(2, 0, 100) is None)
    # U extraction of a halt gives output, of divergence gives 0
    checks.append(u_extract(t_pred(0, 7, 1)) == 8)
    checks.append(u_extract(t_pred(2, 7, 9)) == 0)
    return float(sum(checks) / len(checks))


def bench_kleene_normal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kleene_normal": _bench_kleene_normal(seed)}
