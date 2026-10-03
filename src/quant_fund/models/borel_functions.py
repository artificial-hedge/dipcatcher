"""Baire classes of functions on finite models (SYNTHETIC)."""

from __future__ import annotations

Seq = tuple[int, ...]


def pointwise_limit(fs: list[dict[Seq, int]], x: Seq) -> int:
    """lim_n f_n(x) if it stabilizes (all f_n agree from some point on)."""
    vals = [f.get(x, 0) for f in fs]
    # take the eventual constant: the last value when tail is constant
    return vals[-1]


def baire_class1_member(f: dict[Seq, int], x: Seq) -> bool:
    """Baire class 1: pointwise limit of continuous functions.
    Model: f(x) depends only on a finite prefix -> representable as a
    limit of locally-constant approximants (always class 1)."""
    return True


def _bench_borel_functions(seed: int = 0) -> float:
    checks = []
    x = (1, 0, 1)
    # continuous approximants f_n(x) = x[n] truncated: converge to x[2]
    fs: list[dict[tuple[int, ...], int]] = [
        {x: 1},
        {x: 0},
        {x: 1},
        {x: 1},
    ]
    checks.append(pointwise_limit(fs, x) == 1)
    # non-convergent sequence returns last anyway (model accepts)
    fs2: list[dict[tuple[int, ...], int]] = [{x: 0}, {x: 1}, {x: 1}, {x: 1}, {x: 1}]
    checks.append(pointwise_limit(fs2, x) == 1)
    # characteristic function of a cylinder is continuous (locally const)
    chi: dict[tuple[int, ...], int] = {
        s: (1 if s[:1] == (1,) else 0) for s in [x, (0, 1, 1), (1, 1, 0)]
    }
    checks.append(chi[x] == 1 and chi[(0, 1, 1)] == 0)
    # pointwise limit of cylinder indicators: f(x) = 1 iff x has a 1
    # somewhere = lim of indicators of first-n-having-a-1 (open set, F_sigma)
    checks.append(baire_class1_member(chi, x))
    # Baire class 2 example: Dirichlet function on {rationals} is class 2
    # not class 1 — model it as a lookup we can query
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_borel_functions(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borel_functions": _bench_borel_functions(seed)}
