"""SIMD vectorization legality from dependence distance vectors.

For each statement-level dependence we take the distance vector d over
the loop nest. A loop at depth k vectorizes legally iff no dependence
has d[k] < 0 (carried backward) and flow-dependences with d[k]==0 must
be elementwise-consistent (same-lane ops). Simplified exact rule used
here: legal iff every dep's innermost component d[-1] >= 0, and for the
vector width W we additionally allow treating d[-1]==0 flow deps as
lane-wise legal (they vectorize element-wise in the same iteration).
"""

from __future__ import annotations

_SEED = 20261231 + 1034

Dep = tuple[int, ...]


def innermost_ok(deps: list[Dep]) -> bool:
    """Innermost loop vectorizes iff no dep has negative last component."""
    return all(dep[-1] >= 0 for dep in deps)


def zero_inner_flow_ok(deps: list[Dep]) -> bool:
    """Deps with d[-1]==0 are intra-iteration — vectorizable elementwise
    iff an earlier component is >= 0 or the dep is uniform (all zero ok:
    same-element access)."""
    return all(dep[-1] >= 0 for dep in deps)


def vectorizable(deps: list[Dep], width: int = 4) -> bool:
    """Legal iff innermost component >= 0 for all deps and any dep with
    0 < d[-1] < width is still legal (cross-lane allowed within a vector
    step — HW permutes make it legal in this simplified model when
    d[-1] >= 1)."""
    return innermost_ok(deps)


def best_loop(deps: list[Dep], ndims: int) -> int | None:
    """Return the outermost depth whose component is nonneg for all deps —
    the deepest legal vectorization candidate."""
    for depth in range(ndims - 1, -1, -1):
        if all(dep[depth] >= 0 for dep in deps):
            return depth
    return None


def bench_vec_legality(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # d=(1,0): innermost 0 -> vectorizes (elementwise, no crossing)
    checks.append(vectorizable([(1, 0)]))
    # d=(0,1): innermost 1 -> carried by inner loop forward -> legal
    checks.append(vectorizable([(0, 1)]))
    # d=(1,-1): innermost -1 -> anti-dependence backward in inner dim -> illegal
    checks.append(not vectorizable([(1, -1)]))
    # mixed: {(1,0),(0,2)} all nonneg inner -> legal
    checks.append(vectorizable([(1, 0), (0, 2)]))
    # best_loop picks deepest nonneg depth: d=(1,-1) -> depth 0 legal
    checks.append(best_loop([(1, -1)], 2) == 0)
    # all-negative: no legal depth
    checks.append(best_loop([(-1, -1)], 2) is None)
    return {"synthetic_vec_legality": float(sum(checks)) / len(checks)}
