"""R(3,3)=6: every 2-coloring of K6 has a monochromatic triangle (SYNTHETIC)."""

from __future__ import annotations

import itertools


def mono_triangle(n: int, color: dict[tuple[int, int], int]) -> bool:
    """Exists a monochromatic K3 under edge coloring."""
    for tri in itertools.combinations(range(n), 3):
        cols = {color[(min(a, b), max(a, b))] for a, b in itertools.combinations(tri, 2)}
        if len(cols) == 1:
            return True
    return False


def every_coloring_bad(n: int) -> bool:
    edges = [(min(a, b), max(a, b)) for a, b in itertools.combinations(range(n), 2)]
    for assign in itertools.product((0, 1), repeat=len(edges)):
        color = dict(zip(edges, assign, strict=True))
        if not mono_triangle(n, color):
            return False
    return True


def k5_good_coloring() -> dict[tuple[int, int], int]:
    """The 5-cycle coloring of K5 with no mono triangle."""
    color: dict[tuple[int, int], int] = {}
    for a, b in itertools.combinations(range(5), 2):
        # color 0 if edge on the pentagon cycle, else 1
        if (a - b) % 5 in (1, 4):
            color[(a, b)] = 0
        else:
            color[(a, b)] = 1
    return color


def _bench_ramsey_r33(seed: int = 0) -> float:
    checks = []
    checks.append(every_coloring_bad(6))  # R(3,3) <= 6
    checks.append(not mono_triangle(5, k5_good_coloring()))  # R(3,3) > 5
    checks.append(every_coloring_bad(6))
    # a random-looking coloring of K6 must contain mono triangle
    color6 = {}
    for a, b in itertools.combinations(range(6), 2):
        color6[(a, b)] = (a + b) % 2
    checks.append(mono_triangle(6, color6))
    # verify coloring counterexample on K4 exists too
    color4 = {(0, 1): 0, (0, 2): 0, (1, 2): 1, (0, 3): 1, (1, 3): 1, (2, 3): 0}
    checks.append(not mono_triangle(4, color4))
    checks.append(every_coloring_bad(6))
    return float(sum(checks) / len(checks))


def bench_ramsey_r33(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ramsey_r33": _bench_ramsey_r33(seed)}
