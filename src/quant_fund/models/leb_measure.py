"""Lebesgue outer measure via interval covers (wave 288).

Cantor-set outer measure -> 0 under the standard middle-third cover;
union of intervals recovers the sum of lengths.
"""

import numpy as np

_SEED = 20261231 + 812


def _cantor_level(n: int) -> np.ndarray:
    iv = np.array([[0.0, 1.0]])
    for _ in range(n):
        out = []
        for a, b in iv:
            t = (b - a) / 3
            out += [[a, a + t], [b - t, b]]
        iv = np.array(out)
    return iv


def outer_measure(points_or_cover: np.ndarray) -> float:
    return float(np.sum(points_or_cover[:, 1] - points_or_cover[:, 0]))


def bench_leb_measure(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    # cantor outer measure at level 10 is (2/3)^10 < 0.02 and shrinks
    for n in (6, 8, 10):
        ok += int(outer_measure(_cantor_level(n)) <= (2 / 3) ** n + 1e-12)
    # union [0,0.5] u [0.5,1] has measure 1; overlapping union covered once
    cover = np.array([[0.0, 0.5], [0.5, 1.0]])
    ok += int(abs(outer_measure(cover) - 1.0) < 1e-12)
    return {"synthetic_leb_measure": float(ok == 4)}
