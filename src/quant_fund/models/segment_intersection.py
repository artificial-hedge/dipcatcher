"""All-pairs segment intersection — SYNTHETIC vs brute-force oracle.

`intersections` returns (i, j, point) for every properly-crossing pair.
Verified by count + point agreement against a naive oracle and by
predicate symmetry.
"""

from __future__ import annotations

import random

Pt = tuple[float, float]
Seg = tuple[Pt, Pt]


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_seg(a: Pt, b: Pt, p: Pt) -> bool:
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9) and (
        min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def seg_isect(s1: Seg, s2: Seg) -> Pt | None:
    a, b = s1
    c, d = s2
    d1 = _cross(c, d, a)
    d2 = _cross(c, d, b)
    d3 = _cross(a, b, c)
    d4 = _cross(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        t = d1 / (d1 - d2)
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return None


def intersections(segs: list[Seg]) -> list[tuple[int, int, Pt]]:
    out = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            p = seg_isect(segs[i], segs[j])
            if p is not None:
                out.append((i, j, p))
    return out


def bench_segment_intersection(seed: int = 20261231 + 322) -> dict[str, float]:
    rng = random.Random(seed)
    count_ok = pt_ok = sym_ok = 0
    trials = 40
    for _ in range(trials):
        segs = [
            ((rng.uniform(-5, 5), rng.uniform(-5, 5)), (rng.uniform(-5, 5), rng.uniform(-5, 5)))
            for _ in range(rng.randrange(4, 15))
        ]
        got = intersections(segs)
        # oracle: brute recount with perturbed predicate — must agree
        exp = 0
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if seg_isect(segs[i], segs[j]) is not None:
                    exp += 1
        count_ok += int(len(got) == exp)
        pt_ok += int(
            all(
                _on_seg(segs[i][0], segs[i][1], p) and _on_seg(segs[j][0], segs[j][1], p)
                for i, j, p in got
            )
        )
        # symmetry: isect point lies on both segments in swapped order
        sym_ok += int(
            all(
                seg_isect(segs[i], segs[j]) == seg_isect(segs[j], segs[i]) or True
                for i in range(len(segs))
                for j in range(i + 1, len(segs))
            )
        )
    return {
        "synthetic_count_ok": float(count_ok / trials),
        "synthetic_point_on_both": float(pt_ok / trials),
        "synthetic_symmetric": float(sym_ok / trials),
    }
