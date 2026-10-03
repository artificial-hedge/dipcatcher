"""Sutherland–Hodgman polygon clipping vs convex window — SYNTHETIC.

Output polygon must lie inside the clip window and have area <= the
subject's; for full-containment cases the area is conserved exactly.
"""

from __future__ import annotations

import math
import random

Pt = tuple[float, float]


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _area(poly: list[Pt]) -> float:
    if len(poly) < 3:
        return 0.0
    return (
        abs(
            sum(
                poly[i][0] * poly[(i + 1) % len(poly)][1]
                - poly[(i + 1) % len(poly)][0] * poly[i][1]
                for i in range(len(poly))
            )
        )
        / 2
    )


def clip(subject: list[Pt], window: list[Pt]) -> list[Pt]:
    """Clip `subject` by convex CCW `window`."""
    out = list(subject)
    m = len(window)
    for e in range(m):
        a, b = window[e], window[(e + 1) % m]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for p in inp:
            if _cross(a, b, p) >= -1e-12:  # inside (left of edge)
                if _cross(a, b, s) < -1e-12:
                    out.append(_isect(s, p, a, b))
                out.append(p)
            elif _cross(a, b, s) >= -1e-12:
                out.append(_isect(s, p, a, b))
            s = p
    return out


def _isect(p1: Pt, p2: Pt, p3: Pt, p4: Pt) -> Pt:
    d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
    if abs(d) < 1e-15:
        return p1
    t = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def _inside_window(p: Pt, window: list[Pt]) -> bool:
    m = len(window)
    return all(_cross(window[e], window[(e + 1) % m], p) >= -1e-9 for e in range(m))


def bench_sutherland_hodgman(seed: int = 20261231 + 321) -> dict[str, float]:
    rng = random.Random(seed)
    inside_ok = area_ok = conserve_ok = 0
    trials = 40
    win = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    for _ in range(trials):
        n = rng.randrange(3, 12)
        # random star-shaped polygon around origin
        subj = []
        for k in range(n):
            ang = 6.283185307179586 * k / n
            r = rng.uniform(0.5, 3.0)
            subj.append((r * math.cos(ang), r * math.sin(ang)))
        out = clip(subj, win)
        inside_ok += int(all(_inside_window(p, win) for p in out))
        area_ok += int(_area(out) <= _area(subj) + 1e-9)
        if _area(subj) <= _area(win):
            # if subject wholly inside → unchanged area
            if all(_inside_window(p, win) for p in subj):
                conserve_ok += int(abs(_area(out) - _area(subj)) < 1e-6)
            else:
                conserve_ok += 1  # n/a
        else:
            conserve_ok += 1
    return {
        "synthetic_inside_window": float(inside_ok / trials),
        "synthetic_area_nonincrease": float(area_ok / trials),
        "synthetic_containment_exact": float(conserve_ok / trials),
    }
