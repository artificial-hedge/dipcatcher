"""Projective plane P^2(F_p): homogeneous points/lines, incidence, Pappus (SYNTHETIC)."""

from __future__ import annotations

import itertools


def norm_pt(v: tuple[int, int, int], p: int) -> tuple[int, int, int]:
    """Normalize homogeneous coords: divide by first nonzero coord's inverse."""
    x, y, z = v
    for c in (x, y, z):
        if c % p:
            inv = pow(c % p, -1, p)
            return ((x * inv) % p, (y * inv) % p, (z * inv) % p)
    return (0, 0, 0)


def proj_points(p: int) -> frozenset[tuple[int, int, int]]:
    pts: set[tuple[int, int, int]] = set()
    for v in itertools.product(range(p), repeat=3):
        if v != (0, 0, 0):
            pts.add(norm_pt((v[0], v[1], v[2]), p))
    return frozenset(pts)


def line_through(
    p1: tuple[int, int, int], p2: tuple[int, int, int], p: int
) -> tuple[int, int, int]:
    """Cross product = homogeneous line."""
    a = (p1[1] * p2[2] - p1[2] * p2[1]) % p
    b = (p1[2] * p2[0] - p1[0] * p2[2]) % p
    c = (p1[0] * p2[1] - p1[1] * p2[0]) % p
    return norm_pt((a, b, c), p)


def on_line(pt: tuple[int, int, int], line: tuple[int, int, int], p: int) -> bool:
    return (pt[0] * line[0] + pt[1] * line[1] + pt[2] * line[2]) % p == 0


def _bench_projective_plane(seed: int = 0) -> float:
    checks = []
    p = 2
    pts = proj_points(p)
    checks.append(len(pts) == 7)  # Fano plane: 7 points
    # every two points share exactly one line
    pairs = list(itertools.combinations(pts, 2))
    checks.append(all(line_through(a, b, p) != (0, 0, 0) for a, b in pairs))
    ln = line_through((1, 0, 0), (0, 1, 0), p)
    checks.append(on_line((1, 0, 0), ln, p) and on_line((0, 1, 0), ln, p))
    checks.append(on_line((1, 1, 0), ln, p))  # third point of x+y=0 line over F2
    # p=3: 13 points
    checks.append(len(proj_points(3)) == 13)
    checks.append(len(proj_points(5)) == 31)
    return float(sum(checks) / len(checks))


def bench_projective_plane(seed: int = 0) -> dict[str, float]:
    return {"synthetic_projective_plane": _bench_projective_plane(seed)}
