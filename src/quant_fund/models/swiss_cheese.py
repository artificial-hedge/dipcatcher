"""Swiss-cheese operad: disks in the half-plane (SYNTHETIC)."""

from __future__ import annotations


def disjoint_in_halfplane(centers: list[tuple[float, float]], r: float, plane_h: float) -> bool:
    """Config of little disks in upper half-plane: all inside, pairwise
    disjoint, boundary centers on the x-axis."""
    if any(y < 0 for _, y in centers):
        return False
    for i, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[i + 1 :]:
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < (2 * r) ** 2:
                return False
    return all(y <= plane_h for _, y in centers)


def _bench_swiss_cheese(seed: int = 0) -> float:
    checks = []
    # two open disks separated
    checks.append(disjoint_in_halfplane([(0.0, 0.5), (1.0, 0.5)], 0.3, 1.0))
    # overlapping fails
    checks.append(not disjoint_in_halfplane([(0.0, 0.5), (0.4, 0.5)], 0.3, 1.0))
    # below axis fails (outside half-plane)
    checks.append(not disjoint_in_halfplane([(0.0, -0.1)], 0.3, 1.0))
    # boundary color: centers on x-axis allowed
    checks.append(disjoint_in_halfplane([(0.0, 0.0), (1.0, 0.0)], 0.3, 1.0))
    # single disk trivially valid
    checks.append(disjoint_in_halfplane([(0.5, 0.5)], 0.5, 1.0))
    return float(sum(checks) / len(checks))


def bench_swiss_cheese(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swiss_cheese": _bench_swiss_cheese(seed)}
