"""CW complexes: cell counts and Euler characteristic chi = sum (-1)^i n_i (SYNTHETIC)."""

from __future__ import annotations


def euler_from_cells(cells: list[int]) -> int:
    """Euler characteristic from number of cells per dimension."""
    return int(sum((-1) ** i * c for i, c in enumerate(cells)))


def _bench_cw_complex(seed: int = 0) -> float:
    checks = []
    # S^1 minimal: 1 vertex + 1 edge -> chi = 0
    checks.append(euler_from_cells([1, 1]) == 0)
    # S^2 minimal: 1 vertex + 1 2-cell -> chi = 2
    checks.append(euler_from_cells([1, 0, 1]) == 2)
    # S^2 as octahedron boundary: 6 v + 12 e + 8 f -> chi = 2
    checks.append(euler_from_cells([6, 12, 8]) == 2)
    # torus: 1 v + 2 e + 1 f -> chi = 0
    checks.append(euler_from_cells([1, 2, 1]) == 0)
    # torus as 4x4 grid: 16v+32e+16f -> chi = 0
    checks.append(euler_from_cells([16, 32, 16]) == 0)
    # genus g surface: chi = 2 - 2g; g=2 -> -2 (model: 1v+4e+1f)
    checks.append(euler_from_cells([1, 4, 1]) == -2)
    # disc D^2: chi = 1 (triangle: 3v+3e+1f)
    checks.append(euler_from_cells([3, 3, 1]) == 1)
    # RP^2 minimal: chi = 1 (1v+1e+1f with twisted glueing)
    checks.append(euler_from_cells([1, 1, 1]) == 1)
    return float(sum(checks) / len(checks))


def bench_cw_complex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cw_complex": _bench_cw_complex(seed)}
