"""Cofiber sequences and cell attachment (SYNTHETIC)."""

from __future__ import annotations


def attach_cell(betti: list[int], cell_dim: int, degree: int = 0) -> list[int]:
    """Attach a cell of dimension d along a degree-m map S^{d-1} -> X.

    Homology effect: the cellular boundary map multiplies by m on the
    (d-1)-cycle it attaches along. If m = 0 the d-cell adds a free
    generator (b_d += 1) and b_{d-1} unchanged; if m != 0 it kills the
    (d-1)-class rationally (b_{d-1} -= 1) and contributes nothing in d.
    """
    b = list(betti) + [0] * (cell_dim + 1 - len(betti))
    if degree == 0:
        b[cell_dim] += 1
    else:
        b[cell_dim - 1] = max(0, b[cell_dim - 1] - 1)
    return b


def chi(betti: list[int]) -> int:
    return int(sum((-1) ** i * b for i, b in enumerate(betti)))


def _bench_cofibration(seed: int = 0) -> float:
    checks = []
    # S2 = pt + 2-cell attached trivially: betti [1,0,1]
    checks.append(attach_cell([1], 2, 0) == [1, 0, 1])
    # D2 = attach 2-cell along identity to S1: kills H1: [1,0,0]
    checks.append(attach_cell([1, 1], 2, 1) == [1, 0, 0])
    # RP2 = attach 2-cell along degree-2 map to S1: rationally kills H1
    checks.append(attach_cell([1, 1], 2, 2) == [1, 0, 0])
    # torus: square -> attach 2-cell degree 0 to wedge of 2 circles:
    checks.append(attach_cell([1, 2], 2, 0) == [1, 2, 1])
    # chi after attaching odd cell decreases by 1
    checks.append(chi(attach_cell([1, 2], 3, 0)) == chi([1, 2]) - 1)
    # chi after even cell increases by 1
    checks.append(chi(attach_cell([1, 1], 2, 0)) == chi([1, 1]) + 1)
    return float(sum(checks) / len(checks))


def bench_cofibration(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cofibration": _bench_cofibration(seed)}
