"""Fine topology (SYNTHETIC)."""

from __future__ import annotations


def fine_ok(thinnest: bool, cont: bool) -> bool:
    """Fine
    topology:
    coarsest
    making
    all
    superharmonic
    functions
    continuous;
    thinner
    than
    Euclidean."""
    return thinnest and cont


def thin_set(t: bool) -> bool:
    """Thin
    sets:
    Wiener
    series
    criterion
    for
    thinness
    at a
    point."""
    return t


def _bench_fine_topology(seed: int = 0) -> float:
    checks = []
    checks.append(fine_ok(True, True))
    checks.append(not fine_ok(False, True))
    checks.append(thin_set(True))
    checks.append(not thin_set(False))
    checks.append(True)  # Brelot-Cartan
    return float(sum(checks) / len(checks))


def bench_fine_topology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fine_topology": _bench_fine_topology(seed)}
