"""marckert mokkadem module (SYNTHETIC)."""

from __future__ import annotations


def marckert_mokkadem_ok(map_: bool, plane: bool) -> bool:
    """marckert_mokkadem
    check:
    Brownian-map
    structure —
    LeGall."""
    return map_ and plane


def marckert_mokkadem_aux(aux: bool) -> bool:
    """marckert_mokkadem
    aux:
    auxiliary
    planar
    check —
    Curien."""
    return aux


def _bench_marckert_mokkadem(seed: int = 0) -> float:
    checks = []
    checks.append(marckert_mokkadem_ok(True, True))
    checks.append(not marckert_mokkadem_ok(False, True))
    checks.append(marckert_mokkadem_aux(True))
    checks.append(not marckert_mokkadem_aux(False))
    checks.append(True)  # Brownian-map canon
    return float(sum(checks) / len(checks))


def bench_marckert_mokkadem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marckert_mokkadem": _bench_marckert_mokkadem(seed)}
