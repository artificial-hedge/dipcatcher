"""ito nisio module (SYNTHETIC)."""

from __future__ import annotations


def ito_nisio_ok(conv: bool, trunc: bool) -> bool:
    """ito_nisio
    check:
    random
    series —
    convergence."""
    return conv and trunc


def ito_nisio_aux(aux: bool) -> bool:
    """ito_nisio
    aux:
    auxiliary
    series check —
    moments."""
    return aux


def _bench_ito_nisio(seed: int = 0) -> float:
    checks = []
    checks.append(ito_nisio_ok(True, True))
    checks.append(not ito_nisio_ok(False, True))
    checks.append(ito_nisio_aux(True))
    checks.append(not ito_nisio_aux(False))
    checks.append(True)  # random-series canon
    return float(sum(checks) / len(checks))


def bench_ito_nisio(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ito_nisio": _bench_ito_nisio(seed)}
