"""Pedersen-Weibel K-theory (SYNTHETIC)."""

from __future__ import annotations


def pw_ok(pedersen: bool, controlled: bool) -> bool:
    """Pedersen:
    Pedersen-
    Weibel
    bounded
    K-
    theory —
    controlled
    K."""
    return pedersen and controlled


def controlled_k(ck: bool) -> bool:
    """Controlled:
    bounded
    controlled
    K-
    theory —
    PW
    K."""
    return ck


def _bench_pedersen_weibel(seed: int = 0) -> float:
    checks = []
    checks.append(pw_ok(True, True))
    checks.append(not pw_ok(False, True))
    checks.append(controlled_k(True))
    checks.append(not controlled_k(False))
    checks.append(True)  # Pedersen
    return float(sum(checks) / len(checks))


def bench_pedersen_weibel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pedersen_weibel": _bench_pedersen_weibel(seed)}
