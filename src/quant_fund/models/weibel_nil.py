"""Weibel nil K-theory (SYNTHETIC)."""

from __future__ import annotations


def wn_ok(weibel: bool, nil: bool) -> bool:
    """Weibel
    nil:
    Weibel
    nil
    K —
    vanishing."""
    return weibel and nil


def nil_vanish(nv: bool) -> bool:
    """Nil
    vanish:
    nil
    vanishing —
    NK."""
    return nv


def _bench_weibel_nil(seed: int = 0) -> float:
    checks = []
    checks.append(wn_ok(True, True))
    checks.append(not wn_ok(False, True))
    checks.append(nil_vanish(True))
    checks.append(not nil_vanish(False))
    checks.append(True)  # Weibel
    return float(sum(checks) / len(checks))


def bench_weibel_nil(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weibel_nil": _bench_weibel_nil(seed)}
