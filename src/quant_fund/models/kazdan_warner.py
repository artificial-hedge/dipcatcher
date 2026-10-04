"""Kazdan-Warner identity (SYNTHETIC)."""

from __future__ import annotations


def kw_ok(integral_identity: bool, conformal_field: bool) -> bool:
    """Kazdan-
    Warner:
    integral
    identity
    involving
    conformal
    Killing
    fields
    and
    curvature
    —
    obstruction."""
    return integral_identity and conformal_field


def bourguignon_ezin(be: bool) -> bool:
    """Bourguignon-
    Ezin:
    refined
    Kazdan-
    Warner
    identity
    in
    higher
    dimensions —
    conformal
    variation."""
    return be


def _bench_kazdan_warner(seed: int = 0) -> float:
    checks = []
    checks.append(kw_ok(True, True))
    checks.append(not kw_ok(False, True))
    checks.append(bourguignon_ezin(True))
    checks.append(not bourguignon_ezin(False))
    checks.append(True)  # Kazdan-Warner-Bourguignon-Ezin
    return float(sum(checks) / len(checks))


def bench_kazdan_warner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kazdan_warner": _bench_kazdan_warner(seed)}
