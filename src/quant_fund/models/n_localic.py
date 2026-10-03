"""n-Localic infinity-toposes (SYNTHETIC)."""

from __future__ import annotations


def localic_level(n: int, subobject_lat: bool) -> bool:
    """An n-localic topos is generated under
    colimits by its subobjects of (n-1)-truncated
    objects; 0-localic = spatial frames."""
    return n >= 0 and subobject_lat


def etale_over(n_loc: int) -> int:
    """An n-localic topos is etale over an
    n-localic base; covers by local homeos."""
    return max(0, n_loc)


def _bench_n_localic(seed: int = 0) -> float:
    checks = []
    checks.append(localic_level(0, True))
    checks.append(localic_level(2, True))
    checks.append(not localic_level(1, False))
    checks.append(etale_over(3) == 3)
    checks.append(etale_over(0) == 0)
    return float(sum(checks) / len(checks))


def bench_n_localic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_n_localic": _bench_n_localic(seed)}
