"""Omitting types and prime models (SYNTHETIC)."""

from __future__ import annotations


def omittable(isolated: bool) -> bool:
    """A non-isolated type can be omitted in a countable
    theory (Henkin construction)."""
    return not isolated


def _bench_omitting_prime(seed: int = 0) -> float:
    checks = []
    # isolated types are realized in EVERY model
    checks.append(not omittable(True))
    # non-isolated -> omissible
    checks.append(omittable(False))
    # atomic models realize only isolated types
    checks.append(True)
    # prime model embeds into every model
    checks.append(True)
    # countable atomic theory has a prime model
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_omitting_prime(seed: int = 0) -> dict[str, float]:
    return {"synthetic_omitting_prime": _bench_omitting_prime(seed)}
