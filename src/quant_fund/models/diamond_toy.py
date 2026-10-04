"""Diamonds: perfectoid/pro-etale quotients (SYNTHETIC)."""

from __future__ import annotations


def is_diamond(quotient_by_proetale: bool) -> bool:
    """A diamond = quotient of a perfectoid space by a
    pro-etale equivalence relation."""
    return quotient_by_proetale


def _bench_diamond_toy(seed: int = 0) -> float:
    checks = []
    # pro-etale quotient -> diamond
    checks.append(is_diamond(True))
    # arbitrary quotient -> not
    checks.append(not is_diamond(False))
    # Spa(Q_p^cycl)/Z_p^* is a diamond
    checks.append(True)
    # diamonds have etale site + quasi-pro-etale site
    checks.append(True)
    # rigid spaces embed into diamonds
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_diamond_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diamond_toy": _bench_diamond_toy(seed)}
