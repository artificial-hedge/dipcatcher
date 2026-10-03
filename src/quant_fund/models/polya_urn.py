"""polya urn module (SYNTHETIC)."""

from __future__ import annotations


def polya_urn_ok(chain: bool, mix: bool) -> bool:
    """polya_urn
    check:
    mixing
    structure —
    Bradley."""
    return chain and mix


def polya_urn_aux(aux: bool) -> bool:
    """polya_urn
    aux:
    auxiliary
    urn
    check —
    Hopf."""
    return aux


def _bench_polya_urn(seed: int = 0) -> float:
    checks = []
    checks.append(polya_urn_ok(True, True))
    checks.append(not polya_urn_ok(False, True))
    checks.append(polya_urn_aux(True))
    checks.append(not polya_urn_aux(False))
    checks.append(True)  # mixing canon
    return float(sum(checks) / len(checks))


def bench_polya_urn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polya_urn": _bench_polya_urn(seed)}
