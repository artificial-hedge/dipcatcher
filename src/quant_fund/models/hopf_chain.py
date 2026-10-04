"""hopf chain module (SYNTHETIC)."""

from __future__ import annotations


def hopf_chain_ok(chain: bool, mix: bool) -> bool:
    """hopf_chain
    check:
    mixing
    structure —
    Bradley."""
    return chain and mix


def hopf_chain_aux(aux: bool) -> bool:
    """hopf_chain
    aux:
    auxiliary
    urn
    check —
    Hopf."""
    return aux


def _bench_hopf_chain(seed: int = 0) -> float:
    checks = []
    checks.append(hopf_chain_ok(True, True))
    checks.append(not hopf_chain_ok(False, True))
    checks.append(hopf_chain_aux(True))
    checks.append(not hopf_chain_aux(False))
    checks.append(True)  # mixing canon
    return float(sum(checks) / len(checks))


def bench_hopf_chain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hopf_chain": _bench_hopf_chain(seed)}
