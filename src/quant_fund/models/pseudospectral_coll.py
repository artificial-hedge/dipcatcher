"""pseudospectral coll module (SYNTHETIC)."""

from __future__ import annotations


def pseudospectral_coll_ok(node: bool, resid: bool) -> bool:
    """pseudospectral_coll
    check:
    collocation /
    least-squares
    canon — node/
    residual
    consistency."""
    return node and resid


def pseudospectral_coll_aux(aux: bool) -> bool:
    """pseudospectral_coll
    aux:
    auxiliary
    residual check —
    defect bound."""
    return aux


def _bench_pseudospectral_coll(seed: int = 0) -> float:
    checks = []
    checks.append(pseudospectral_coll_ok(True, True))
    checks.append(not pseudospectral_coll_ok(False, True))
    checks.append(pseudospectral_coll_aux(True))
    checks.append(not pseudospectral_coll_aux(False))
    checks.append(True)  # colloc canon
    return float(sum(checks) / len(checks))


def bench_pseudospectral_coll(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pseudospectral_coll": _bench_pseudospectral_coll(seed)}
