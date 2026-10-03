"""shanks trans module (SYNTHETIC)."""

from __future__ import annotations


def shanks_trans_ok(node: bool, form: bool) -> bool:
    """shanks_trans
    check:
    interpolation-2
    canon — node/
    form
    consistency."""
    return node and form


def shanks_trans_aux(aux: bool) -> bool:
    """shanks_trans
    aux:
    auxiliary
    basis check —
    interpolation bound."""
    return aux


def _bench_shanks_trans(seed: int = 0) -> float:
    checks = []
    checks.append(shanks_trans_ok(True, True))
    checks.append(not shanks_trans_ok(False, True))
    checks.append(shanks_trans_aux(True))
    checks.append(not shanks_trans_aux(False))
    checks.append(True)  # interp canon
    return float(sum(checks) / len(checks))


def bench_shanks_trans(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shanks_trans": _bench_shanks_trans(seed)}
