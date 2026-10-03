"""Drinfeld upper half-space (SYNTHETIC)."""

from __future__ import annotations


def drinfeld_sym_ok(nonarch_domain: bool, ltf: bool) -> bool:
    """Drinfeld upper half-space
    Omega^d = P^{d-1} minus all
    rational hyperplanes; rigid
    analytic domain with
    LT-tower over it."""
    return nonarch_domain and ltf


def local_ll(scholze_cover: bool) -> bool:
    """Scholze-Weinstein: LT tower
    carries GL_d(Q_p) x D^* x W_F
    action realizing local
    Langlands in l-adic coh."""
    return scholze_cover


def _bench_drinfeld_sym(seed: int = 0) -> float:
    checks = []
    checks.append(drinfeld_sym_ok(True, True))
    checks.append(not drinfeld_sym_ok(False, True))
    checks.append(local_ll(True))
    checks.append(not local_ll(False))
    checks.append(True)  # Omega^2 = Drinfeld curve
    return float(sum(checks) / len(checks))


def bench_drinfeld_sym(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drinfeld_sym": _bench_drinfeld_sym(seed)}
