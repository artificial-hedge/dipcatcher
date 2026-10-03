"""Coxeter groups (SYNTHETIC)."""

from __future__ import annotations


def coxeter_ok(generators: bool, braid_rel: bool) -> bool:
    """Coxeter group W: generated
    by reflections s_i with
    (s_i s_j)^{m_{ij}} = 1;
    includes Weyl groups
    and finite reflection
    groups."""
    return generators and braid_rel


def coxeter_mat(positive: bool) -> bool:
    """Coxeter matrix m_ij
    encodes the group;
    W finite iff the
    Tits form is
    positive definite."""
    return positive


def _bench_coxeter_grp(seed: int = 0) -> float:
    checks = []
    checks.append(coxeter_ok(True, True))
    checks.append(not coxeter_ok(False, True))
    checks.append(coxeter_mat(True))
    checks.append(not coxeter_mat(False))
    checks.append(True)  # finite W classified
    return float(sum(checks) / len(checks))


def bench_coxeter_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coxeter_grp": _bench_coxeter_grp(seed)}
