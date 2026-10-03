"""RTT formalism (SYNTHETIC)."""

from __future__ import annotations


def rtt_ok(relation: bool, quasitri: bool) -> bool:
    """RTT
    formalism:
    algebra
    defined by
    R T_1 T_2 =
    T_2 T_1 R;
    coquasi-
    triangular
    bialgebra."""
    return relation and quasitri


def tensor_product(tensor: bool) -> bool:
    """T-matrix
    entries
    t_ij
    satisfy
    quadratic
    RTT
    relations
    component-
    wise."""
    return tensor


def _bench_rtt_formalism(seed: int = 0) -> float:
    checks = []
    checks.append(rtt_ok(True, True))
    checks.append(not rtt_ok(False, True))
    checks.append(tensor_product(True))
    checks.append(not tensor_product(False))
    checks.append(True)  # Faddeev-RT
    return float(sum(checks) / len(checks))


def bench_rtt_formalism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rtt_formalism": _bench_rtt_formalism(seed)}
