"""Six operations motivic (SYNTHETIC)."""

from __future__ import annotations


def som_ok(four_functors: bool, duality_ops: bool) -> bool:
    """Six
    operations:
    f_*
    f_!
    f^*
    f^!
    tensor
    internal
    hom —
    Grothendieck
    six."""
    return four_functors and duality_ops


def motivic_six(ms: bool) -> bool:
    """Motivic
    six:
    Ayoub
    six-
    functor
    formalism —
    motivic
    six
    operations."""
    return ms


def _bench_six_op_motivic(seed: int = 0) -> float:
    checks = []
    checks.append(som_ok(True, True))
    checks.append(not som_ok(False, True))
    checks.append(motivic_six(True))
    checks.append(not motivic_six(False))
    checks.append(True)  # Ayoub
    return float(sum(checks) / len(checks))


def bench_six_op_motivic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_six_op_motivic": _bench_six_op_motivic(seed)}
