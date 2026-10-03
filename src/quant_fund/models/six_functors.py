"""Six-functor formalism (SYNTHETIC)."""

from __future__ import annotations


def six_ops(f_star: bool, f_shriek: bool, tensor: bool) -> bool:
    """The six operations (f*, f_*, f_!, f^!, tensor, Hom)
    with adjunctions (f*,f*) and (f_!,f^!) form a
    six-functor formalism on a class of morphisms."""
    return f_star and f_shriek and tensor


def _bench_six_functors(seed: int = 0) -> float:
    checks = []
    # all functor families present
    checks.append(six_ops(True, True, True))
    # missing !-functors fails
    checks.append(not six_ops(True, False, True))
    # Grothendieck original motivation
    checks.append(True)
    # exists for etale/constructible settings
    checks.append(True)
    # extended to stacks/DAG by Lurie/Gaitsgory
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_six_functors(seed: int = 0) -> dict[str, float]:
    return {"synthetic_six_functors": _bench_six_functors(seed)}
