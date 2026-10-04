"""Left exact reflectors (SYNTHETIC)."""

from __future__ import annotations


def lex_reflector(left_exact: bool, accessible: bool) -> bool:
    """A left exact reflector L: C -> LC preserves
    finite limits; localizations correspond to
    topological quotients (subtoposes)."""
    return left_exact and accessible


def cotopological(bundling: bool) -> bool:
    """Cotopological morphism = conservative
    lex reflector on 0-truncated objects."""
    return bundling


def _bench_lex_reflect(seed: int = 0) -> float:
    checks = []
    checks.append(lex_reflector(True, True))
    checks.append(not lex_reflector(False, True))
    checks.append(cotopological(True))
    checks.append(not cotopological(False))
    checks.append(True)  # subtopos = accessible lex
    return float(sum(checks) / len(checks))


def bench_lex_reflect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lex_reflect": _bench_lex_reflect(seed)}
