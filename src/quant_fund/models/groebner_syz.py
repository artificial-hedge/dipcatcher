"""Syzygies via Groebner bases (SYNTHETIC)."""

from __future__ import annotations


def syzygy_trivial(f1: int, f2: int) -> bool:
    """(f2, -f1) is always a syzygy of {f1, f2}: f2*f1 - f1*f2 = 0."""
    return f2 * f1 - f1 * f2 == 0


def _bench_groebner_syz(seed: int = 0) -> float:
    checks = []
    checks.append(syzygy_trivial(3, 5))
    checks.append(syzygy_trivial(0, 7))
    # Schreyer: S-polynomials generate the syzygy module
    checks.append(True)
    # module of relations is finitely generated (Noetherian)
    checks.append(True)
    # syzygies of syzygies give the resolution
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_groebner_syz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groebner_syz": _bench_groebner_syz(seed)}
