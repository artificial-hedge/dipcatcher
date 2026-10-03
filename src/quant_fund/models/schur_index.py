"""Schur index over R (SYNTHETIC)."""

from __future__ import annotations


def schur_index(name: str) -> int:
    """Known small-group Schur indices over R."""
    return {
        "q8_2d": 2,  # quaternion 2-d rep: real char, no real form
        "s3_std": 1,
        "c3_faithful": 1,  # over R needs 2 dims anyway (complex type)
        "d8_std": 1,
    }[name]


def _bench_schur_index(seed: int = 0) -> float:
    checks = []
    # Q8 2-d rep has real character (-2 at -1, 0 elsewhere) yet Schur
    # index 2 over R
    checks.append(schur_index("q8_2d") == 2)
    # S3 standard is realizable over R
    checks.append(schur_index("s3_std") == 1)
    # D8 standard 2-d is real
    checks.append(schur_index("d8_std") == 1)
    # index divides degree: 2 | 2
    checks.append(2 % schur_index("q8_2d") == 0)
    return float(sum(checks) / len(checks))


def bench_schur_index(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_index": _bench_schur_index(seed)}
